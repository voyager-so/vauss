//! AI drivers: harnesses that connect a model to the `vauss` MCP server for one run.
//!
//! - `claude`: `claude -p` on the local Claude Code login (no API key). Built-in tools, settings
//!   files, skills and other MCP servers are off; the model only sees the `vauss` tools.
//! - `codex`: `codex exec` with the same server injected via `-c` overrides, user config ignored,
//!   shell tools disabled, read-only sandbox.
//!
//! The outcome is whatever the agent passed to `finish` (persisted by core as
//! `agent/finish.json`); a session that ends without it is `failed`.

use std::{
    io::{BufRead, BufReader, Read, Write},
    path::Path,
    process::{Child, Command, Stdio},
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use anyhow::{Context, bail};
use serde::Serialize;
use serde_json::{Value, json};
use splat_contracts::RunStatus;
use splat_core::RunCtx;

pub const SYSTEM_PROMPT: &str = include_str!("prompt.md");
const SERVER_NAME: &str = "vauss";
/// job_status may block up to 600 s, and synchronous tools (evaluate: renders behind the shared
/// GPU lock, then LPIPS) have taken over 900 s on a busy host.
const TOOL_TIMEOUT_S: u64 = 3600;
/// Env the MCP server (and the jobs it starts) may need; codex does not forward the parent env.
const FORWARD_ENV: &[&str] = &[
    "PATH", "HOME", "USER", "LANG", "TMPDIR", "CUDA_VISIBLE_DEVICES", "XDG_RUNTIME_DIR", "SYSTEMROOT", "USERPROFILE",
    "APPDATA", "LOCALAPPDATA",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Driver {
    Claude,
    Codex,
}

impl Driver {
    pub fn name(self) -> &'static str {
        match self {
            Driver::Claude => "claude",
            Driver::Codex => "codex",
        }
    }

    pub fn default_model(self) -> &'static str {
        match self {
            Driver::Claude => "claude-opus-5-5",
            Driver::Codex => "gpt-6-astra",
        }
    }
}

impl FromStr for Driver {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> anyhow::Result<Self> {
        match s {
            "claude" => Ok(Driver::Claude),
            "codex" => Ok(Driver::Codex),
            _ => bail!("unknown driver {s:?} (expected claude or codex)"),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DriverOptions {
    pub model: Option<String>,
    /// low | medium | high | xhigh | max. Default medium.
    pub effort: Option<String>,
    /// MCP server argv. Default: this executable `serve --run-dir <root>`.
    pub server_cmd: Option<Vec<String>>,
    /// Set to stop the session: the harness runs in its own process group (so a terminal Ctrl-C
    /// does not reach it directly; the caller sets this instead), gets SIGINT, then SIGKILL
    /// after CANCEL_GRACE.
    pub cancel: Option<Arc<AtomicBool>>,
}

const CANCEL_GRACE: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, Serialize)]
pub struct DriverResult {
    pub driver: String,
    pub model: String,
    pub status: RunStatus,
    pub finish: Option<Value>,
    pub wall_s: f64,
    pub turns: Option<u32>,
    pub cost_usd: Option<f64>,
    pub usage: Value,
    pub error: Option<String>,
    /// Tool calls the agent made (any tool counts as having started work).
    pub tool_calls: u32,
    /// The harness could not start (not installed, not logged in, or failed before its first tool
    /// call): the caller runs the fixed pipeline instead. Never set for a cancelled session.
    pub unavailable: Option<String>,
}

/// Run one agent session to completion. Blocks. Safe to re-run after a crash: the agent
/// re-reads state through the tools.
pub fn run_driver(run: &dyn RunCtx, driver: Driver, opts: &DriverOptions) -> anyhow::Result<DriverResult> {
    let model = opts.model.clone().unwrap_or_else(|| driver.default_model().into());
    let effort = opts.effort.clone().unwrap_or_else(|| "medium".into());
    let server = match &opts.server_cmd {
        Some(cmd) => cmd.clone(),
        None => {
            let exe = std::env::current_exe().context("locate vauss executable")?;
            vec![exe.display().to_string(), "serve".into(), "--run-dir".into(), run.root().display().to_string()]
        }
    };
    let res = DriverResult {
        driver: driver.name().into(),
        model: model.clone(),
        status: RunStatus::Failed,
        finish: None,
        wall_s: 0.0,
        turns: None,
        cost_usd: None,
        usage: json!({}),
        error: None,
        tool_calls: 0,
        unavailable: None,
    };
    let argv = match driver {
        Driver::Claude => claude_argv(run, &model, &effort, &server),
        Driver::Codex => codex_argv(run, &model, &effort, &server),
    };
    run_argv(run, driver, &argv, opts, res)
}

fn run_argv(run: &dyn RunCtx, driver: Driver, argv: &[String], opts: &DriverOptions, mut res: DriverResult) -> anyhow::Result<DriverResult> {
    let t0 = Instant::now();
    if let Some(reason) = preflight(driver, &argv[0]) {
        res.error = Some(format!("{} {reason}", driver.name()));
        res.unavailable = Some(reason);
    } else if let Err(e) = stream(run, driver, argv, opts.cancel.as_deref(), &mut res) {
        res.error = Some(format!("{e:#}"));
    }
    res.wall_s = t0.elapsed().as_secs_f64();
    let finish_path = run.root().join("agent").join("finish.json");
    if let Ok(text) = std::fs::read_to_string(&finish_path) {
        let finish: Value = serde_json::from_str(&text).context("parse agent/finish.json")?;
        res.status = serde_json::from_value(finish["status"].clone()).context("finish.json status")?;
        res.finish = Some(finish);
    } else if res.error.is_none() {
        res.error = Some("agent ended without calling finish".into());
    }
    let cancelled = opts.cancel.as_deref().is_some_and(|c| c.load(Ordering::SeqCst)) || res.error.as_deref() == Some("cancelled");
    if res.finish.is_none() && res.tool_calls == 0 && !cancelled && res.unavailable.is_none() {
        res.unavailable = Some(unavailable_reason(res.error.as_deref().unwrap_or("failed at startup")));
    }
    run.journal("driver_finished", serde_json::to_value(&res)?)?;
    eprintln!(
        "[{}] outcome: {}{}",
        res.driver,
        serde_json::to_value(res.status)?.as_str().unwrap_or("?"),
        res.error.as_deref().map(|e| format!(" ({e})")).unwrap_or_default()
    );
    Ok(res)
}

/// Messages that mean the harness has no usable login (Claude Code: "Not logged in · Please run
/// /login"; Codex: "codex login").
const AUTH_PATTERNS: &[&str] = &[
    "not logged in", "please run /login", "/login", "codex login", "login required", "invalid api key",
    "authentication_error", "authentication failed", "unauthorized", "oauth token",
];

/// Plain-English reason for a harness that failed before doing any work.
fn unavailable_reason(error: &str) -> String {
    let lower = error.to_lowercase();
    if lower.contains("spawn ") && (lower.contains("no such file") || lower.contains("not found")) {
        "not installed".into()
    } else if AUTH_PATTERNS.iter().any(|p| lower.contains(p)) {
        format!("not logged in: {}", short(error, 160))
    } else {
        short(error, 200)
    }
}

/// Cheap check before spawning: is the harness installed and logged in? `claude auth status` and
/// `codex login status` exit non-zero without a login. Inconclusive answers (timeout, unknown
/// subcommand on an old CLI) count as available; the session itself is still watched for a
/// startup failure.
fn preflight(driver: Driver, bin: &str) -> Option<String> {
    let args: &[&str] = match driver {
        Driver::Claude => &["auth", "status"],
        Driver::Codex => &["login", "status"],
    };
    let mut child = match Command::new(bin).args(args).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Some("not installed".into()),
        Err(_) => return None,
    };
    let deadline = Instant::now() + Duration::from_secs(20);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(50)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };
    let mut text = String::new();
    for pipe in [child.stdout.take().map(|p| Box::new(p) as Box<dyn std::io::Read>), child.stderr.take().map(|p| Box::new(p) as Box<dyn std::io::Read>)] {
        if let Some(mut p) = pipe {
            let _ = p.read_to_string(&mut text);
        }
    }
    let lower = text.to_lowercase();
    let logged_out = lower.contains("\"loggedin\": false") || lower.contains("\"loggedin\":false") || lower.contains("not logged in");
    (logged_out || (!status.success() && AUTH_PATTERNS.iter().any(|p| lower.contains(p)))).then(|| "not logged in".into())
}

pub fn user_prompt(run: &dyn RunCtx) -> String {
    let name = run.source().file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let goal = if run.goal().trim().is_empty() {
        "A viewable splat of the main subject."
    } else {
        run.goal()
    };
    let mut prompt = format!(
        "Source video: {name}\nGoal: {goal}\n\nCall `run_default_pipeline` FIRST (normally with {{}}). Then add only goal-requested extras on its candidates. \
         Look at the held-out comparison images before you finish, and end with exactly one `finish` call."
    );
    if let Some(options) = std::fs::read(run.root().join("fixed.json")).ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .filter(|v| !v["preset"].is_null() || v["steps_explicit"] == true || v["cap_max_explicit"] == true)
    {
        prompt.push_str(&format!(
            "\n\nResolved CLI options: {options}\n\
             run_default_pipeline inherits these settings automatically; call it with no overrides unless \
             the goal explicitly requests different settings. Do not recreate the pipeline stage by stage. \
             A fast preview finishes partial with the unevaluated-preview limitation."
        ));
    }
    prompt
}

/// Runaway guard so a looping agent ends on its own: Claude's --max-turns, and for Codex (which has no
/// turn flag) the number of tool calls. SPLAT_AGENT_MAX_TURNS overrides.
pub const DEFAULT_MAX_TURNS: u32 = 300;

fn max_turns() -> u32 {
    std::env::var("SPLAT_AGENT_MAX_TURNS").ok().and_then(|v| v.parse().ok()).filter(|&n| n > 0).unwrap_or(DEFAULT_MAX_TURNS)
}

pub fn claude_argv(run: &dyn RunCtx, model: &str, effort: &str, server: &[String]) -> Vec<String> {
    let mcp = json!({"mcpServers": {SERVER_NAME: {"type": "stdio", "command": server[0], "args": &server[1..]}}});
    let args = [
        "-p", &user_prompt(run),
        "--output-format", "stream-json", "--verbose",
        "--model", model, "--effort", effort,
        "--system-prompt", SYSTEM_PROMPT,
        "--tools", "",
        "--mcp-config", &mcp.to_string(), "--strict-mcp-config",
        "--allowedTools", &format!("mcp__{SERVER_NAME}"),
        "--permission-mode", "dontAsk",
        "--setting-sources", "",
        "--disable-slash-commands",
        "--max-turns", &max_turns().to_string(),
    ]; // fmt: skip
    std::iter::once("claude").chain(args).map(String::from).collect()
}

/// TOML literal for a `-c key=value` override: JSON strings/arrays/numbers are valid TOML;
/// objects become inline tables.
fn toml(v: &Value) -> String {
    match v {
        Value::Object(m) => {
            let items: Vec<String> = m.iter().map(|(k, v)| format!("{}={}", Value::from(k.as_str()), toml(v))).collect();
            format!("{{{}}}", items.join(","))
        }
        other => other.to_string(),
    }
}

pub fn codex_argv(run: &dyn RunCtx, model: &str, effort: &str, server: &[String]) -> Vec<String> {
    let env: serde_json::Map<String, Value> = std::env::vars()
        .filter(|(k, _)| FORWARD_ENV.contains(&k.as_str()) || k.starts_with("SPLAT_"))
        .map(|(k, v)| (k, Value::from(v)))
        .collect();
    let srv = format!("mcp_servers.{SERVER_NAME}");
    let overrides = [
        (format!("{srv}.command"), json!(server[0])),
        (format!("{srv}.args"), json!(&server[1..])),
        (format!("{srv}.env"), Value::Object(env)),
        (format!("{srv}.startup_timeout_sec"), json!(60)),
        (format!("{srv}.tool_timeout_sec"), json!(TOOL_TIMEOUT_S)),
        (format!("{srv}.default_tools_approval_mode"), json!("approve")),
        ("approval_policy".into(), json!("never")),
        ("model_reasoning_effort".into(), json!(effort)),
        ("developer_instructions".into(), json!(SYSTEM_PROMPT)),
    ];
    let mut argv: Vec<String> = [
        "codex", "exec", "--json", "--ignore-user-config", "--skip-git-repo-check",
        "--sandbox", "read-only", "--cd", &run.root().display().to_string(), "--model", model,
        "--disable", "shell_tool", "--disable", "unified_exec",
    ].map(String::from).into(); // fmt: skip
    for (k, v) in overrides {
        argv.push("-c".into());
        argv.push(format!("{k}={}", toml(&v)));
    }
    argv.push(user_prompt(run));
    argv
}

fn stream(
    run: &dyn RunCtx,
    driver: Driver,
    argv: &[String],
    cancel: Option<&AtomicBool>,
    res: &mut DriverResult,
) -> anyhow::Result<()> {
    let agent_dir = run.root().join("agent");
    std::fs::create_dir_all(&agent_dir)?;
    let name = driver.name();
    let stderr = std::fs::File::create(agent_dir.join(format!("{name}.stderr.log")))?;
    let mut events = std::fs::File::create(agent_dir.join(format!("{name}.events.jsonl")))?;
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..]).current_dir(run.root()).stdin(Stdio::null()).stdout(Stdio::piped()).stderr(stderr);
    #[cfg(unix)]
    std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
    // The harness runs in its own process group, so a signal to `vauss run` never reaches it.
    // If we die without cancelling (SIGKILL, panic), the kernel SIGTERMs the harness; its stdio
    // MCP server then sees EOF and exits.
    #[cfg(target_os = "linux")]
    unsafe {
        std::os::unix::process::CommandExt::pre_exec(&mut cmd, || {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
            Ok(())
        });
    }
    if driver == Driver::Claude {
        cmd.env("MCP_TOOL_TIMEOUT", (TOOL_TIMEOUT_S * 1000).to_string()).env("MCP_TIMEOUT", "60000");
    }
    let mut child = cmd.spawn().with_context(|| format!("spawn {} (is it installed and on PATH?)", argv[0]))?;
    let stdout = child.stdout.take().expect("piped stdout");
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut state = StreamState::default();
    loop {
        let line = match rx.recv_timeout(Duration::from_millis(500)) {
            Ok(line) => line?,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if cancel.is_some_and(|c| c.load(Ordering::SeqCst)) {
                    eprintln!("[{name}] cancelled; interrupting");
                    res.error = Some("cancelled".into());
                    interrupt(&mut child)?;
                    return Ok(());
                }
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        writeln!(events, "{line}")?;
        let Ok(ev) = serde_json::from_str::<Value>(&line) else { continue };
        let out = match driver {
            Driver::Claude => claude_event(&ev, res),
            Driver::Codex => codex_event(&ev, res, &mut state),
        };
        for o in out {
            match o {
                Out::Log(s) => eprintln!("[{name}] {s}"),
                Out::Journal(event, data) => run.journal(event, data)?,
            }
        }
        if state.tool_calls > max_turns() {
            eprintln!("[{name}] {} tool calls exceeds the cap of {}; interrupting", state.tool_calls, max_turns());
            res.error = Some(format!("error_max_turns: {} tool calls exceeded the cap of {}", state.tool_calls, max_turns()));
            interrupt(&mut child)?;
            return Ok(());
        }
    }
    let code = child.wait()?;
    if !code.success() && res.error.is_none() {
        res.error = Some(format!("{name} exited {code}; see agent/{name}.stderr.log"));
    }
    Ok(())
}

/// SIGINT the harness's process group (harness + the MCP server it spawned), SIGKILL after
/// CANCEL_GRACE, then reap.
fn interrupt(child: &mut Child) -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        let pgid = -(child.id() as i32);
        unsafe { libc::kill(pgid, libc::SIGINT) };
        let deadline = Instant::now() + CANCEL_GRACE;
        while Instant::now() < deadline {
            if child.try_wait()?.is_some() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        unsafe { libc::kill(pgid, libc::SIGKILL) };
    }
    #[cfg(not(unix))]
    child.kill()?;
    child.wait()?;
    Ok(())
}

enum Out {
    Log(String),
    Journal(&'static str, Value),
}

#[derive(Default)]
struct StreamState {
    usage: serde_json::Map<String, Value>,
    tool_calls: u32,
}

fn short(s: &str, n: usize) -> String {
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if s.chars().count() <= n { s } else { format!("{}…", s.chars().take(n - 1).collect::<String>()) }
}

/// One line from a tool result's text blocks: status, job, error, a few headline metrics.
fn result_brief(texts: &[&str], n_images: usize) -> String {
    let Some(r) = texts.iter().find_map(|t| serde_json::from_str::<Value>(t).ok().filter(|v| v.get("status").is_some()))
    else {
        return short(&texts.join(" "), 160);
    };
    let mut parts = vec![r["status"].as_str().unwrap_or("?").to_string()];
    if let Some(j) = r["job_id"].as_str() {
        parts.push(j.into());
    }
    if let Some(e) = r.get("error").filter(|e| !e.is_null()) {
        parts.push(format!("{}: {}", e["code"].as_str().unwrap_or("?"), short(e["message"].as_str().unwrap_or(""), 120)));
    }
    if let Some(m) = r["metrics"].as_object() {
        parts.extend(m.iter().take(6).map(|(k, v)| format!("{k}={}", short(&v.to_string(), 40))));
    }
    if n_images > 0 {
        parts.push(format!("+{n_images} images"));
    }
    parts.join(" ")
}

/// (texts, image count) from MCP content blocks.
fn blocks(content: &Value) -> (Vec<&str>, usize) {
    match content {
        Value::String(s) => (vec![s.as_str()], 0),
        Value::Array(items) => (
            items.iter().filter(|b| b["type"] == "text").filter_map(|b| b["text"].as_str()).collect(),
            items.iter().filter(|b| b["type"] == "image").count(),
        ),
        _ => (vec![], 0),
    }
}

fn tool_name(name: &str) -> &str {
    name.strip_prefix("mcp__vauss__").or_else(|| name.strip_prefix("mcp__splat__")).unwrap_or(name)
}

/// `claude -p --output-format stream-json --verbose` events.
fn claude_event(ev: &Value, res: &mut DriverResult) -> Vec<Out> {
    let mut out = vec![];
    match ev["type"].as_str() {
        Some("system") if ev["subtype"] == "init" => {
            let servers: serde_json::Map<String, Value> = ev["mcp_servers"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|s| (s["name"].as_str().unwrap_or("?").to_string(), s["status"].clone()))
                .collect();
            out.push(Out::Log(format!("session {} model={} mcp={}", ev["session_id"], ev["model"], Value::Object(servers.clone()))));
            if servers.get(SERVER_NAME).and_then(Value::as_str) != Some("connected") {
                res.error = Some(format!("MCP server {SERVER_NAME} not connected: {}", Value::Object(servers.clone())));
            }
            out.push(Out::Journal(
                "driver_started",
                json!({"driver": "claude", "model": res.model, "returned_model": ev["model"], "session_id": ev["session_id"], "mcp": servers}),
            ));
        }
        Some("assistant") => {
            for b in ev["message"]["content"].as_array().into_iter().flatten() {
                match b["type"].as_str() {
                    Some("text") if !b["text"].as_str().unwrap_or("").trim().is_empty() => {
                        let text = b["text"].as_str().unwrap_or("");
                        out.push(Out::Log(short(text, 240)));
                        out.push(Out::Journal("agent_text", json!({"driver": "claude", "text": short(text, 4000)})));
                    }
                    Some("tool_use") => {
                        res.tool_calls += 1;
                        out.push(Out::Log(format!(
                        "→ {} {}",
                        tool_name(b["name"].as_str().unwrap_or("?")),
                        short(&b["input"].to_string(), 160)
                    )))
                    }
                    _ => {}
                }
            }
        }
        Some("user") => {
            for b in ev["message"]["content"].as_array().into_iter().flatten().filter(|b| b["type"] == "tool_result") {
                let (texts, n) = blocks(&b["content"]);
                out.push(Out::Log(format!("← {}", result_brief(&texts, n))));
            }
        }
        Some("result") => {
            res.turns = ev["num_turns"].as_u64().map(|n| n as u32);
            res.cost_usd = ev["total_cost_usd"].as_f64();
            res.usage = ev["usage"].clone();
            let subtype = ev["subtype"].as_str().unwrap_or("");
            if (ev["is_error"] == true || subtype.starts_with("error")) && res.error.is_none() {
                res.error = Some(format!("{}: {}", ev["subtype"].as_str().unwrap_or("error"), short(&ev["result"].to_string(), 400)));
            }
            out.push(Out::Log(format!(
                "done: {}, {} turns, ${:.2}",
                ev["subtype"].as_str().unwrap_or("?"),
                res.turns.unwrap_or(0),
                res.cost_usd.unwrap_or(0.0)
            )));
        }
        _ => {}
    }
    out
}

/// `codex exec --json` events.
fn codex_event(ev: &Value, res: &mut DriverResult, state: &mut StreamState) -> Vec<Out> {
    let mut out = vec![];
    let item = &ev["item"];
    match (ev["type"].as_str().unwrap_or(""), item["type"].as_str().unwrap_or("")) {
        ("thread.started", _) => {
            out.push(Out::Log(format!("thread {}", ev["thread_id"])));
            out.push(Out::Journal("driver_started", json!({"driver": "codex", "model": res.model, "thread_id": ev["thread_id"]})));
        }
        ("item.started", "mcp_tool_call") => {
            state.tool_calls += 1;
            res.tool_calls += 1;
            out.push(Out::Log(format!("→ {} {}", item["tool"].as_str().unwrap_or("?"), short(&item["arguments"].to_string(), 160))))
        }
        ("item.completed", "mcp_tool_call") => {
            let (texts, n) = blocks(&item["result"]["content"]);
            let brief = if texts.is_empty() {
                format!("error: {}", item["error"])
            } else {
                result_brief(&texts, n)
            };
            out.push(Out::Log(format!("← {brief}")));
        }
        ("item.completed", "agent_message") if !item["text"].as_str().unwrap_or("").trim().is_empty() => {
            let text = item["text"].as_str().unwrap_or("");
            out.push(Out::Log(short(text, 240)));
            out.push(Out::Journal("agent_text", json!({"driver": "codex", "text": short(text, 4000)})));
        }
        ("item.completed", t @ ("command_execution" | "file_change" | "web_search")) => {
            out.push(Out::Log(format!("unexpected {t}: {}", short(&item.to_string(), 160))))
        }
        ("turn.completed", _) => {
            res.turns = Some(res.turns.unwrap_or(0) + 1);
            for (k, v) in ev["usage"].as_object().into_iter().flatten() {
                if let Some(n) = v.as_f64() {
                    let prev = state.usage.get(k).and_then(Value::as_f64).unwrap_or(0.0);
                    state.usage.insert(k.clone(), json!(prev + n));
                }
            }
            res.usage = Value::Object(state.usage.clone());
        }
        (t @ ("turn.failed" | "error"), _) => {
            let detail = if ev["error"].is_null() { ev["message"].clone() } else { ev["error"].clone() };
            res.error = Some(short(&detail.to_string(), 400));
            out.push(Out::Log(format!("{t}: {}", res.error.as_deref().unwrap_or(""))));
        }
        _ => {}
    }
    out
}

/// Where `finish` lands; core's finish builtin writes it.
pub fn finish_path(root: &Path) -> std::path::PathBuf {
    root.join("agent").join("finish.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brief_finds_status_json_among_texts() {
        let t = r#"{"status":"succeeded","job_id":null,"metrics":{"psnr":24.9},"error":null}"#;
        assert_eq!(result_brief(&["[Image: x]", t], 2), "succeeded psnr=24.9 +2 images");
    }

    #[test]
    fn toml_inline_table() {
        assert_eq!(toml(&json!({"A": "b\"c", "N": 1})), r#"{"A"="b\"c","N"=1}"#);
    }

    /// Minimal run for driving a fake harness: journal lines are dropped.
    struct FakeRun(tempfile::TempDir);
    impl RunCtx for FakeRun {
        fn root(&self) -> &Path { self.0.path() }
        fn run_id(&self) -> &str { "run_test" }
        fn source(&self) -> &Path { Path::new("room.mp4") }
        fn goal(&self) -> &str { "" }
        fn new_artifact_dir(&self, _: splat_contracts::ArtifactKind) -> anyhow::Result<(String, std::path::PathBuf)> { bail!("unused") }
        fn register(&self, _: splat_contracts::Artifact) -> anyhow::Result<()> { bail!("unused") }
        fn artifact(&self, _: &str) -> anyhow::Result<splat_contracts::Artifact> { bail!("unused") }
        fn path(&self, _: &str) -> anyhow::Result<std::path::PathBuf> { bail!("unused") }
        fn journal(&self, _: &str, _: Value) -> anyhow::Result<()> { Ok(()) }
        fn start_job(&self, _: splat_contracts::JobSpec) -> anyhow::Result<String> { bail!("unused") }
    } // fmt: skip

    /// Runs a fake `claude` (a shell script; `auth status` exits `auth_exit`) and returns the result.
    #[cfg(unix)]
    fn fake_claude(auth_exit: i32, session: &str) -> DriverResult {
        use std::os::unix::fs::PermissionsExt;
        let run = FakeRun(tempfile::tempdir().unwrap());
        let bin = run.root().join("fake-claude");
        let script = format!("#!/bin/sh\nif [ \"$1\" = auth ]; then echo '{{\"loggedIn\": {}}}'; exit {auth_exit}; fi\n{session}\n", auth_exit == 0);
        std::fs::write(&bin, script).unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        let res = DriverResult {
            driver: "claude".into(), model: "m".into(), status: RunStatus::Failed, finish: None, wall_s: 0.0,
            turns: None, cost_usd: None, usage: json!({}), error: None, tool_calls: 0, unavailable: None,
        }; // fmt: skip
        run_argv(&run, Driver::Claude, &[bin.display().to_string()], &DriverOptions::default(), res).unwrap()
    }

    const NOT_LOGGED_IN: &str = r#"echo '{"type":"result","subtype":"success","is_error":true,"result":"Not logged in · Please run /login"}'; exit 1"#;

    #[cfg(unix)]
    #[test]
    fn not_logged_in_falls_back_but_mid_run_failure_does_not() {
        // Preflight: `claude auth status` says logged out, so the session never starts.
        let r = fake_claude(1, "exit 99");
        assert_eq!(r.unavailable.as_deref(), Some("not logged in"));
        // Preflight inconclusive, then the session itself fails at startup with the login message.
        let r = fake_claude(0, NOT_LOGGED_IN);
        assert!(r.unavailable.as_deref().unwrap().starts_with("not logged in"), "{r:?}");
        // The agent called a tool before dying: today's failure, no fallback.
        let mid = r#"echo '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"mcp__splat__extract_frames","input":{}}]}}'; exit 1"#;
        let r = fake_claude(0, mid);
        assert_eq!((r.tool_calls, r.unavailable.is_none(), r.error.is_some()), (1, true, true), "{r:?}");
        // Not installed.
        let run = FakeRun(tempfile::tempdir().unwrap());
        let res = DriverResult { tool_calls: 0, unavailable: None, ..r };
        let r = run_argv(&run, Driver::Claude, &["/nonexistent/claude".into()], &DriverOptions::default(), res).unwrap();
        assert_eq!(r.unavailable.as_deref(), Some("not installed"));
    }

    #[test]
    fn tool_name_strips_prefix() {
        assert_eq!(tool_name("mcp__splat__finish"), "finish");
        assert_eq!(tool_name("mcp__vauss__finish"), "finish");
        assert_eq!(tool_name("finish"), "finish");
    }
}
