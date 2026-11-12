//! Render the same journal events consumed by front-ends, including agent/MCP subprocesses.
use serde_json::Value;
use splat_core::{Run, RunCtx};
use std::{fs::File, io::{BufRead, BufReader, Seek, SeekFrom}, sync::{Arc, atomic::{AtomicBool, Ordering}}, thread, time::{Duration, Instant}};

pub struct Reporter {
    done: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}
impl Reporter {
    pub fn start(run: &Run) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let stop = done.clone();
        let run = run.clone();
        // Do not replay the old run's output on resume/refine.
        let offset = std::fs::metadata(run.root().join("journal.jsonl")).map(|m| m.len()).unwrap_or(0);
        let thread = thread::spawn(move || {
            let mut offset = offset;
            let mut last_tick = Instant::now();
            loop {
                let finished = stop.load(Ordering::Relaxed);
                if last_tick.elapsed() >= Duration::from_secs(2) && !finished {
                    if let Err(e) = splat_core::progress::refresh(&run) {
                        eprintln!("Progress unavailable: {e:#}");
                    }
                    last_tick = Instant::now();
                }
                if let Ok(mut file) = File::open(run.root().join("journal.jsonl")) {
                    let _ = file.seek(SeekFrom::Start(offset));
                    let mut reader = BufReader::new(file);
                    let mut line = String::new();
                    while reader.read_line(&mut line).is_ok_and(|n| n > 0) {
                        if !line.ends_with('\n') { break; }
                        offset += line.len() as u64;
                        if let Ok(e) = serde_json::from_str::<Value>(&line) {
                            if e["event"] == "progress" { render(&e["data"]); }
                        }
                        line.clear();
                    }
                }
                if finished { break; }
                thread::sleep(Duration::from_millis(100));
            }
        });
        Self { done, thread: Some(thread) }
    }
}
impl Drop for Reporter {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() { let _ = t.join(); }
    }
}
fn duration(seconds: f64) -> String {
    let seconds = seconds.ceil().max(0.) as u64;
    if seconds >= 60 { format!("{}m{:02}s", seconds / 60, seconds % 60) } else { format!("{seconds}s") }
}
fn render(p: &Value) {
    let mut detail = String::new();
    if let (Some(step), Some(total)) = (p["step"].as_u64(), p["total_steps"].as_u64()) {
        if let Some(fraction) = p["fraction"].as_f64() {
            detail += &format!(" {:.1}% ·", fraction * 100.);
        }
        detail += &format!(" step {step}/{total}");
        if let Some(n) = p["splats"].as_u64() { detail += &format!(" · {n} splats"); }
        if let Some(rate) = p["it_s"].as_f64() { detail += &format!(" · {rate:.1} it/s"); }
        detail += ",";
    }
    let elapsed = duration(p["elapsed_s"].as_f64().unwrap_or(0.));
    let status = p["status"].as_str().unwrap_or("running");
    let tail = if status != "running" { status.to_string() }
        else if let Some(eta) = p["eta_s"].as_f64() {
            format!("~{} left ({})", duration(eta), if p["eta_source"] == "model" { "estimate; compute only" } else { "measured" })
        } else { "ETA unknown".into() };
    let remote = p["remote_status"].as_str().map(|s| format!(", cloud {s}")).unwrap_or_default();
    let note = p["note"].as_str().map(|s| format!(" — {s}")).unwrap_or_default();
    eprintln!("[{}/{}] {} …{detail} {elapsed}, {tail}{remote}{note}",
        p["index"], p["stages"], p["label"].as_str().unwrap_or("Working"));
}
