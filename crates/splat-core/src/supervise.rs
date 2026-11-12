//! A run's supervisor (driver, pipeline, job launcher) detached from the terminal that started it.
//!
//! `vauss run --cloud` used to supervise the run inside the launching process. When that process tree was
//! torn down (an agent session restart signals every descendant of its shell), the signal reached the
//! supervisor, which stopped the run and cancelled its fal requests, and the job workers, which died without
//! a completion marker (`job_owner_lost`), in one case two runs at 43.6k of 45k MAX steps.
//! Now the launching process starts the supervisor in its own session through a short-lived intermediate
//! (so its parent is no longer in the launching tree) and only follows its output. Ctrl-C there still stops
//! the run; any other end of the launching process (SIGTERM, SIGHUP, a closed pipe) leaves the run going.
//! `vauss status` / `splat logs --follow` watch it and `vauss stop` ends it.

use anyhow::{Context, Result, ensure};
use std::{
    ffi::OsString,
    fs,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Duration,
};

/// Set by the follower on the detached supervisor: where it writes its exit code. Removed from the
/// environment at startup ([`take_env`]), so the supervisor's own children never inherit it.
const EXIT_ENV: &str = "SPLAT_SUPERVISOR_EXIT";
/// `detached` / `attached`: overrides the default (detached for `--cloud` runs on Unix).
pub const MODE_ENV: &str = "SPLAT_SUPERVISOR";
/// Per-launch output of detached supervisors, under the run dir (never staged to fal).
pub const LOG_DIR: &str = "supervisor-logs";
/// Internal command of the intermediate process that starts the supervisor and exits.
pub const LAUNCH_COMMAND: &str = "__splat_supervise";

static EXIT_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Call first in `main`, while single-threaded: whether this process is a detached supervisor.
pub fn take_env() {
    EXIT_PATH.get_or_init(|| {
        let path = std::env::var_os(EXIT_ENV).map(PathBuf::from);
        // SAFETY: main thread, before any other thread starts.
        unsafe { std::env::remove_var(EXIT_ENV) };
        path
    });
}

/// True in the detached supervisor itself (never in the process that follows it).
pub fn is_detached() -> bool {
    EXIT_PATH.get().is_some_and(Option::is_some)
}

/// Whether a run command should hand its supervision to a detached process: `cloud` runs by default
/// (their fal jobs bill on with nobody to collect them), any run with SPLAT_SUPERVISOR=detached, none with
/// SPLAT_SUPERVISOR=attached. Never on Windows, and never inside the detached supervisor.
pub fn wanted(cloud: bool) -> bool {
    if !cfg!(unix) || is_detached() {
        return false;
    }
    match std::env::var(MODE_ENV).as_deref() {
        Ok("attached") => false,
        Ok("detached") => true,
        _ => cloud,
    }
}

/// The detached supervisor's last step: its exit code for the follower (after flushing its output).
pub fn record_exit(code: i32) {
    if let Some(Some(path)) = EXIT_PATH.get() {
        std::io::stdout().flush().ok();
        std::io::stderr().flush().ok();
        let tmp = path.with_extension("exit.tmp");
        if fs::write(&tmp, code.to_string()).is_ok() {
            fs::rename(&tmp, path).ok();
        }
    }
}

/// End a worker whose outputs are all written: flush, [`record_exit`], then `_exit`, which skips the C
/// exit handlers. `std::process::exit` still runs them, and with ORT's CUDA provider and the GPU
/// driver loaded their static destructors race the still-running pool threads: evaluate wrote its
/// result, then aborted in exit() with "corrupted double-linked list" (exit 134, job_8f0b376b188c4e0a).
pub fn exit_now(code: i32) -> ! {
    std::io::stdout().flush().ok();
    std::io::stderr().flush().ok();
    record_exit(code);
    #[cfg(unix)]
    // SAFETY: plain process exit after flushing; C stdio buffers are flushed first and no state is used after.
    unsafe {
        libc::fflush(std::ptr::null_mut());
        libc::_exit(code)
    }
    #[cfg(not(unix))]
    std::process::exit(code)
}

struct Paths {
    out: PathBuf,
    err: PathBuf,
    pid: PathBuf,
    exit: PathBuf,
}
fn paths(prefix: &Path) -> Paths {
    let with = |ext: &str| {
        let mut p = prefix.as_os_str().to_owned();
        p.push(ext);
        PathBuf::from(p)
    };
    Paths { out: with(".out"), err: with(".err"), pid: with(".pid"), exit: with(".exit") }
}

/// [`LAUNCH_COMMAND`] `<prefix> <args...>`: start this binary with `args` in its own session, output to
/// `<prefix>.out` / `.err`, record its pid in `<prefix>.pid`, and return at once. Exiting orphans the
/// supervisor, so it is reparented out of the launching process tree.
pub fn launch(prefix: &Path, args: &[OsString]) -> Result<i32> {
    let p = paths(prefix);
    let mut command = std::process::Command::new(std::env::current_exe()?);
    command
        .args(args)
        .env(EXIT_ENV, &p.exit)
        .stdin(std::process::Stdio::null())
        .stdout(fs::File::create(&p.out)?)
        .stderr(fs::File::create(&p.err)?);
    crate::detach_command(&mut command)?;
    let child = command.spawn().context("start the run's supervisor")?;
    crate::atomic_json(&p.pid, &child.id())?;
    Ok(0)
}

/// Run this command (`args`: this binary's arguments) as a detached supervisor of the run at `root`
/// and follow it until it exits: its stdout and stderr come through here and its exit code is returned.
#[cfg(unix)]
pub fn run_detached(root: &Path, args: &[OsString]) -> Result<i32> {
    let fresh = !root.exists();
    let dir = root.join(LOG_DIR);
    fs::create_dir_all(&dir)?;
    let stamp = chrono_stamp();
    let prefix = dir.join(format!("{stamp}-{}", &uuid::Uuid::new_v4().simple().to_string()[..8]));
    let status = std::process::Command::new(std::env::current_exe()?)
        .arg(LAUNCH_COMMAND)
        .arg(&prefix)
        .args(args)
        .stdin(std::process::Stdio::null())
        .status()
        .context("start the run's supervisor")?;
    ensure!(status.success(), "could not start the run's supervisor ({status})");
    let p = paths(&prefix);
    let pid: u32 = serde_json::from_slice(&fs::read(&p.pid)?)?;
    let identity = crate::jobs::process_identity(pid).ok();
    eprintln!(
        "Supervisor: pid {pid}, detached (output in {}). Ctrl-C stops the run; if this process ends any other \
         way the run continues (`vauss status`, `splat logs --follow`, `vauss stop`).",
        dir.display()
    );
    on_interrupt();
    let alive = || identity.as_ref().is_some_and(|i| crate::jobs::identity_matches(i).unwrap_or(false));
    let interrupt = || {
        // The supervisor's own pid only, while its identity still matches (its ctrlc handler stops the run).
        if alive() {
            // SAFETY: kill(2) has no memory-safety preconditions.
            unsafe { libc::kill(pid as i32, libc::SIGINT) };
        }
    };
    let code = follow(&p, &mut std::io::stdout(), &mut std::io::stderr(), &alive, interrupted_count, &interrupt,
        Duration::from_millis(200))?;
    // A run that never got past preflight leaves nothing but its (forwarded) log: remove the directory made for it.
    let only_log = || fs::read_dir(root).is_ok_and(|d| d.flatten().all(|e| e.file_name() == LOG_DIR));
    if fresh && code != 0 && only_log() {
        fs::remove_dir_all(root).ok();
    }
    Ok(code)
}

#[cfg(not(unix))]
pub fn run_detached(_root: &Path, _args: &[OsString]) -> Result<i32> {
    anyhow::bail!("a detached supervisor needs Unix")
}

#[cfg_attr(not(unix), allow(dead_code))]
fn chrono_stamp() -> String {
    crate::utc_now().chars().filter(|c| c.is_ascii_digit()).take(14).collect()
}

#[cfg_attr(not(unix), allow(dead_code))]
/// Copy what `path` gained since `offset` to `to`. False when `to` is closed.
fn pump(path: &Path, offset: &mut u64, to: &mut dyn Write) -> bool {
    let Ok(mut file) = fs::File::open(path) else { return true };
    if file.seek(SeekFrom::Start(*offset)).is_err() {
        return true;
    }
    let mut bytes = Vec::new();
    if file.read_to_end(&mut bytes).is_err() || bytes.is_empty() {
        return true;
    }
    *offset += bytes.len() as u64;
    to.write_all(&bytes).and_then(|()| to.flush()).is_ok()
}

#[cfg_attr(not(unix), allow(dead_code))]
/// Follow a detached supervisor: forward its output, pass interrupts on, and return its exit code once
/// it records one. 1 when it vanished without one (killed), or when this side's output closed (nobody
/// is reading; the supervisor goes on).
fn follow(p: &Paths, out: &mut dyn Write, err: &mut dyn Write, alive: impl Fn() -> bool,
          interrupts: impl Fn() -> usize, interrupt: impl Fn(), poll: Duration) -> Result<i32> {
    let (mut out_at, mut err_at, mut passed) = (0u64, 0u64, 0usize);
    loop {
        // Read before draining: everything written before the exit code is forwarded.
        let exit = fs::read_to_string(&p.exit).ok();
        let running = exit.is_none() && alive();
        if !pump(&p.out, &mut out_at, out) || !pump(&p.err, &mut err_at, err) {
            return Ok(1);
        }
        if let Some(code) = exit {
            return Ok(code.trim().parse().unwrap_or(1));
        }
        if !running {
            // It may have recorded its code between the two checks.
            if let Ok(code) = fs::read_to_string(&p.exit) {
                pump(&p.out, &mut out_at, out);
                pump(&p.err, &mut err_at, err);
                return Ok(code.trim().parse().unwrap_or(1));
            }
            writeln!(err, "The run's supervisor exited without an exit status (killed?); see {}", p.err.display()).ok();
            return Ok(1);
        }
        let n = interrupts();
        if n > passed {
            passed = n;
            interrupt();
        }
        std::thread::sleep(poll);
    }
}

#[cfg_attr(not(unix), allow(dead_code))]
static INTERRUPTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg_attr(not(unix), allow(dead_code))]
fn interrupted_count() -> usize {
    INTERRUPTS.load(std::sync::atomic::Ordering::SeqCst)
}

/// SIGINT only (Ctrl-C): counted and passed on to the supervisor. SIGTERM / SIGHUP keep their default
/// action, ending this follower while the run goes on.
#[cfg(unix)]
fn on_interrupt() {
    extern "C" fn count(_: libc::c_int) {
        INTERRUPTS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
    // SAFETY: the handler only touches an atomic (async-signal-safe).
    unsafe { libc::signal(libc::SIGINT, count as *const () as libc::sighandler_t) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    #[test]
    fn follow_forwards_output_and_exit_code() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let p = paths(&tmp.path().join("s"));
        fs::write(&p.out, "{\"status\":")?;
        fs::write(&p.err, "Run: x\n")?;
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let polls = AtomicUsize::new(0);
        let p2 = paths(&tmp.path().join("s"));
        let code = follow(&p, &mut out, &mut err, || {
            // The supervisor writes more, then its exit code, while being followed.
            if polls.fetch_add(1, Ordering::SeqCst) == 2 {
                fs::OpenOptions::new().append(true).open(&p2.out).unwrap().write_all(b"\"succeeded\"}\n").unwrap();
                fs::write(&p2.exit, "5").unwrap();
            }
            true
        }, || 0, || {}, Duration::from_millis(1))?;
        assert_eq!(code, 5);
        assert_eq!(String::from_utf8(out)?, "{\"status\":\"succeeded\"}\n");
        assert_eq!(String::from_utf8(err)?, "Run: x\n");
        Ok(())
    }

    #[test]
    fn follow_passes_interrupts_on_once_each() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let p = paths(&tmp.path().join("s"));
        let (sent, polls) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let exit = p.exit.clone();
        let code = follow(&p, &mut Vec::new(), &mut Vec::new(), || {
            let n = polls.fetch_add(1, Ordering::SeqCst);
            if n == 6 {
                fs::write(&exit, "130").unwrap();
            }
            true
        }, || if polls.load(Ordering::SeqCst) >= 2 { 1 } else { 0 }, || { sent.fetch_add(1, Ordering::SeqCst); },
            Duration::from_millis(1))?;
        assert_eq!(code, 130);
        assert_eq!(sent.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn follow_reports_a_supervisor_that_vanished() -> Result<()> {
        let tmp = tempfile::tempdir()?;
        let p = paths(&tmp.path().join("s"));
        fs::write(&p.err, "partial\n")?;
        let mut err = Vec::new();
        let code = follow(&p, &mut Vec::new(), &mut err, || false, || 0, || {}, Duration::from_millis(1))?;
        assert_eq!(code, 1);
        let err = String::from_utf8(err)?;
        assert!(err.starts_with("partial\n") && err.contains("without an exit status"), "{err}");
        Ok(())
    }

    #[test]
    fn follow_stops_when_its_output_is_closed() -> Result<()> {
        struct Closed;
        impl Write for Closed {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::ErrorKind::BrokenPipe.into())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let tmp = tempfile::tempdir()?;
        let p = paths(&tmp.path().join("s"));
        fs::write(&p.out, "progress\n")?;
        let checked = AtomicBool::new(false);
        let code = follow(&p, &mut Closed, &mut Vec::new(), || { checked.store(true, Ordering::SeqCst); true },
            || 0, || {}, Duration::from_millis(1))?;
        assert_eq!(code, 1);
        assert!(checked.load(Ordering::SeqCst));
        Ok(())
    }
}
