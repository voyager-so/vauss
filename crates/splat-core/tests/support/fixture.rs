use splat_contracts::JobSpec;
use splat_core::{Run, RunCtx};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};
static INTERRUPTED: AtomicBool = AtomicBool::new(false);
#[cfg(unix)]
extern "C" fn on_signal(_: libc::c_int) {
    INTERRUPTED.store(true, Ordering::SeqCst);
}
#[cfg(windows)]
unsafe extern "system" fn on_console_signal(event: u32) -> i32 {
    if event == 0 || event == 1 {
        INTERRUPTED.store(true, Ordering::SeqCst);
        1
    } else {
        0
    }
}
fn main() -> anyhow::Result<()> {
    if let Some(result) = splat_core::handle_internal_command() {
        match result {
            Ok(code) => std::process::exit(code),
            Err(e) => {
                eprintln!("{e:#}");
                std::process::exit(127);
            }
        }
    }
    let args: Vec<String> = std::env::args().collect();
    match args[1].as_str() {
        "dispatch" | "dispatch-crash" => {
            let mut registry = splat_core::Registry::new();
            registry.register_backend(std::sync::Arc::new(splat_core::FakeBackend::default()));
            let run = Run::open_with_registry(&args[2], std::sync::Arc::new(registry), true)?;
            let result = splat_core::dispatch(&run, &args[3], serde_json::from_str(&args[4])?);
            if args[1] == "dispatch-crash" {
                std::process::exit(73);
            }
            println!("{}", serde_json::to_string(&result)?);
        }
        "sleep" => {
            std::thread::sleep(Duration::from_millis(args[2].parse()?));
        }
        "exit" => std::process::exit(args[2].parse()?),
        "checkpoint" => {
            #[cfg(unix)]
            unsafe {
                libc::signal(libc::SIGINT, on_signal as *const () as libc::sighandler_t);
            }
            #[cfg(windows)]
            unsafe {
                windows_sys::Win32::System::Console::SetConsoleCtrlHandler(
                    Some(on_console_signal),
                    1,
                );
            }
            std::fs::write(Path::new(&args[2]).with_extension("ready"), "{}")?;
            while !INTERRUPTED.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(10));
            }
            std::thread::sleep(Duration::from_millis(150));
            std::fs::write(&args[2], "{\"saved\":true}")?;
        }
        "start-crash" => {
            let run = Run::open(&args[2])?;
            run.start_job(JobSpec {
                tool: "train_splats".into(),
                argv: vec![args[0].clone(), "sleep".into(), "300".into()],
                env: Default::default(),
                cwd: None,
                on_complete: "core:exit".into(),
                meta: Default::default(),
            })?;
            std::process::exit(73);
        }
        _ => anyhow::bail!("Unknown fixture action"),
    }
    Ok(())
}
