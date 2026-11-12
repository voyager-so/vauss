//! `dry_run <claude|codex> <run_dir> <fake_server_exe> [model]`: a real driver session against the
//! fake MCP server (`cargo build -p splat-mcp --example fake_server`).

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use splat_agent::{DriverOptions, run_driver};
use splat_core::Run;

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (driver, root, server) = (args[1].parse()?, std::path::PathBuf::from(&args[2]), &args[3]);
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root)?;
    let source = root.with_extension("source.webm");
    std::fs::write(&source, b"fake")?;
    let run = Run::create(&root, &source, "a viewable splat and a textured mesh of the statue", "fake")?;
    let opts = DriverOptions {
        model: args.get(4).cloned(),
        effort: Some("low".into()),
        server_cmd: Some(vec![server.clone(), "--run-dir".into(), root.display().to_string()]),
        cancel: Some(Arc::new(AtomicBool::new(false))),
    };
    // DRY_CANCEL_AFTER_S=N exercises cancellation.
    if let Some(secs) = std::env::var("DRY_CANCEL_AFTER_S").ok().and_then(|s| s.parse().ok()) {
        let flag = opts.cancel.clone().unwrap();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_secs(secs));
            flag.store(true, Ordering::SeqCst);
        });
    }
    let res = run_driver(&run, driver, &opts)?;
    println!("{}", serde_json::to_string_pretty(&res)?);
    Ok(())
}
