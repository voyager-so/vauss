//! `fake_server --run-dir R`: the splat MCP server over a fake backend, for driver dry runs.

mod dispatch;

use std::sync::Arc;

use splat_core::{Run, RunCtx};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let root = args.iter().position(|a| a == "--run-dir").and_then(|i| args.get(i + 1)).expect("--run-dir R");
    let run: Arc<dyn RunCtx> = Arc::new(Run::open(root)?);
    splat_mcp::serve(run.clone(), dispatch::fake_dispatch(run))
}
