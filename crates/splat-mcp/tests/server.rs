#[path = "../examples/fake_server/dispatch.rs"]
mod dispatch;

use std::sync::Arc;

use image::{Rgb, RgbImage};
use rmcp::{
    ServiceExt,
    model::{CallToolRequestParams, ContentBlock},
};
use serde_json::{Value, json};
use splat_contracts::{Status, ToolResult};
use splat_core::{Run, RunCtx};
use splat_mcp::{SplatServer, result_content};

fn new_run(name: &str) -> Arc<dyn RunCtx> {
    let dir = std::env::temp_dir().join(format!("splat-mcp-test-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("video.mp4");
    std::fs::write(&src, b"not really a video").unwrap();
    Arc::new(Run::create(dir.join("run"), &src, "the statue", "fake").unwrap())
}

fn call(name: &str, args: Value) -> CallToolRequestParams {
    CallToolRequestParams::new(name.to_string()).with_arguments(args.as_object().unwrap().clone())
}

fn first_json(content: &[ContentBlock]) -> Value {
    let ContentBlock::Text(t) = &content[0] else { panic!("first block is not text") };
    serde_json::from_str(&t.text).unwrap()
}

#[tokio::test]
async fn tools_images_jobs_and_finish_over_mcp() -> anyhow::Result<()> {
    let run = new_run("e2e");
    let (server_io, client_io) = tokio::io::duplex(1 << 20);
    let server = SplatServer::new(run.clone(), dispatch::fake_dispatch(run.clone()));
    tokio::spawn(async move { server.serve(server_io).await.unwrap().waiting().await });
    let client = ().serve(client_io).await?;

    let tools = client.list_all_tools().await?;
    assert_eq!(tools.len(), splat_contracts::tool_schemas().len());
    let train = tools.iter().find(|t| t.name == "train_splats").unwrap();
    assert_eq!(train.input_schema["additionalProperties"], json!(false));
    let all = serde_json::to_string(&tools)?;
    for banned in ["$ref", "definitions", "allOf", "uint32", "\"items\":["] {
        assert!(!all.contains(banned), "tool schemas contain {banned}");
    }
    let extract = tools.iter().find(|t| t.name == "extract_frames").unwrap();
    assert_eq!(extract.input_schema["properties"]["strategy"]["enum"], json!(["uniform", "sharpness_window"]));

    let inspect = client.call_tool(call("inspect_input", json!({"sample_count": 3}))).await?;
    assert_eq!(first_json(&inspect.content)["status"], "succeeded");
    assert_eq!(inspect.content.iter().filter(|b| matches!(b, ContentBlock::Image(_))).count(), 6);

    let started = client.call_tool(call("reconstruct_sfm", json!({"frames_id": "x"}))).await?;
    let job_id = first_json(&started.content)["job_id"].as_str().unwrap().to_string();
    let done = client.call_tool(call("job_status", json!({"job_id": job_id, "wait_s": 1}))).await?;
    assert_eq!(first_json(&done.content)["status"], "succeeded");

    let unknown = client.call_tool(call("nope", json!({}))).await?;
    assert_eq!(unknown.is_error, Some(true));

    client.call_tool(call("finish", json!({"status": "refused", "explanation": "pure rotation"}))).await?;
    let finish: Value = serde_json::from_slice(&std::fs::read(run.root().join("agent/finish.json"))?)?;
    assert_eq!(finish["status"], "refused");
    client.cancel().await?;
    Ok(())
}

#[test]
fn many_large_images_fit_the_message_budget() {
    let run = new_run("budget");
    let mut seed = 7u32;
    let mut images = vec![];
    for i in 0..12 {
        let img = RgbImage::from_fn(1920, 1080, |_, _| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            Rgb([(seed >> 24) as u8, (seed >> 16) as u8, (seed >> 8) as u8])
        });
        let name = format!("noise_{i}.png");
        img.save(run.root().join(&name)).unwrap();
        images.push(name);
    }
    let mut result: ToolResult = serde_json::from_value(json!({"schema_version": 1, "tool": "inspect_input",
        "status": "succeeded", "job_id": null, "error": null, "wall_s": null})).unwrap();
    result.images = images;
    assert_eq!(result.status, Status::Succeeded);
    let content = result_content(run.root(), &result);
    let total: usize = content.iter().map(|b| match b { ContentBlock::Image(i) => i.data.len(), _ => 0 }).sum();
    assert_eq!(content.iter().filter(|b| matches!(b, ContentBlock::Image(_))).count(), 12);
    assert!(total < 800_000, "{total} base64 bytes");
}
