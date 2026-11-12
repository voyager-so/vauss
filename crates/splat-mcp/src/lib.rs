//! stdio MCP server: the one tool surface every AI driver uses, bound to a single run.
//!
//! Lists every tool in `splat_contracts::tool_schemas()` and forwards every call, including the
//! `finish`/`record_decision` builtins, to the caller's dispatch (`splat_core::dispatch` in the
//! CLI; it persists `agent/finish.json`). Images named in a ToolResult come back as MCP image
//! content so the model actually sees renders. stdout is the protocol channel; log to stderr.

use std::{path::Path, sync::Arc};

use anyhow::Context;
use base64::Engine;
use image::{DynamicImage, ImageFormat, imageops::FilterType};
use rmcp::{
    ErrorData, RoleServer, ServerHandler, ServiceExt,
    model::{
        CallToolRequestParams, CallToolResponse, CallToolResult, ContentBlock, Implementation, ListToolsResult,
        PaginatedRequestParams, ServerCapabilities, ServerConfig, Tool,
    },
    service::RequestContext,
};
use serde_json::Value;
use splat_contracts::{Status, ToolError, ToolResult, tool_schemas};
use splat_core::RunCtx;

/// Claude's vision long-edge sweet spot; larger inputs get downscaled by the model anyway.
pub const MAX_IMAGE_EDGE: u32 = 1568;
/// Encoded bytes, per image.
const MAX_IMAGE_BYTES: usize = 300_000;
/// Encoded bytes for all images in one result. Codex turns results over ~1 MB into a text dump
/// (the model then sees base64, not images) and the Claude SDK reader caps lines at 1 MB;
/// base64 adds 4/3, so this keeps a full result near 750 KB.
const RESULT_IMAGE_BYTES: usize = 560_000;

/// Runs one tool by name. May block (job_status waits up to 600 s).
pub type Dispatch = Arc<dyn Fn(&str, Value) -> anyhow::Result<ToolResult> + Send + Sync>;

pub fn finish_path(root: &Path) -> std::path::PathBuf {
    root.join("agent").join("finish.json")
}

#[derive(Clone)]
pub struct SplatServer {
    run: Arc<dyn RunCtx>,
    dispatch: Dispatch,
    tools: Arc<Vec<Tool>>,
}

impl SplatServer {
    pub fn new(run: Arc<dyn RunCtx>, dispatch: Dispatch) -> Self {
        Self { run, dispatch, tools: Arc::new(tools()) }
    }

    async fn call(&self, name: &str, args: Value) -> CallToolResult {
        let (this, name_owned) = (self.clone(), name.to_string());
        match tokio::task::spawn_blocking(move || (this.dispatch)(&name_owned, args)).await {
            Ok(Ok(result)) => self.render(result),
            Ok(Err(e)) => error_result(name, "tool_error", &format!("{e:#}")),
            Err(e) => error_result(name, "internal_error", &e.to_string()),
        }
    }

    fn render(&self, result: ToolResult) -> CallToolResult {
        let is_error = result.status == Status::Failed;
        let mut out = if is_error { CallToolResult::error(vec![]) } else { CallToolResult::success(vec![]) };
        out.content = result_content(self.run.root(), &result);
        out.structured_content = serde_json::to_value(&result).ok();
        out
    }
}

impl ServerHandler for SplatServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("vauss", env!("CARGO_PKG_VERSION")))
    }

    async fn list_tools(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListToolsResult, ErrorData> {
        Ok(ListToolsResult::with_all_items((*self.tools).clone()))
    }

    async fn call_tool(
        &self,
        request: CallToolRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<CallToolResponse, ErrorData> {
        if !self.tools.iter().any(|t| t.name == request.name) {
            return Ok(error_result(&request.name, "unknown_tool", &format!("no tool named {:?}", request.name)).into());
        }
        let args = Value::Object(request.arguments.unwrap_or_default());
        Ok(self.call(&request.name, args).await.into())
    }
}

pub fn tools() -> Vec<Tool> {
    tool_schemas()
        .into_iter()
        .map(|(name, desc, schema)| {
            let mut root = serde_json::to_value(schema).expect("schema serializes");
            let defs = root.as_object_mut().and_then(|o| o.remove("definitions")).unwrap_or_default();
            let Value::Object(mut obj) = portable_schema(root, &defs) else { unreachable!("root schema is an object") };
            obj.remove("$schema");
            Tool::new(name, desc, Arc::new(obj))
        })
        .collect()
}

/// Rewrite schemars draft-07 output into the subset every driver accepts. Codex silently drops a
/// tool whose schema it rejects (tuple-form `items` hid extract_frames), so: inline `$ref`s,
/// unwrap single-element `allOf`, turn tuple `items` into one item schema (bounds stay in
/// min/maxItems), and drop schemars' non-standard numeric `format`s.
fn portable_schema(v: Value, defs: &Value) -> Value {
    match v {
        Value::Object(mut o) => {
            if let Some(r) = o.remove("$ref").as_ref().and_then(Value::as_str) {
                let target = defs.get(r.trim_start_matches("#/definitions/")).cloned().unwrap_or_default();
                let Value::Object(t) = portable_schema(target, defs) else { return Value::Object(o) };
                o.extend(t);
            }
            if let Some(Value::Array(mut all)) = o.remove("allOf") {
                if all.len() == 1 {
                    if let Value::Object(inner) = portable_schema(all.remove(0), defs) {
                        o.extend(inner);
                    }
                } else {
                    o.insert("allOf".into(), Value::Array(all));
                }
            }
            if let Some(Value::Array(mut items)) = o.remove("items") {
                if !items.is_empty() {
                    o.insert("items".into(), items.remove(0));
                }
            }
            if o.get("format").and_then(Value::as_str).is_some_and(|f| matches!(f, "uint" | "uint8" | "uint16" | "uint32" | "uint64" | "int8" | "int16" | "int32" | "int64" | "float" | "double")) {
                o.remove("format");
            }
            Value::Object(o.into_iter().map(|(k, v)| (k, portable_schema(v, defs))).collect())
        }
        Value::Array(a) => Value::Array(a.into_iter().map(|v| portable_schema(v, defs)).collect()),
        other => other,
    }
}

fn error_result(tool: &str, code: &str, message: &str) -> CallToolResult {
    let result = ToolResult {
        schema_version: splat_contracts::SCHEMA_VERSION,
        tool: tool.into(),
        status: Status::Failed,
        job_id: None,
        artifacts: vec![],
        metrics: Default::default(),
        images: vec![],
        warnings: vec![],
        error: Some(ToolError { code: code.into(), message: message.into(), retryable: false, evidence: vec![] }),
        wall_s: None,
    };
    CallToolResult::error(vec![ContentBlock::text(serde_json::to_string(&result).unwrap_or_default())])
}

/// ToolResult JSON, then each image labelled with its run-relative path so the model can cite it.
/// Images share RESULT_IMAGE_BYTES so the whole result stays under the drivers' message limits.
pub fn result_content(root: &Path, result: &ToolResult) -> Vec<ContentBlock> {
    let mut content = vec![ContentBlock::text(serde_json::to_string_pretty(result).unwrap_or_default())];
    let per_image = MAX_IMAGE_BYTES.min(RESULT_IMAGE_BYTES / result.images.len().max(1));
    for rel in &result.images {
        match encode_image(&root.join(rel), per_image) {
            Ok((data, mime)) => {
                content.push(ContentBlock::text(format!("image: {rel}")));
                content.push(ContentBlock::image(data, mime));
            }
            Err(e) => content.push(ContentBlock::text(format!("[image {rel} unreadable: {e:#}]"))),
        }
    }
    content
}

/// Downscale to <= MAX_IMAGE_EDGE, then further until the encoding fits `max_bytes`.
/// Returns (base64, mime). JPEG unless the image has alpha.
pub fn encode_image(path: &Path, max_bytes: usize) -> anyhow::Result<(String, &'static str)> {
    let img = image::open(path).with_context(|| format!("open {}", path.display()))?;
    let alpha = img.color().has_alpha();
    let mut edge = MAX_IMAGE_EDGE.min(img.width().max(img.height()));
    loop {
        let frame = if img.width().max(img.height()) > edge { img.resize(edge, edge, FilterType::Triangle) } else { img.clone() };
        let mut buf = std::io::Cursor::new(Vec::new());
        let mime = if alpha {
            DynamicImage::ImageRgba8(frame.to_rgba8()).write_to(&mut buf, ImageFormat::Png)?;
            "image/png"
        } else {
            let rgb = frame.to_rgb8();
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 85).encode_image(&rgb)?;
            "image/jpeg"
        };
        let bytes = buf.into_inner();
        if bytes.len() <= max_bytes || edge <= 256 {
            return Ok((base64::engine::general_purpose::STANDARD.encode(bytes), mime));
        }
        // Bytes scale roughly with pixel count: jump near the target instead of creeping down.
        let scale = ((max_bytes as f64 / bytes.len() as f64).sqrt() * 0.9).min(0.8);
        edge = ((edge as f64 * scale) as u32).max(256);
    }
}

/// Serve MCP over stdio until the client disconnects.
pub async fn serve_stdio(run: Arc<dyn RunCtx>, dispatch: Dispatch) -> anyhow::Result<()> {
    eprintln!("Vauss MCP server on {}", run.root().display());
    let service = SplatServer::new(run, dispatch).serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}

/// Blocking entry point for `vauss serve --run-dir R`.
pub fn serve(run: Arc<dyn RunCtx>, dispatch: Dispatch) -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread().enable_all().build()?.block_on(serve_stdio(run, dispatch))
}
