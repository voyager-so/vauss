//! Side-by-side review page: rough render | depth | each generated clip, one row per shot.

use std::fmt::Write as _;
use std::path::{Component, Path, PathBuf};

use anyhow::Result;

use crate::generate::GenRecord;
use crate::layout::ShotDir;

const CSS: &str = ":root{--bg:#fff;--fg:#1a1a1a;--muted:#666;--card:#f4f4f5;--line:#ddd;--hi:#b45309;--ok:#15803d}
@media (prefers-color-scheme:dark){:root{--bg:#111214;--fg:#e8e8ea;--muted:#9a9aa2;--card:#1b1c1f;--line:#2c2d31}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:0}
.shot{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:16px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px;margin-top:10px}
figure{margin:0}video{width:100%;border-radius:4px;background:#000}figcaption{color:var(--muted);font-size:12px}
.meta{color:var(--muted);font-size:12px}.risk-high{color:var(--hi)}.risk-low{color:var(--ok)}
code{font-size:12px;word-break:break-word}";

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// Relative path from `base` dir to `target` (both absolute).
fn rel(target: &Path, base: &Path) -> String {
    let (t, b): (Vec<Component>, Vec<Component>) = (target.components().collect(), base.components().collect());
    let common = t.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let mut p = PathBuf::new();
    for _ in common..b.len() {
        p.push("..");
    }
    p.extend(&t[common..]);
    p.to_string_lossy().replace('\\', "/")
}

fn mime(p: &Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() {
        Some("webm") => "video/webm",
        Some("mov") => "video/quicktime",
        _ => "video/mp4",
    }
}

fn video(src: &Path, root: &Path, cap: &str) -> String {
    format!(
        r#"<figure><video controls loop muted playsinline preload="metadata"><source src="{}" type="{}"></video><figcaption>{cap}</figcaption></figure>"#,
        esc(&rel(src, root)),
        mime(src)
    )
}

pub fn write_review(shot_dirs: &[PathBuf], out: &Path, title: &str) -> Result<PathBuf> {
    let root = std::path::absolute(out.parent().unwrap_or(Path::new(".")))?;
    let mut rows = String::new();
    for d in shot_dirs {
        let d = std::path::absolute(d)?;
        let shot = ShotDir(d.clone());
        let rec: serde_json::Value = serde_json::from_slice(&std::fs::read(d.join("shot.json"))?)?;
        let r = shot.render_meta()?;
        let path = shot.path()?;
        let risk = &rec["risk"];
        let mut cells = vec![
            video(&shot.rgb_video(), &root, &format!("rough vauss render ({}) · alpha {:.3}", r.engine, r.alpha_mean)),
            video(&shot.depth_video(), &root, "depth (disparity, near = bright)"),
        ];
        let gens: Vec<GenRecord> = serde_json::from_value(rec["generated"].clone()).unwrap_or_default();
        for g in gens {
            match (&g.video, &g.error) {
                (Some(v), None) => {
                    let cost = g.usd_estimate.map_or("price n/a".into(), |c| format!("~${c:.2}"));
                    let cap = format!("<b>{}</b> · {}s · {cost} · {}", esc(&g.model), g.wall_s.unwrap_or(0.0), esc(&g.note));
                    cells.push(video(v, &root, &cap));
                }
                (_, e) => cells.push(format!(
                    "<figure><figcaption><b>{}</b> failed: {}</figcaption></figure>",
                    esc(&g.model),
                    esc(&crate::tail(e.as_deref().unwrap_or("no video returned"), 300))
                )),
            }
        }
        let mv = path.movement.as_ref().map(|m| serde_json::to_value(m.movement).unwrap_or_default());
        let _ = write!(
            rows,
            r#"<section class="shot"><h2>{}</h2><div class="meta">{} · {}s @ {} fps · {}×{} · risk <span class="risk-{}">{}</span> (max {} scene radii from a capture camera) · render {:.1}s</div><div class="meta">prompt: <code>{}</code></div><div class="grid">{}</div></section>"#,
            esc(rec["name"].as_str().unwrap_or(&path.name)),
            esc(mv.as_ref().and_then(|v| v.as_str()).unwrap_or("custom")),
            path.duration_s,
            path.fps,
            path.width,
            path.height,
            esc(risk["risk"].as_str().unwrap_or("?")),
            esc(risk["risk"].as_str().unwrap_or("?")),
            risk["novelty_max"],
            r.render_s,
            esc(rec["prompt"].as_str().unwrap_or("")),
            cells.join("")
        );
    }
    std::fs::write(
        out,
        format!(
            r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{t}</title><style>{CSS}</style></head><body><h1>{t}</h1><div class="meta">camera path → vauss render → video model. Videos are relative to this file.</div>{rows}</body></html>"#,
            t = esc(title)
        ),
    )?;
    Ok(out.to_path_buf())
}
