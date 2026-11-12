//! Offline replay: detail_pack PREVIEW.glb DETAIL.glb OUT_DIR. No credentials or paid requests.
use anyhow::{Context, Result};
use splat_assets::{CatalogItem, NormalizeOpts, SizeTarget, normalize_glb};
fn main() -> Result<()> {
    let a: Vec<_> = std::env::args().collect();
    if a.get(1).is_some_and(|s| s == "--serve") {
        let scene = a.get(2).context("--serve needs a replay scene directory")?;
        return splat_assets::spawn(splat_assets::ServeOpts { inputs: vec![scene.into()], port: Some(8899), ..Default::default() })?.wait();
    }
    anyhow::ensure!(a.len() == 4, "usage: detail_pack PREVIEW.glb DETAIL.glb OUT_DIR");
    let (_, b) = normalize_glb(&std::fs::read(&a[1])?, &NormalizeOpts { target: SizeTarget::Height(0.9), ..NormalizeOpts::default() })?;
    let item = CatalogItem { id:"replay".into(),name:"Bakeoff replay".into(), category:"chair".into(),glb:"items/preview.glb".into(),thumb:"".into(),dims_m:b.dims_m,license:"generated bakeoff".into(),source:"public generated bakeoff".into(),generated:true,product:None };
    let work = std::path::Path::new(&a[3]).join("work/replay");
    std::fs::create_dir_all(&work)?;
    std::fs::copy(&a[1], work.join("raw.glb"))?;
    let (preview, _) = normalize_glb(&std::fs::read(&a[1])?, &NormalizeOpts { target: SizeTarget::Height(0.9), ..NormalizeOpts::default() })?;
    std::fs::write(std::path::Path::new(&a[3]).join("preview.glb"), preview)?;
    let report = splat_assets::detail::compress_item(&std::fs::read(&a[2])?, &item, std::path::Path::new(&a[3]), &|s| eprintln!("{s}"))?;
    let out = std::path::Path::new(&a[3]).join("report.json");
    std::fs::write(out,serde_json::to_vec_pretty(&report)?).context("save report")?;
    std::fs::write(std::path::Path::new(&a[3]).join("items/replay-detail.json"), serde_json::to_vec_pretty(&report)?)?;
    println!("{}",serde_json::to_string_pretty(&report)?);
    Ok(())
}
