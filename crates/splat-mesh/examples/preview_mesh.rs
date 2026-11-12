//! Render the standard orbit previews of a mesh file: `preview_mesh <mesh.glb|obj|ply> <out dir> [--z-up]`.

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let up = args.iter().any(|a| a == "--z-up").then_some(glam::Vec3::Z);
    let out = std::path::PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out)?;
    let t = std::time::Instant::now();
    let imgs = splat_mesh::preview::previews_for_file(args[0].as_ref(), &[], up, 4, 512)?;
    for (i, im) in imgs.iter().enumerate() {
        im.save(out.join(format!("geometry-{i}.png")))?;
    }
    eprintln!("{} previews in {:.2}s", imgs.len(), t.elapsed().as_secs_f64());
    Ok(())
}
