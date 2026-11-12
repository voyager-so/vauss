use splat_fetch::{Fetcher, LicenseConsent, asset_for, default_cache_dir, host_target};
use std::{fs, path::Path};

/// `fetch ASSET [--stage TOOLS_DIR]` verifies a pinned asset into the cache. With `--stage`,
/// it also copies the verified install to `TOOLS_DIR/<asset-id>/` (the release-archive layout).
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let id = args.next().ok_or_else(|| {
        anyhow::anyhow!("usage: fetch ASSET [--allow-gpl] [--target TARGET] [--stage TOOLS_DIR]")
    })?;
    let mut consent = LicenseConsent::Default;
    let mut target = host_target().to_owned();
    let mut stage = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--allow-gpl" => consent = LicenseConsent::AllowGpl,
            "--target" => {
                target = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing target"))?
            }
            "--stage" => {
                stage = Some(
                    args.next()
                        .ok_or_else(|| anyhow::anyhow!("missing stage directory"))?,
                )
            }
            _ => anyhow::bail!("unknown argument: {arg}"),
        }
    }
    let fetcher = Fetcher::new(default_cache_dir()?);
    let fetcher = match std::env::var("SPLAT_GITHUB_TOKEN") {
        Ok(token) => fetcher.with_github_token(token),
        Err(_) => fetcher,
    };
    let asset = asset_for(&id, &target)?;
    let path = fetcher.ensure_asset(&asset, consent)?;
    let Some(stage) = stage else {
        println!("{}", path.display());
        return Ok(());
    };
    // The entrypoint is relative to the asset's install root; walk back up to that root.
    let mut root = path.as_path();
    for _ in Path::new(&asset.entrypoint).components() {
        root = root.parent().expect("entrypoint is inside its install root");
    }
    let dest = Path::new(&stage).join(&asset.id);
    anyhow::ensure!(!dest.exists(), "{} already exists", dest.display());
    copy_tree(root, &dest)?;
    println!("{}", dest.join(&asset.entrypoint).display());
    Ok(())
}

fn copy_tree(from: &Path, to: &Path) -> anyhow::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let (src, dst) = (entry.path(), to.join(entry.file_name()));
        let ty = entry.file_type()?;
        if ty.is_symlink() {
            #[cfg(unix)]
            std::os::unix::fs::symlink(fs::read_link(&src)?, &dst)?;
            #[cfg(not(unix))]
            fs::copy(&src, &dst)?;
        } else if ty.is_dir() {
            copy_tree(&src, &dst)?;
        } else {
            // fs::copy keeps permission bits, so staged executables stay executable.
            fs::copy(&src, &dst)?;
        }
    }
    Ok(())
}
