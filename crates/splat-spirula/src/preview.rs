//! Mesh previews for the agent, via the shared `splat_mesh::preview` (same framing and
//! rasterizer as the Brush engine): four orbit views of the subject, colour (texture sampled at
//! the vertices) beside shaded geometry.
use crate::media::jpeg;
use anyhow::Result;
use std::path::{Path, PathBuf};

/// Spirula meshes are in its training frame, which is Z-up with the ground near z = 0.
pub(crate) fn render(path: &Path, out: &Path) -> Result<Vec<PathBuf>> {
    let images = splat_mesh::preview::previews_for_file(path, &[], Some(splat_mesh::glam::Vec3::Z), 4, 512)?;
    std::fs::create_dir_all(out)?;
    let mut paths = vec![];
    for (i, image) in images.into_iter().enumerate() {
        let target = out.join(format!("geometry-{i}.jpg"));
        jpeg(&target, &image::DynamicImage::ImageRgb8(image), 85)?;
        paths.push(target);
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn geometry_is_rendered_for_ply_and_obj() {
        let tmp = tempfile::tempdir().unwrap();
        for (ext, data) in [
            ("obj", "v -1 -1 0\nv 1 -1 0\nv 0 1 1\nf 1 2 3\n"),
            (
                "ply",
                "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n-1 -1 0\n1 -1 0\n0 1 1\n3 0 1 2\n",
            ),
        ] {
            let path = tmp.path().join(format!("mesh.{ext}"));
            std::fs::write(&path, data).unwrap();
            let previews = render(&path, &tmp.path().join(ext)).unwrap();
            assert_eq!(previews.len(), 4);
            let im = image::open(&previews[0]).unwrap().to_rgb8();
            assert!(im.pixels().filter(|p| p[0] < 200).count() > 1000);
        }
    }
}
