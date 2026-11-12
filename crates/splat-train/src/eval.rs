#[cfg(not(target_family = "wasm"))]
use std::path::Path;

use anyhow::Result;
use brush_dataset::scene::view_to_packed_data;
use brush_loss::{ImageLossConfig, image_loss_eval, psnr_from_mse};
use brush_render::camera::Camera;
use brush_render::gaussian_splats::Splats;
use brush_render::{AlphaMode, RenderAux, TextureMode, render_splats};
use burn::tensor::{Device, Int, Tensor, s};
use glam::Vec3;
use image::DynamicImage;

pub struct EvalSample {
    pub gt_img: DynamicImage,
    /// The scored render: float, clamped to [0, 1], then an 8-bit round-trip.
    pub rendered: Tensor<3>,
    /// The same render before the 8-bit round-trip (float, clamped to
    /// [0, 1]): what the common paired scorer receives.
    pub rendered_float: Tensor<3>,
    pub psnr: Tensor<1>,
    pub ssim: Tensor<1>,
    pub render_aux: RenderAux,
}

/// Test-time appearance for held-out views, applied to the float render
/// before it is scored or exported, e.g. a PPISP controller predicting
/// exposure/colour from the render plus per-camera vignetting/CRF. It gets the
/// camera and the render only, never the ground truth, so it cannot leak the
/// target into the score.
pub trait EvalAppearance: Send + Sync {
    /// `render` is `[H, W, 3]` float on the reference black background.
    fn apply(&self, view: &EvalView<'_>, render: Tensor<3>) -> Tensor<3>;
}

/// What an [`EvalAppearance`] may know about a held-out view (never its
/// ground truth). Extended by adding fields, so implementors keep compiling.
#[non_exhaustive]
pub struct EvalView<'a> {
    pub camera: &'a Camera,
    /// Dense index of the view's physical camera (intrinsics group), for
    /// per-camera ISP state such as vignetting and CRF. 0 for single-camera
    /// captures.
    pub camera_index: usize,
}

pub async fn eval_stats(
    splats: Splats,
    gt_cam: &Camera,
    gt_img: DynamicImage,
    alpha_mode: AlphaMode,
    device: &Device,
) -> Result<EvalSample> {
    eval_stats_with(splats, gt_cam, gt_img, alpha_mode, device, None).await
}

/// [`eval_stats`] with an optional [`EvalAppearance`] (and the view's camera
/// index) applied to the render.
pub async fn eval_stats_with(
    splats: Splats,
    gt_cam: &Camera,
    gt_img: DynamicImage,
    alpha_mode: AlphaMode,
    device: &Device,
    appearance: Option<(&dyn EvalAppearance, usize)>,
) -> Result<EvalSample> {
    let res = glam::uvec2(gt_img.width(), gt_img.height());

    let (gt_packed_data, _has_alpha) = view_to_packed_data(gt_img.clone(), alpha_mode);
    let gt_packed: Tensor<2, Int> = Tensor::from_data(gt_packed_data, device);

    // Render on reference black background.
    let (img, render_aux) =
        render_splats(splats, gt_cam, res, Vec3::ZERO, None, TextureMode::Float).await;
    let render_rgb = img.slice(s![.., .., 0..3]);
    let render_rgb = match appearance {
        Some((appearance, camera_index)) => appearance.apply(
            &EvalView {
                camera: gt_cam,
                camera_index,
            },
            render_rgb,
        ),
        None => render_rgb,
    };
    let rendered_float = render_rgb.clamp(0.0, 1.0);

    // Simulate an 8-bit roundtrip for fair comparison.
    let render_rgb = (rendered_float.clone() * 255.0).round() / 255.0;

    let cfg = |l1, ssim| ImageLossConfig {
        l1_weight: l1,
        ssim_weight: ssim,
        composite_bg: None,
        mask: false,
        alpha_weight: 0.0,
    };
    // MSE = mean(L1^2) since |a - b|^2 == (a - b)^2.
    let mse = image_loss_eval(render_rgb.clone(), gt_packed.clone(), cfg(1.0, 0.0))
        .powi_scalar(2)
        .mean();
    let psnr = psnr_from_mse(mse);
    let ssim = image_loss_eval(render_rgb.clone(), gt_packed, cfg(0.0, 1.0)).mean();

    Ok(EvalSample {
        gt_img,
        psnr,
        ssim,
        rendered: render_rgb,
        rendered_float,
        render_aux,
    })
}

impl EvalSample {
    #[cfg(not(target_family = "wasm"))]
    pub async fn save_to_disk(&self, path: &Path) -> anyhow::Result<()> {
        self.save_to_disk_background(path.to_path_buf()).await?.await?
    }

    /// Reads the render back, then converts, PNG-encodes and writes it on a
    /// blocking thread; the caller awaits the handle when it needs the file.
    #[cfg(not(target_family = "wasm"))]
    pub async fn save_to_disk_background(
        &self,
        path: std::path::PathBuf,
    ) -> anyhow::Result<tokio::task::JoinHandle<anyhow::Result<()>>> {
        use image::Rgb32FImage;
        log::info!("Saving eval image to disk.");
        let img = self.rendered.clone();
        let [h, w, _] = [img.dims()[0], img.dims()[1], img.dims()[2]];
        let data = img.into_data_async().await?.try_into_vec::<f32>()?;
        let parent = path.parent().expect("Eval must have a filename").to_path_buf();
        tokio::fs::create_dir_all(&parent).await?;
        Ok(tokio::task::spawn_blocking(move || {
            let img: image::DynamicImage = Rgb32FImage::from_raw(w as u32, h as u32, data)
                .expect("Failed to create image from tensor")
                .into();
            let img: image::DynamicImage = img.into_rgb8().into();
            log::info!("Saving eval view to {path:?}");
            img.save(&path)?;
            Ok(())
        }))
    }
}
