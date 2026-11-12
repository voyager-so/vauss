//! Video models on fal.ai, one adapter each. Payloads follow docs/research/06-video-gen.md
//! (checked against fal's live OpenAPI, Oct 8 2026). Prices are list prices for the run ledger.

use serde_json::{Value, json};

/// Local inputs a model can ask to have uploaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Input {
    /// RGB pass as an mp4 (H.264, or MPEG-4 part 2 when only VP9 is available for people).
    RgbVideo,
    /// Depth pass conformed to 16 fps / 81 frames for Wan VACE, as an mp4.
    Depth16Video,
    /// Depth pass at 16 fps covering the whole shot (81..241 frames, i.e. 5-15 s).
    DepthFull16Video,
    FirstFrame,
    LastFrame,
    /// Real capture photo near the opening pose (appearance anchor).
    Reference,
}

pub struct Ctx<'a> {
    pub prompt: &'a str,
    pub duration_s: f64,
    /// Uploaded URL per requested input, in `VideoModel::inputs` order.
    pub urls: &'a [String],
    /// "16:9" or "9:16", from the shot's frame size.
    pub aspect_ratio: &'a str,
}

/// Frames for a full-length 16 fps Wan VACE conditioning video.
pub fn wan_frames(duration_s: f64) -> u32 {
    ((duration_s * 16.0).round() as u32 + 1).clamp(81, 241)
}

pub struct VideoModel {
    pub key: &'static str,
    pub endpoint: &'static str,
    pub inputs: &'static [Input],
    pub payload: fn(&Ctx) -> Value,
    pub usd_estimate: fn(f64) -> Option<f64>,
    pub note: &'static str,
    /// Rejects anything but H.264 input video (H3 422s on MPEG-4 part 2 and VP9).
    pub h264_only: bool,
}

fn secs(d: f64) -> u32 {
    (d.round() as u32).clamp(3, 15)
}

pub const MODELS: &[VideoModel] = &[
    VideoModel {
        key: "luma_ray32_v2v",
        endpoint: "luma/agent/ray/v3.2/video-to-video",
        inputs: &[Input::RgbVideo],
        payload: |c| {
            json!({
                "video_url": c.urls[0], "prompt": c.prompt, "resolution": "720p",
                "duration": if c.duration_s > 7.5 { "10s" } else { "5s" }, "edit_strength": "adhere_2",
            })
        },
        usd_estimate: |d| Some(0.216 * if d > 7.5 { 10.0 } else { 5.0 }),
        note: "RGB restyle (adhere_2)",
        h264_only: false,
    },
    VideoModel {
        key: "wan_vace_depth",
        endpoint: "fal-ai/wan-22-vace-fun-a14b/depth",
        inputs: &[Input::Depth16Video, Input::Reference],
        payload: |c| {
            json!({
                "video_url": c.urls[0], "preprocess": false, // already disparity, near = bright
                "ref_image_urls": [c.urls[1]], "prompt": c.prompt,
                "negative_prompt": "cuts, flicker, warped geometry, floating objects, text, watermark",
                "resolution": "720p", "aspect_ratio": "16:9", "num_frames": 81, "frames_per_second": 16,
                "match_input_num_frames": true, "match_input_frames_per_second": true,
                "enable_auto_downsample": false, "temporal_downsample_factor": 0, "num_interpolated_frames": 0,
                "enable_prompt_expansion": false, "seed": 42,
            })
        },
        usd_estimate: |_| Some(0.10 * 81.0 / 16.0),
        note: "depth pass + capture photo reference",
        h264_only: false,
    },
    VideoModel {
        key: "wan_vace_depth_full",
        endpoint: "fal-ai/wan-22-vace-fun-a14b/depth",
        inputs: &[Input::DepthFull16Video, Input::Reference],
        payload: |c| {
            json!({
                "video_url": c.urls[0], "preprocess": false,
                "ref_image_urls": [c.urls[1]], "prompt": c.prompt,
                "negative_prompt": "cuts, flicker, warped geometry, floating objects, text, watermark, camera shake",
                "resolution": "720p", "aspect_ratio": c.aspect_ratio, "num_frames": wan_frames(c.duration_s), "frames_per_second": 16,
                "match_input_num_frames": true, "match_input_frames_per_second": true,
                "enable_auto_downsample": false, "temporal_downsample_factor": 0, "num_interpolated_frames": 0,
                "enable_prompt_expansion": false, "seed": 42,
            })
        },
        usd_estimate: |d| Some(0.10 * wan_frames(d) as f64 / 16.0),
        note: "depth pass (whole shot, up to 15 s) + capture photo reference",
        h264_only: false,
    },
    VideoModel {
        key: "kling_o3_edit",
        endpoint: "fal-ai/kling-video/o3/pro/video-to-video/edit",
        inputs: &[Input::RgbVideo],
        payload: |c| json!({"video_url": c.urls[0], "prompt": c.prompt, "keep_audio": false}),
        usd_estimate: |d| Some(0.168 * d.clamp(3.0, 15.0)),
        note: "RGB video edit (relight/restyle), 1080p, keeps structure and camera; 3-15 s",
        h264_only: false,
    },
    VideoModel {
        key: "wan27_edit",
        endpoint: "fal-ai/wan/v2.7/edit-video",
        inputs: &[Input::RgbVideo],
        payload: |c| json!({"video_url": c.urls[0], "prompt": c.prompt, "resolution": "720p"}),
        usd_estimate: |d| Some(0.10 * d.clamp(2.0, 10.0)),
        note: "RGB video edit, 720p, cheap; 2-10 s",
        h264_only: false,
    },
    VideoModel {
        key: "ltx23_render_to_real",
        endpoint: "fal-ai/ltx-2.3-quality/render-to-real",
        inputs: &[Input::RgbVideo],
        payload: |c| {
            json!({
                "video_url": c.urls[0], "prompt": format!("3DREAL. {}", c.prompt), "intensity": "light",
                "resolution": "720p", "num_frames": ((c.duration_s * 24.0).round() as u32 + 1).min(169), // fal caps 1152x704 at ~169 frames
                "frames_per_second": 24, "generate_audio": false, "seed": 42,
            })
        },
        usd_estimate: |d| Some(0.0024075 * 1280.0 * 720.0 / 1e6 * ((d * 24.0).round() + 1.0)),
        note: "render-to-real (light)",
        h264_only: false,
    },
    VideoModel {
        key: "kling_v3_keyframes",
        endpoint: "fal-ai/kling-video/v3/pro/image-to-video",
        inputs: &[Input::FirstFrame, Input::LastFrame],
        payload: |c| {
            json!({
                "start_image_url": c.urls[0], "end_image_url": c.urls[1], "prompt": c.prompt,
                "duration": secs(c.duration_s).to_string(), "generate_audio": false,
            })
        },
        usd_estimate: |d| Some(0.112 * secs(d) as f64),
        note: "first/last rendered frames only; in-between path is the model's",
        h264_only: false,
    },
    VideoModel {
        key: "h3max_3d",
        endpoint: "minimax/h3-max/3d-to-video",
        inputs: &[Input::RgbVideo, Input::Reference],
        payload: |c| json!({"video_url": c.urls[0], "prompt": c.prompt, "resolution": "768P", "reference_image_urls": [c.urls[1]]}),
        usd_estimate: |d| Some(0.08 * d.max(5.0)), // plus reference-token charges
        note: "previs-to-video; camera adherence not guaranteed",
        h264_only: true,
    },
];

pub fn model(key: &str) -> Option<&'static VideoModel> {
    MODELS.iter().find(|m| m.key == key)
}

pub fn keys() -> Vec<&'static str> {
    MODELS.iter().map(|m| m.key).collect()
}
