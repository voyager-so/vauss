//! Video encoding via an ffmpeg subprocess. The binary is resolved by the caller (CLI config →
//! pinned LGPL download) and handed in. The LGPL build has no libx264, so the codec is probed:
//! H.264 in mp4 when an LGPL-compatible encoder works here (OS encoders, openh264, Vulkan Video on
//! the GPU), else VP9 in WebM. Both play in every current browser; MPEG-4 part 2 does not, so people never get it.
//! Model uploads are another matter: fal's LTX and H3 servers reject VP9 (tested Oct 8 2026,
//! 422 "could not be read"), so without H.264 the upload copy is MPEG-4 part 2 in mp4 (ffmpeg's
//! own LGPL encoder; Wan, Luma and LTX accept it, H3 needs H.264).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};

/// One ffmpeg encoder setup: args before the input (hardware device), a filter tail that ends in
/// the encoder's pixel format, and the encoder args.
#[derive(Debug, PartialEq, Eq)]
pub struct Enc {
    pre: &'static [&'static str],
    vf: &'static str,
    args: &'static [&'static str],
}

/// The codec every video we write uses, picked once per ffmpeg binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    /// H.264 in mp4 with this encoder.
    H264(&'static Enc),
    /// VP9 (libvpx) in WebM for people; MPEG-4 part 2 mp4 for model uploads.
    Vp9,
}

const YUV: &str = "format=yuv420p";
/// H.264 encoders we may use, best first. libx264 only appears in a user-supplied ffmpeg (GPL is
/// theirs, not ours); the OS encoders, openh264 and Vulkan Video keep the pinned build LGPL-clean.
/// h264_vulkan runs on the GPU's fixed-function encoder (NVENC/VCN/QSV), not its compute cores.
const H264: &[Enc] = &[
    Enc { pre: &[], vf: YUV, args: &["-c:v", "libx264", "-crf", "16", "-preset", "medium"] },
    Enc { pre: &[], vf: YUV, args: &["-c:v", "h264_videotoolbox", "-allow_sw", "1", "-b:v", "12M"] },
    Enc { pre: &[], vf: YUV, args: &["-c:v", "h264_mf", "-b:v", "12M"] },
    Enc { pre: &[], vf: YUV, args: &["-c:v", "libopenh264", "-b:v", "12M"] },
    Enc {
        pre: &["-init_hw_device", "vulkan=vk", "-filter_hw_device", "vk"],
        vf: "format=nv12,hwupload",
        args: &["-c:v", "h264_vulkan", "-b:v", "12M"],
    },
];
const MPEG4: Enc = Enc { pre: &[], vf: YUV, args: &["-c:v", "mpeg4", "-q:v", "2"] };
const VP9: Enc = Enc {
    pre: &[],
    vf: YUV,
    args: &["-c:v", "libvpx-vp9", "-crf", "24", "-b:v", "0", "-deadline", "good", "-cpu-used", "4", "-row-mt", "1", "-threads", "4"],
};

impl Codec {
    /// Extension of the human-facing file.
    pub fn ext(self) -> &'static str {
        match self {
            Codec::H264(_) => "mp4",
            Codec::Vp9 => "webm",
        }
    }
    fn enc(self) -> &'static Enc {
        match self {
            Codec::H264(e) => e,
            Codec::Vp9 => &VP9,
        }
    }
    /// Human-facing encoder as (args before the inputs, filter tail ending in its pixel format,
    /// encoder args), for callers that build their own ffmpeg command (e.g. `splat ad`'s cut).
    pub fn parts(self) -> (&'static [&'static str], &'static str, &'static [&'static str]) {
        let e = self.enc();
        (e.pre, e.vf, e.args)
    }
    /// Encoder for a model-upload mp4.
    fn upload_enc(self) -> &'static Enc {
        match self {
            Codec::H264(e) => e,
            Codec::Vp9 => &MPEG4,
        }
    }
}

pub struct Ffmpeg {
    path: PathBuf,
    codec: OnceLock<Codec>,
}

impl Ffmpeg {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into(), codec: OnceLock::new() }
    }

    /// Dev/worker default: SPLAT_FFMPEG (set on job env by `Shots`), else `ffmpeg` on PATH.
    pub fn from_env() -> Self {
        Self::new(std::env::var_os("SPLAT_FFMPEG").map(PathBuf::from).unwrap_or_else(|| "ffmpeg".into()))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// First H.264 encoder that actually encodes a test clip here (OS and GPU encoders can be listed
    /// but unusable, e.g. no Media Foundation H.264 MFT or no Vulkan video queue), else VP9.
    pub fn codec(&self) -> Codec {
        *self.codec.get_or_init(|| {
            let list = Command::new(&self.path)
                .args(["-hide_banner", "-encoders"])
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
                .unwrap_or_default();
            H264.iter()
                .find(|e| list.split_whitespace().any(|w| w == e.args[1]) && self.trial(e))
                .map_or(Codec::Vp9, Codec::H264)
        })
    }

    fn trial(&self, e: &Enc) -> bool {
        let vf = format!("scale=256:144,{}", e.vf);
        Command::new(&self.path)
            .args(["-hide_banner", "-loglevel", "error"])
            .args(e.pre)
            .args(["-f", "lavfi", "-i", "testsrc=s=256x144:r=8:d=0.5", "-vf", &vf])
            .args(e.args)
            .args(["-f", "null", "-"])
            .output()
            .is_ok_and(|o| o.status.success())
    }

    fn run(&self, args: &[&str]) -> Result<()> {
        let out = Command::new(&self.path)
            .args(["-y", "-hide_banner", "-loglevel", "error"])
            .args(args)
            .output()
            .with_context(|| format!("running {}", self.path.display()))?;
        if !out.status.success() {
            bail!("ffmpeg failed: {}", String::from_utf8_lossy(&out.stderr));
        }
        Ok(())
    }

    /// Transcode `input_args` (ending in `-i SRC`) through `vf` + `e` to `out`.
    fn transcode(&self, e: &Enc, input_args: &[&str], vf: &str, extra: &[&str], out: &Path) -> Result<()> {
        let vf = if vf.is_empty() { e.vf.to_string() } else { format!("{vf},{}", e.vf) };
        let mut args: Vec<&str> = e.pre.to_vec();
        args.extend_from_slice(input_args);
        args.extend_from_slice(&["-vf", &vf]);
        args.extend_from_slice(extra);
        args.extend_from_slice(e.args);
        if out.extension().is_some_and(|x| x == "mp4") {
            args.extend_from_slice(&["-movflags", "+faststart"]);
        }
        args.push(out.to_str().context("non-UTF-8 path")?);
        self.run(&args)
    }

    /// `frames_dir/%05d.png` -> `stem.<mp4|webm>` (4:2:0, even dims). Returns the written path and
    /// removes a stale copy in the other container, so a shot never holds two encodings of a pass.
    pub fn encode_video(&self, frames_dir: &Path, fps: u32, stem: &Path, gray: bool) -> Result<PathBuf> {
        let codec = self.codec();
        let out = stem.with_extension(codec.ext());
        let pattern = frames_dir.join("%05d.png");
        let vf = if gray { "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=gray" } else { "scale=trunc(iw/2)*2:trunc(ih/2)*2" };
        let fps = fps.to_string();
        self.transcode(codec.enc(), &["-framerate", &fps, "-i", pattern.to_str().context("non-UTF-8 path")?], vf, &[], &out)?;
        for ext in ["mp4", "webm"] {
            let other = stem.with_extension(ext);
            if other != out {
                let _ = std::fs::remove_file(other);
            }
        }
        Ok(out)
    }

    /// Human-facing clip from an arbitrary input (`input_args` ends in `-i SRC`) -> `stem.<mp4|webm>`.
    pub fn encode_clip(&self, input_args: &[&str], vf: &str, extra: &[&str], stem: &Path) -> Result<PathBuf> {
        let codec = self.codec();
        let out = stem.with_extension(codec.ext());
        self.transcode(codec.enc(), input_args, vf, extra, &out)?;
        Ok(out)
    }

    /// Resample to `fps` and pad/trim to exactly `frames` (Wan VACE wants 81..241 frames at 16 fps).
    /// `out` is an mp4 for upload: H.264 if we have it, else MPEG-4 part 2.
    pub fn conform(&self, src: &Path, fps: u32, frames: u32, out: &Path) -> Result<()> {
        let vf = format!("fps={fps},tpad=stop_mode=clone:stop={frames}");
        let n = frames.to_string();
        self.transcode(self.codec().upload_enc(), &["-i", src.to_str().context("non-UTF-8 path")?], &vf, &["-frames:v", &n], out)
    }

    /// An mp4 of `src` for model upload: `src` itself if it is already mp4 (H.264), else a
    /// transcode into `out` (H.264 if we have it, else MPEG-4 part 2).
    pub fn upload_mp4(&self, src: &Path, out: &Path) -> Result<PathBuf> {
        if src.extension().is_some_and(|e| e == "mp4") {
            return Ok(src.to_path_buf());
        }
        if !newer_than(out, src) {
            self.transcode(self.codec().upload_enc(), &["-i", src.to_str().context("non-UTF-8 path")?], "", &[], out)?;
        }
        Ok(out.to_path_buf())
    }
}

/// `out` exists and is at least as new as `src` (a derived file that needs no rebuild).
pub fn newer_than(out: &Path, src: &Path) -> bool {
    let mtime = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    matches!((mtime(out), mtime(src)), (Some(o), Some(s)) if o >= s)
}
