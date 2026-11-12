use crate::util::atomic_write;
use anyhow::Result;
use regex::Regex;
use serde_json::{Value, json};
use std::{fs, path::Path, sync::LazyLock};

static STEP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"step\s+(\d+)/(\d+)\s+\([^)]*\)\s+splats\s+(\d+)\s+\[elapsed\s+([\d:.]+)\s*\|\s*ETA\s+([\d:.\-]+)\](.*)").unwrap()
});
static METRIC: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\w+)=([-+\d.eE]+)").unwrap());
fn seconds(s: &str) -> Option<f64> {
    let fields: Vec<f64> = s
        .split(':')
        .map(str::parse)
        .collect::<std::result::Result<_, _>>()
        .ok()?;
    (fields.len() == 3).then(|| fields[0] * 3600.0 + fields[1] * 60.0 + fields[2])
}
/// Parse Spirula's training-batch progress. Initial zero PSNR/SSIM means unmeasured.
pub fn parse_progress(line: &str) -> Option<Value> {
    let c = STEP.captures(line)?;
    let mut row = json!({"step":c[1].parse::<u64>().ok()?,"total_steps":c[2].parse::<u64>().ok()?,"splats":c[3].parse::<u64>().ok()?,"elapsed_s":seconds(&c[4]),"eta_s":seconds(&c[5]),"scope":"training_batch"});
    for cap in METRIC.captures_iter(&c[6]) {
        if let Ok(v) = cap[2].parse::<f64>() {
            row[cap[1].to_owned()] = if v.is_finite() { json!(v) } else { Value::Null };
        }
    }
    if row["psnr"].as_f64() == Some(0.0) && row["ssim"].as_f64() == Some(0.0) {
        row["psnr"] = Value::Null;
        row["ssim"] = Value::Null;
    }
    Some(row)
}
pub(crate) fn refresh(log: &Path, series: &Path) -> Result<Value> {
    let bytes = match fs::read(log) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Value::Null),
        Err(e) => return Err(e.into()),
    };
    let text = String::from_utf8_lossy(&bytes);
    let mut out = Vec::new();
    let mut last = Value::Null;
    // Only newline/CR-terminated records are complete while a subprocess is writing.
    for line in text
        .split_inclusive(['\n', '\r'])
        .filter(|l| l.ends_with(['\n', '\r']))
    {
        if let Some(row) = parse_progress(line) {
            serde_json::to_writer(&mut out, &row)?;
            out.push(b'\n');
            last = row;
        }
    }
    atomic_write(series, &out)?;
    Ok(last)
}
