use std::vec;

use brush_render::gaussian_splats::{SplatRenderMode, Splats};
use brush_render::sh::sh_coeffs_for_degree;
use burn::tensor::Transaction;
use glam::Vec3;
use serde::ser::SerializeStruct;
use serde::{Serialize, Serializer};
use serde_ply::{SerializeError, SerializeOptions};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ExportError {
    #[error("Failed to fetch splat data from GPU")]
    FetchFailed,
    #[error("Failed to convert tensor data to f32 - data may be corrupted")]
    DataConversion,
    #[error("PLY serialization failed: {0}")]
    Serialize(#[from] SerializeError),
}

// Dynamic PLY structure that only includes needed SH coefficients
#[derive(Debug)]
struct DynamicPlyGaussian {
    x: f32,
    y: f32,
    z: f32,
    scale_0: f32,
    scale_1: f32,
    scale_2: f32,
    opacity: f32,
    rot_0: f32,
    rot_1: f32,
    rot_2: f32,
    rot_3: f32,
    f_dc_0: f32,
    f_dc_1: f32,
    f_dc_2: f32,
    rest_coeffs: Vec<f32>,
}

impl Serialize for DynamicPlyGaussian {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Calculate total number of fields: 11 core + 3 DC + rest_coeffs
        let field_count = 14 + self.rest_coeffs.len();
        let mut state = serializer.serialize_struct("DynamicPlyGaussian", field_count)?;

        state.serialize_field("x", &self.x)?;
        state.serialize_field("y", &self.y)?;
        state.serialize_field("z", &self.z)?;
        state.serialize_field("scale_0", &self.scale_0)?;
        state.serialize_field("scale_1", &self.scale_1)?;
        state.serialize_field("scale_2", &self.scale_2)?;
        state.serialize_field("opacity", &self.opacity)?;
        state.serialize_field("rot_0", &self.rot_0)?;
        state.serialize_field("rot_1", &self.rot_1)?;
        state.serialize_field("rot_2", &self.rot_2)?;
        state.serialize_field("rot_3", &self.rot_3)?;

        // Serialize DC components
        state.serialize_field("f_dc_0", &self.f_dc_0)?;
        state.serialize_field("f_dc_1", &self.f_dc_1)?;
        state.serialize_field("f_dc_2", &self.f_dc_2)?;

        // Serialize rest coefficients.
        const SH_NAMES: [&str; 72] = brush_serde_macros::sh_field_names!();
        for (name, val) in SH_NAMES.iter().zip(&self.rest_coeffs) {
            state.serialize_field(name, val)?;
        }

        state.end()
    }
}

#[derive(Serialize)]
struct DynamicPly {
    vertex: Vec<DynamicPlyGaussian>,
}

/// Compact neural appearance (`brush_render::kernels::neural`): standard
/// geometry, the base colour as `f_dc_*` (so SH-only viewers show the
/// view-independent colour exactly), the latents as `f_latent_*`, then the
/// shared decoder once as its own element. Versioned by a header comment.
#[derive(Serialize)]
struct NeuralPly {
    vertex: Vec<NeuralPlyGaussian>,
    neural_decoder: Vec<DecoderWeight>,
}

#[derive(Serialize)]
struct DecoderWeight {
    w: f32,
}

#[derive(Serialize)]
struct NeuralPlyGaussian {
    x: f32,
    y: f32,
    z: f32,
    scale_0: f32,
    scale_1: f32,
    scale_2: f32,
    opacity: f32,
    rot_0: f32,
    rot_1: f32,
    rot_2: f32,
    rot_3: f32,
    f_dc_0: f32,
    f_dc_1: f32,
    f_dc_2: f32,
    f_latent_0: f32,
    f_latent_1: f32,
    f_latent_2: f32,
    f_latent_3: f32,
    f_latent_4: f32,
    f_latent_5: f32,
    f_latent_6: f32,
    f_latent_7: f32,
}

/// Header comment marking a neural-appearance PLY and its decoder layout.
pub const NEURAL_PLY_COMMENT: &str = "Neural appearance: x5b-v1";

async fn read_neural_data(splats: Splats) -> Result<NeuralPly, ExportError> {
    let decoder = splats.decoder.as_ref().expect("neural splats").val();
    let data = Transaction::default()
        .register(splats.transforms.val())
        .register(splats.raw_opacities.val())
        .register(splats.sh_coeffs.val())
        .register(decoder)
        .execute_async()
        .await
        .map_err(|_fetch| ExportError::FetchFailed)?;
    let vecs: Vec<Vec<f32>> = data
        .into_iter()
        .map(|x| x.convert::<f32>().try_into_vec().map_err(|_convert| ExportError::DataConversion))
        .collect::<Result<Vec<_>, _>>()?;
    let [t, opac, rows, decoder]: [Vec<f32>; 4] =
        vecs.try_into().map_err(|_convert| ExportError::DataConversion)?;
    let row = (brush_render::kernels::neural::NEURAL_ROW_COEFFS * 3) as usize;
    let vertex = (0..splats.num_splats() as usize)
        .map(|i| {
            let (t, r) = (&t[i * 10..i * 10 + 10], &rows[i * row..i * row + row]);
            let rn = (t[3] * t[3] + t[4] * t[4] + t[5] * t[5] + t[6] * t[6]).sqrt().max(1e-12);
            NeuralPlyGaussian {
                x: t[0],
                y: t[1],
                z: t[2],
                scale_0: t[7],
                scale_1: t[8],
                scale_2: t[9],
                opacity: opac[i],
                rot_0: t[3] / rn,
                rot_1: t[4] / rn,
                rot_2: t[5] / rn,
                rot_3: t[6] / rn,
                f_dc_0: r[0],
                f_dc_1: r[1],
                f_dc_2: r[2],
                f_latent_0: r[3],
                f_latent_1: r[4],
                f_latent_2: r[5],
                f_latent_3: r[6],
                f_latent_4: r[7],
                f_latent_5: r[8],
                f_latent_6: r[9],
                f_latent_7: r[10],
            }
        })
        .collect();
    let neural_decoder = decoder.into_iter().map(|w| DecoderWeight { w }).collect();
    Ok(NeuralPly { vertex, neural_decoder })
}

async fn read_splat_data(splats: Splats) -> Result<DynamicPly, ExportError> {
    let data = Transaction::default()
        .register(splats.transforms.val())
        .register(splats.raw_opacities.val())
        .register(splats.sh_coeffs.val().permute([0, 2, 1])) // Permute to inria format ([n, channel, coeffs]).
        .execute_async()
        .await
        .map_err(|_fetch| ExportError::FetchFailed)?;

    let vecs: Vec<Vec<f32>> = data
        .into_iter()
        .map(|x| {
            // Convert on the host so half SH export needs no f32 GPU copy.
            x.convert::<f32>().try_into_vec()
                .map_err(|_convert| ExportError::DataConversion)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let [transforms, raw_opacities, sh_coeffs]: [Vec<f32>; 3] = vecs
        .try_into()
        .map_err(|_convert| ExportError::DataConversion)?;

    let sh_coeffs_num = sh_coeffs_for_degree(splats.sh_degree()) as usize;
    let sh_degree = splats.sh_degree();

    // Calculate how many rest coefficients we should export based on the actual SH degree
    // SH coefficients structure:
    // - DC component (degree 0): f_dc_0, f_dc_1, f_dc_2 (always present)
    // - Rest coefficients: f_rest_0 through f_rest_N (degree 1+)
    //
    // Total coefficients per channel = (degree + 1)^2
    // Rest coefficients per channel = total - 1 (excluding DC component)
    // Examples:
    // - Degree 0: 1 total, 0 rest coefficients per channel
    // - Degree 1: 4 total, 3 rest coefficients per channel
    // - Degree 2: 9 total, 8 rest coefficients per channel
    // - Degree 3: 16 total, 15 rest coefficients per channel
    let coeffs_per_channel = sh_coeffs_for_degree(sh_degree) as usize;
    let rest_coeffs_per_channel = coeffs_per_channel - 1;

    let vertices = (0..splats.num_splats())
        .map(|i| {
            let i = i as usize;
            // Read SH data from [coeffs, channel] format
            let sh_start = i * sh_coeffs_num * 3;
            let sh_end = (i + 1) * sh_coeffs_num * 3;
            let splat_sh = &sh_coeffs[sh_start..sh_end];
            let [sh_red, sh_green, sh_blue] = [
                &splat_sh[0..sh_coeffs_num],
                &splat_sh[sh_coeffs_num..sh_coeffs_num * 2],
                &splat_sh[sh_coeffs_num * 2..sh_coeffs_num * 3],
            ];
            let sh_red_rest = if sh_red.len() > 1 && rest_coeffs_per_channel > 0 {
                &sh_red[1..=rest_coeffs_per_channel]
            } else {
                &[]
            };
            let sh_green_rest = if sh_green.len() > 1 && rest_coeffs_per_channel > 0 {
                &sh_green[1..=rest_coeffs_per_channel]
            } else {
                &[]
            };
            let sh_blue_rest = if sh_blue.len() > 1 && rest_coeffs_per_channel > 0 {
                &sh_blue[1..=rest_coeffs_per_channel]
            } else {
                &[]
            };

            let rest_coeffs = [sh_red_rest, sh_green_rest, sh_blue_rest].concat();
            // transforms layout: means(3) + rotations(4) + log_scales(3) = stride 10
            let t = i * 10;
            // Normalize the quaternion before export.
            let (r0, r1, r2, r3): (f32, f32, f32, f32) = (
                transforms[t + 3],
                transforms[t + 4],
                transforms[t + 5],
                transforms[t + 6],
            );
            let rn = (r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3).sqrt().max(1e-12);
            DynamicPlyGaussian {
                x: transforms[t],
                y: transforms[t + 1],
                z: transforms[t + 2],
                scale_0: transforms[t + 7],
                scale_1: transforms[t + 8],
                scale_2: transforms[t + 9],
                rot_0: r0 / rn,
                rot_1: r1 / rn,
                rot_2: r2 / rn,
                rot_3: r3 / rn,
                opacity: raw_opacities[i],
                f_dc_0: sh_red[0],
                f_dc_1: sh_green[0],
                f_dc_2: sh_blue[0],
                rest_coeffs,
            }
        })
        .collect();
    Ok(DynamicPly { vertex: vertices })
}

pub async fn splat_to_ply(splats: Splats, up_axis: Option<Vec3>) -> Result<Vec<u8>, ExportError> {
    // Fold any 3D-filter floor into the stored scales/opacity so the ply holds
    // ordinary derived values — the floor is never written as a separate field.
    let splats = splats.bake_min_scale();
    let sh_degree = splats.sh_degree();
    let neural = splats.decoder.is_some();

    let render_mode_str = match splats.render_mode() {
        SplatRenderMode::Default => "default",
        SplatRenderMode::Mip => "mip",
        SplatRenderMode::Surfel => "surfel",
    };

    let mut comments = vec!["Exported from Brush".to_owned()];
    if let Some(up) = up_axis {
        comments.push(format!("Vertical axis: {} {} {}", up.x, up.y, up.z));
    } else {
        comments.push("Vertical axis: y".to_owned());
    }
    comments.push(format!("SplatRenderMode: {render_mode_str}"));
    let options = |comments| SerializeOptions::binary_le().with_comments(comments);
    if neural {
        comments.push(NEURAL_PLY_COMMENT.to_owned());
        let ply = read_neural_data(splats).await?;
        return Ok(serde_ply::to_bytes(&ply, options(comments))?);
    }
    comments.push(format!("SH degree: {sh_degree}"));
    write_splat_ply(splats, comments).await
}

/// The SH PLY, byte for byte what serializing [`DynamicPly`] gives, written
/// directly: serde_ply's own header (from a one-vertex dummy, count patched)
/// followed by packed little-endian f32 rows encoded in parallel, with no
/// per-splat allocation and the f16 -> f32 SH conversion done in the same pass.
/// Falls back to the serde path if the dummy's layout is not the expected one.
async fn write_splat_ply(splats: Splats, comments: Vec<String>) -> Result<Vec<u8>, ExportError> {
    let options = || SerializeOptions::binary_le().with_comments(comments.clone());
    let n = splats.num_splats() as usize;
    let coeffs = sh_coeffs_for_degree(splats.sh_degree()) as usize;
    let rest = coeffs - 1;
    let stride = 14 + 3 * rest;

    let dummy = DynamicPly {
        vertex: vec![DynamicPlyGaussian {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            scale_0: 0.0,
            scale_1: 0.0,
            scale_2: 0.0,
            opacity: 0.0,
            rot_0: 0.0,
            rot_1: 0.0,
            rot_2: 0.0,
            rot_3: 0.0,
            f_dc_0: 0.0,
            f_dc_1: 0.0,
            f_dc_2: 0.0,
            rest_coeffs: vec![0.0; 3 * rest],
        }],
    };
    let probe = serde_ply::to_bytes(&dummy, options())?;
    const END: &[u8] = b"end_header\n";
    const ONE: &[u8] = b"element vertex 1\n";
    let header_end = probe.windows(END.len()).position(|w| w == END).map(|p| p + END.len());
    let count_at = probe.windows(ONE.len()).position(|w| w == ONE);
    let (Some(header_end), Some(count_at)) = (header_end, count_at) else {
        let ply = read_splat_data(splats).await?;
        return Ok(serde_ply::to_bytes(&ply, options())?);
    };
    if n == 0 || probe.len() != header_end + stride * 4 {
        let ply = read_splat_data(splats).await?;
        return Ok(serde_ply::to_bytes(&ply, options())?);
    }

    let data = Transaction::default()
        .register(splats.transforms.val())
        .register(splats.raw_opacities.val())
        // Read back as stored ([n, coeffs, channel]) and reorder on the CPU: a
        // permuted read is a full contiguous GPU copy of the SH first (96 B per
        // splat at degree 3, the run's board peak at the final export).
        .register(splats.sh_coeffs.val())
        .execute_async()
        .await
        .map_err(|_fetch| ExportError::FetchFailed)?;
    let [t, o, sh]: [burn::tensor::TensorData; 3] =
        data.try_into().map_err(|_convert| ExportError::DataConversion)?;
    let t = t.convert::<f32>();
    let o = o.convert::<f32>();
    let t = t.as_slice::<f32>().map_err(|_convert| ExportError::DataConversion)?;
    let o = o.as_slice::<f32>().map_err(|_convert| ExportError::DataConversion)?;
    let sh_src = match sh.dtype {
        burn::tensor::DType::F16 => ShSrc::F16(
            sh.as_slice::<burn::tensor::f16>()
                .map_err(|_convert| ExportError::DataConversion)?,
        ),
        _ => ShSrc::F32(sh.as_slice::<f32>().map_err(|_convert| ExportError::DataConversion)?),
    };
    if t.len() != n * 10 || o.len() != n || sh_src.len() != n * coeffs * 3 {
        return Err(ExportError::DataConversion);
    }

    let mut out = Vec::with_capacity(header_end + 32 + n * stride * 4);
    out.extend_from_slice(&probe[..count_at]);
    out.extend_from_slice(format!("element vertex {n}\n").as_bytes());
    out.extend_from_slice(&probe[count_at + ONE.len()..header_end]);
    let body_start = out.len();
    out.resize(body_start + n * stride * 4, 0);
    let body = &mut out[body_start..];

    let encode = |first: usize, rows: &mut [u8]| {
        for (r, row) in rows.chunks_exact_mut(stride * 4).enumerate() {
            let i = first + r;
            let tt = &t[i * 10..i * 10 + 10];
            // Same arithmetic as `read_splat_data`, so the bytes match.
            let (r0, r1, r2, r3) = (tt[3], tt[4], tt[5], tt[6]);
            let rn = (r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3).sqrt().max(1e-12);
            let mut w = row.chunks_exact_mut(4);
            let mut put = |v: f32| w.next().unwrap().copy_from_slice(&v.to_le_bytes());
            for v in [tt[0], tt[1], tt[2], tt[7], tt[8], tt[9], o[i], r0 / rn, r1 / rn, r2 / rn, r3 / rn] {
                put(v);
            }
            let base = i * coeffs * 3;
            for ch in 0..3 {
                put(sh_src.get(base + ch));
            }
            for ch in 0..3 {
                for k in 1..=rest {
                    put(sh_src.get(base + k * 3 + ch));
                }
            }
        }
    };
    encode_parallel(body, stride * 4, &encode);
    Ok(out)
}

#[derive(Clone, Copy)]
enum ShSrc<'a> {
    F16(&'a [burn::tensor::f16]),
    F32(&'a [f32]),
}

impl ShSrc<'_> {
    fn len(&self) -> usize {
        match self {
            Self::F16(s) => s.len(),
            Self::F32(s) => s.len(),
        }
    }

    #[inline]
    fn get(&self, i: usize) -> f32 {
        match self {
            Self::F16(s) => s[i].to_f32(),
            Self::F32(s) => s[i],
        }
    }
}

/// Fill `body` (rows of `row_bytes`) with `encode(first_row, rows)` over
/// disjoint row ranges, on scoped threads where available.
fn encode_parallel(body: &mut [u8], row_bytes: usize, encode: &(dyn Fn(usize, &mut [u8]) + Sync)) {
    let rows = body.len() / row_bytes;
    #[cfg(not(target_family = "wasm"))]
    {
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16);
        let per = rows.div_ceil(threads).max(16_384);
        if rows > per {
            std::thread::scope(|scope| {
                for (c, chunk) in body.chunks_mut(per * row_bytes).enumerate() {
                    scope.spawn(move || encode(c * per, chunk));
                }
            });
            return;
        }
    }
    let _ = rows;
    encode(0, body);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::import::load_splat_from_ply;
    use crate::test_utils::create_test_splats;

    use std::io::Cursor;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[cfg(target_family = "wasm")]
    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

    async fn assert_coeffs_match(orig: &Splats, imported: &Splats) {
        let orig_sh: Vec<f32> = orig
            .sh_coeffs
            .val()
            .into_data_async()
            .await
            .unwrap()
            .try_into_vec()
            .expect("Failed to convert SH coefficients to vector");
        let import_sh: Vec<f32> = imported
            .sh_coeffs
            .val()
            .into_data_async()
            .await
            .unwrap()
            .try_into_vec()
            .expect("Failed to convert SH coefficients to vector");

        assert_eq!(orig_sh.len(), import_sh.len());
        for (i, (&orig, &imported)) in orig_sh.iter().zip(import_sh.iter()).enumerate() {
            assert!(
                (orig - imported).abs() < 1e-6_f32,
                "SH coeffs mismatch at index {i}: orig={orig}, imported={imported}",
            );
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_sh_degree_exports() {
        let _device = brush_cube::test_helpers::test_device().await;
        for degree in 0..=2 {
            let splats = create_test_splats(degree);
            assert_eq!(splats.sh_degree(), degree);

            let ply_data = read_splat_data(splats.clone()).await.unwrap();
            let expected_rest_coeffs = if degree == 0 {
                0
            } else {
                (sh_coeffs_for_degree(degree) - 1) * 3
            };

            assert_eq!(
                ply_data.vertex[0].rest_coeffs.len(),
                expected_rest_coeffs as usize
            );
            assert!(splat_to_ply(splats, None).await.is_ok());
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_ply_field_count_matches_sh_degree() {
        let _device = brush_cube::test_helpers::test_device().await;
        let test_cases = [(0, 0), (1, 9), (2, 24)];

        for (degree, expected_rest_fields) in test_cases {
            let splats = create_test_splats(degree);
            let ply_bytes = splat_to_ply(splats, None).await.unwrap();
            let ply_string = String::from_utf8_lossy(&ply_bytes);

            let actual_rest_fields = ply_string.matches("property float f_rest_").count();
            assert_eq!(
                actual_rest_fields, expected_rest_fields,
                "Degree {degree} should have {expected_rest_fields} f_rest_ fields",
            );

            assert!(ply_string.contains("f_dc_0"));
            if expected_rest_fields > 0 {
                assert!(ply_string.contains("f_rest_0"));
                assert!(!ply_string.contains(&format!("f_rest_{expected_rest_fields}")));
            } else {
                assert!(!ply_string.contains("f_rest_0"));
            }
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_roundtrip_sh_coefficient_ordering() {
        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();

        for degree in [0, 1, 2] {
            let original_splats = create_test_splats(degree);
            let ply_bytes = splat_to_ply(original_splats.clone(), None)
                .await
                .expect("Failed to serialize splats");

            let cursor = Cursor::new(ply_bytes);
            let imported_message = load_splat_from_ply(cursor, None)
                .await
                .expect("Failed to deserialize splats");
            let imported_splats = imported_message
                .data
                .into_splats(&device, SplatRenderMode::Default);

            assert_eq!(imported_splats.sh_degree(), degree);
            assert_coeffs_match(&original_splats, &imported_splats).await;
        }
    }

    #[wasm_bindgen_test(unsupported = tokio::test)]
    async fn test_export_roundtrip_multiple_splats() {
        use crate::test_utils::create_test_splats_with_count;

        let device: burn::tensor::Device = brush_cube::test_helpers::test_device().await.into();
        let num_splats = 100;

        for degree in [0, 1, 2, 3] {
            let original = create_test_splats_with_count(degree, num_splats);
            assert_eq!(original.num_splats(), num_splats as u32);

            let ply_bytes = splat_to_ply(original.clone(), None)
                .await
                .expect("Failed to export splats");

            assert!(!ply_bytes.is_empty(), "Exported PLY should not be empty");

            let cursor = Cursor::new(ply_bytes);
            let imported_message = load_splat_from_ply(cursor, None)
                .await
                .expect("Failed to reimport exported splats");
            let imported = imported_message
                .data
                .into_splats(&device, SplatRenderMode::Default);

            assert_eq!(
                imported.num_splats(),
                num_splats as u32,
                "Splat count mismatch after roundtrip"
            );
            assert_eq!(imported.sh_degree(), degree);
            assert_coeffs_match(&original, &imported).await;
        }
    }
}
