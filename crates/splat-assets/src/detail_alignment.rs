//! Geometry-only yaw registration for uncompressed provider GLBs. No claim of exact reconstruction.
//! Sampled bidirectional surface distance resolves the provider's image-aligned yaw; tight vertex
//! bounds avoid the large overestimate from rotating an accessor's axis-aligned box.
use anyhow::{Context, Result, ensure};
use glam::{DMat4, DQuat, DVec3};
use serde_json::{Value, json};

fn document(bytes: &[u8]) -> Result<(Value, &[u8])> {
    ensure!(bytes.len() >= 20 && &bytes[..4] == b"glTF", "not a GLB");
    let n = u32::from_le_bytes(bytes[12..16].try_into()?) as usize;
    let j = serde_json::from_slice(bytes.get(20..20+n).context("GLB JSON truncated")?)?;
    let at = 20+n;
    let header = bytes.get(at..at+8).context("GLB has no embedded BIN")?;
    ensure!(u32::from_le_bytes(header[4..8].try_into()?) == 0x004e4942, "expected BIN chunk");
    let size = u32::from_le_bytes(header[..4].try_into()?) as usize;
    Ok((j, bytes.get(at+8..at+8+size).context("GLB BIN truncated")?))
}
fn vector(v: &Value, fallback: DVec3) -> DVec3 {
    v.as_array().filter(|a|a.len()==3).map_or(fallback, |a|DVec3::new(a[0].as_f64().unwrap_or(fallback.x),a[1].as_f64().unwrap_or(fallback.y),a[2].as_f64().unwrap_or(fallback.z)))
}
fn points(bytes: &[u8]) -> Result<Vec<DVec3>> {
    let (j,bin) = document(bytes)?;
    let mut out=Vec::new();
    fn visit(i: usize, parent:DMat4, j:&Value, bin:&[u8], out:&mut Vec<DVec3>, depth:usize)->Result<()> {
        ensure!(depth<64,"GLB node hierarchy too deep");
        let n=j["nodes"].get(i).context("invalid node")?;
        let local=if let Some(a)=n["matrix"].as_array() {
            ensure!(a.len()==16,"bad node matrix");let mut m=[0.;16];for(k,v)in a.iter().enumerate(){m[k]=v.as_f64().context("bad matrix number")?;}DMat4::from_cols_array(&m)
        }else{
            let q=if let Some(a)=n["rotation"].as_array(){ensure!(a.len()==4,"bad rotation");DQuat::from_xyzw(a[0].as_f64().context("bad quaternion")?,a[1].as_f64().context("bad quaternion")?,a[2].as_f64().context("bad quaternion")?,a[3].as_f64().context("bad quaternion")?)}else{DQuat::IDENTITY};
            DMat4::from_scale_rotation_translation(vector(&n["scale"],DVec3::ONE),q,vector(&n["translation"],DVec3::ZERO))
        };
        let world=parent*local;
        if let Some(mesh)=n["mesh"].as_u64(){
            for p in j["meshes"][mesh as usize]["primitives"].as_array().context("missing primitives")? {
                ensure!(p["extensions"]["KHR_draco_mesh_compression"].is_null(),"alignment requires uncompressed provider GLB");
                let a=&j["accessors"][p["attributes"]["POSITION"].as_u64().context("missing POSITION")? as usize];
                ensure!(a["componentType"]==5126 && a["type"]=="VEC3" && a["sparse"].is_null(),"alignment requires float VEC3 POSITION");
                let v=&j["bufferViews"][a["bufferView"].as_u64().context("missing vertex buffer")? as usize];
                ensure!(v["buffer"].as_u64().unwrap_or(0)==0 && v["extensions"]["EXT_meshopt_compression"].is_null(),"alignment requires embedded uncompressed positions");
                let count=a["count"].as_u64().context("missing vertex count")? as usize;
                ensure!(count <= 5_000_000 && out.len()+count<=5_000_000,"alignment vertex limit exceeded");
                let stride=v["byteStride"].as_u64().unwrap_or(12) as usize;
                ensure!(stride>=12,"invalid vertex stride");
                let start=v["byteOffset"].as_u64().unwrap_or(0) as usize+a["byteOffset"].as_u64().unwrap_or(0) as usize;
                for k in 0..count {
                    let b=bin.get(start+k*stride..start+k*stride+12).context("truncated positions")?;
                    let xyz=DVec3::new(f32::from_le_bytes(b[0..4].try_into()?) as f64,f32::from_le_bytes(b[4..8].try_into()?) as f64,f32::from_le_bytes(b[8..12].try_into()?) as f64);
                    let xyz=world.transform_point3(xyz);ensure!(xyz.is_finite(),"nonfinite vertex");out.push(xyz);
                }
            }
        }
        if let Some(children)=n["children"].as_array(){for c in children{visit(c.as_u64().context("invalid child")? as usize,world,j,bin,out,depth+1)?;}}
        Ok(())
    }
    let scene=j["scene"].as_u64().unwrap_or(0) as usize;
    for n in j["scenes"][scene]["nodes"].as_array().context("missing scene roots")? {visit(n.as_u64().context("bad root")? as usize,DMat4::IDENTITY,&j,bin,&mut out,0)?;}
    ensure!(!out.is_empty(),"empty provider mesh");Ok(out)
}
fn bounds(p:&[DVec3])->(DVec3,DVec3){p.iter().fold((DVec3::splat(f64::INFINITY),DVec3::splat(f64::NEG_INFINITY)),|(lo,hi),p|(lo.min(*p),hi.max(*p)))}
fn sample(p:&[DVec3])->Result<Vec<DVec3>> {
    let(lo,hi)=bounds(p);let h=hi.y-lo.y;ensure!(h>1e-8,"flat provider mesh");
    let center=DVec3::new((lo.x+hi.x)/2.,lo.y,(lo.z+hi.z)/2.);
    let stride=(p.len()/256).max(1);Ok(p.iter().step_by(stride).take(256).map(|p|(*p-center)/h).collect())
}
fn score(a:&[DVec3],b:&[DVec3])->f64 {
    fn one(a:&[DVec3],b:&[DVec3])->f64 {a.iter().map(|p|b.iter().map(|q|p.distance_squared(*q)).fold(f64::INFINITY,f64::min)).sum::<f64>()/a.len() as f64}
    (one(a,b)+one(b,a))/2.
}

pub fn align(raw:&[u8],preview:&[u8],target:[f32;3])->Result<(Vec<u8>,[f32;3],f64,f64)> {
    let full=points(raw)?;let reference=sample(&points(preview)?)?;let source=sample(&full)?;
    let mut best=(f64::INFINITY,0.);
    let evaluate=|angle:f64|{let q=DQuat::from_rotation_y(angle);let rotated:Vec<_>=source.iter().map(|p|q**p).collect();score(&rotated,&reference)};
    for step in 0..36 {let angle=(step as f64*10.).to_radians();let score=evaluate(angle);if score<best.0{best=(score,angle);}}
    let coarse=best.1;for step in -5..=5 {let angle=coarse+(step as f64).to_radians();let score=evaluate(angle);if score<best.0{best=(score,angle);}}
    ensure!(best.0.sqrt()<=0.15,"detail geometry does not align with preview (normalized RMS {:.3}); keeping preview",best.0.sqrt());
    let rotation=DQuat::from_rotation_y(best.1);let rotated:Vec<_>=full.iter().map(|p|rotation**p).collect();let(lo,hi)=bounds(&rotated);let size=hi-lo;
    ensure!(size.min_element()>1e-8,"degenerate detail bounds");
    let scale=(DVec3::new(target[0] as f64,target[1] as f64,target[2] as f64)/size).min_element();
    let offset=-DVec3::new((lo.x+hi.x)/2.,lo.y,(lo.z+hi.z)/2.)*scale;
    let(j,bin)=document(raw)?;let mut j=j;let scene=j["scene"].as_u64().unwrap_or(0) as usize;let roots=j["scenes"][scene]["nodes"].clone();
    let nodes=j["nodes"].as_array_mut().context("missing nodes")?;let index=nodes.len();
    nodes.push(json!({"name":"detail-preview-frame","children":roots,"translation":offset.to_array(),"rotation":rotation.to_array(),"scale":[scale,scale,scale]}));j["scenes"][scene]["nodes"]=json!([index]);
    let mut json=serde_json::to_vec(&j)?;while json.len()%4!=0{json.push(b' ');}
    let mut out=Vec::new();out.extend_from_slice(b"glTF");out.extend_from_slice(&2u32.to_le_bytes());out.extend_from_slice(&u32::try_from(28+json.len()+bin.len())?.to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());out.extend_from_slice(&0x4e4f534au32.to_le_bytes());out.extend_from_slice(&json);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());out.extend_from_slice(&0x004e4942u32.to_le_bytes());out.extend_from_slice(bin);
    let d=size*scale;Ok((out,[d.x as f32,d.y as f32,d.z as f32],best.1.to_degrees(),best.0.sqrt()))
}
