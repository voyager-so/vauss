//! Quadric-error edge-collapse decimation (Garland & Heckbert 1997), run as threshold sweeps:
//! each sweep collapses every edge whose error is under a rising threshold, skipping edges whose
//! triangles were touched earlier in the sweep and collapses that would flip a face. Border
//! edges are kept so holes don't grow.

use glam::DVec3;

use crate::Mesh;

#[derive(Clone, Copy, Default)]
struct Quadric([f64; 10]);

impl Quadric {
    fn plane(n: DVec3, d: f64) -> Self {
        let (a, b, c) = (n.x, n.y, n.z);
        Self([a * a, a * b, a * c, a * d, b * b, b * c, b * d, c * c, c * d, d * d])
    }

    fn add(self, o: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] + o.0[i]))
    }

    #[rustfmt::skip]
    fn det(&self, a: [usize; 9]) -> f64 {
        let m = |i: usize| self.0[a[i]];
        m(0) * m(4) * m(8) + m(2) * m(3) * m(7) + m(1) * m(5) * m(6)
            - m(2) * m(4) * m(6) - m(0) * m(5) * m(7) - m(1) * m(3) * m(8)
    }

    fn error(&self, p: DVec3) -> f64 {
        let q = &self.0;
        let (x, y, z) = (p.x, p.y, p.z);
        q[0] * x * x + 2.0 * q[1] * x * y + 2.0 * q[2] * x * z + 2.0 * q[3] * x + q[4] * y * y
            + 2.0 * q[5] * y * z + 2.0 * q[6] * y + q[7] * z * z + 2.0 * q[8] * z + q[9]
    }
}

struct Tri {
    v: [usize; 3],
    err: [f64; 4],
    deleted: bool,
    dirty: bool,
    n: DVec3,
}

struct Vert {
    p: DVec3,
    color: [f64; 3],
    q: Quadric,
    tstart: usize,
    tcount: usize,
    border: bool,
}

#[derive(Clone, Copy)]
struct Ref {
    tid: usize,
    tvertex: usize,
}

struct State {
    tris: Vec<Tri>,
    verts: Vec<Vert>,
    refs: Vec<Ref>,
}

/// Reduce `mesh` to at most `target` triangles (no-op if already there). Positions are
/// optimised per collapse; colours are averaged.
pub fn decimate(mesh: &mut Mesh, target: usize) {
    if mesh.triangles.len() <= target || mesh.triangles.is_empty() {
        return;
    }
    // Normalise to a unit box so the error thresholds are scale-free.
    let Some(b) = mesh.bounds() else { return };
    let center = ((b.min + b.max) * 0.5).as_dvec3();
    let scale = (b.max - b.min).max_element().max(1e-12) as f64;
    let mut s = State {
        verts: mesh
            .positions
            .iter()
            .zip(&mesh.colors)
            .map(|(p, c)| Vert {
                p: (p.as_dvec3() - center) / scale,
                color: c.map(f64::from),
                q: Quadric::default(),
                tstart: 0,
                tcount: 0,
                border: false,
            })
            .collect(),
        tris: mesh.triangles.iter().map(|t| Tri { v: t.map(|i| i as usize), err: [0.0; 4], deleted: false, dirty: false, n: DVec3::ZERO }).collect(),
        refs: vec![],
    };
    let mut deleted_tris = 0usize;
    let start_count = s.tris.len();
    let mut del0 = vec![];
    let mut del1 = vec![];
    for iteration in 0..200 {
        if start_count - deleted_tris <= target {
            break;
        }
        if iteration % 5 == 0 {
            s.update_mesh(iteration);
        }
        for t in &mut s.tris {
            t.dirty = false;
        }
        let threshold = 1e-9 * ((iteration + 3) as f64).powi(7);
        for ti in 0..s.tris.len() {
            {
                let t = &s.tris[ti];
                if t.err[3] > threshold || t.deleted || t.dirty {
                    continue;
                }
            }
            for j in 0..3 {
                if s.tris[ti].err[j] >= threshold {
                    continue;
                }
                let i0 = s.tris[ti].v[j];
                let i1 = s.tris[ti].v[(j + 1) % 3];
                if s.verts[i0].border || s.verts[i1].border {
                    continue;
                }
                let (_, p) = s.edge_error(i0, i1);
                del0.clear();
                del0.resize(s.verts[i0].tcount, false);
                del1.clear();
                del1.resize(s.verts[i1].tcount, false);
                if s.flipped(p, i1, i0, &mut del0) || s.flipped(p, i0, i1, &mut del1) {
                    continue;
                }
                s.verts[i0].p = p;
                s.verts[i0].q = s.verts[i0].q.add(s.verts[i1].q);
                let c1 = s.verts[i1].color;
                for (c, o) in s.verts[i0].color.iter_mut().zip(c1) {
                    *c = (*c + o) * 0.5;
                }
                let tstart = s.refs.len();
                deleted_tris += s.update_triangles(i0, i0, &del0);
                deleted_tris += s.update_triangles(i0, i1, &del1);
                let tcount = s.refs.len() - tstart;
                s.verts[i0].tstart = tstart;
                s.verts[i0].tcount = tcount;
                break;
            }
            if start_count - deleted_tris <= target {
                break;
            }
        }
    }
    // Write back the surviving triangles and the vertices they use.
    mesh.positions = s.verts.iter().map(|v| (v.p * scale + center).as_vec3()).collect();
    mesh.colors = s.verts.iter().map(|v| v.color.map(|c| c.round().clamp(0.0, 255.0) as u8)).collect();
    mesh.triangles = s.tris.iter().filter(|t| !t.deleted).map(|t| t.v.map(|i| i as u32)).collect();
    mesh.compact();
}

impl State {
    fn edge_error(&self, a: usize, b: usize) -> (f64, DVec3) {
        let q = self.verts[a].q.add(self.verts[b].q);
        let border = self.verts[a].border && self.verts[b].border;
        let det = q.det([0, 1, 2, 1, 4, 5, 2, 5, 7]);
        if det.abs() > 1e-15 && !border {
            let p = DVec3::new(
                -1.0 / det * q.det([1, 2, 3, 4, 5, 6, 5, 7, 8]),
                1.0 / det * q.det([0, 2, 3, 1, 5, 6, 2, 7, 8]),
                -1.0 / det * q.det([0, 1, 3, 1, 4, 6, 2, 5, 8]),
            );
            // Guard against wild solutions from near-singular quadrics.
            let (pa, pb) = (self.verts[a].p, self.verts[b].p);
            if p.distance(pa) <= 2.0 * pa.distance(pb) + 1e-9 {
                return (q.error(p), p);
            }
        }
        let (pa, pb) = (self.verts[a].p, self.verts[b].p);
        let pm = (pa + pb) * 0.5;
        [(q.error(pa), pa), (q.error(pb), pb), (q.error(pm), pm)].into_iter().min_by(|x, y| x.0.total_cmp(&y.0)).expect("three candidates")
    }

    /// Would moving `i0` to `p` (collapsing edge to `i1`) flip a face around `i0`? Marks the
    /// triangles that contain the edge (they disappear) in `deleted`.
    fn flipped(&self, p: DVec3, i1: usize, i0: usize, deleted: &mut [bool]) -> bool {
        let v0 = &self.verts[i0];
        for k in 0..v0.tcount {
            let r = self.refs[v0.tstart + k];
            let t = &self.tris[r.tid];
            if t.deleted {
                continue;
            }
            let s = r.tvertex;
            let id1 = t.v[(s + 1) % 3];
            let id2 = t.v[(s + 2) % 3];
            if id1 == i1 || id2 == i1 {
                deleted[k] = true;
                continue;
            }
            let d1 = (self.verts[id1].p - p).normalize_or_zero();
            let d2 = (self.verts[id2].p - p).normalize_or_zero();
            if d1.dot(d2).abs() > 0.999 {
                return true;
            }
            let n = d1.cross(d2).normalize_or_zero();
            deleted[k] = false;
            if n.dot(t.n) < 0.2 {
                return true;
            }
        }
        false
    }

    /// Re-point the triangles around `v` at `i0`, dropping those marked deleted. Returns the
    /// number of triangles deleted.
    fn update_triangles(&mut self, i0: usize, v: usize, deleted: &[bool]) -> usize {
        let mut removed = 0;
        let (tstart, tcount) = (self.verts[v].tstart, self.verts[v].tcount);
        for k in 0..tcount {
            let r = self.refs[tstart + k];
            if self.tris[r.tid].deleted {
                continue;
            }
            if deleted[k] {
                self.tris[r.tid].deleted = true;
                removed += 1;
                continue;
            }
            self.tris[r.tid].v[r.tvertex] = i0;
            self.tris[r.tid].dirty = true;
            let tv = self.tris[r.tid].v;
            let e0 = self.edge_error(tv[0], tv[1]).0;
            let e1 = self.edge_error(tv[1], tv[2]).0;
            let e2 = self.edge_error(tv[2], tv[0]).0;
            self.tris[r.tid].err = [e0, e1, e2, e0.min(e1).min(e2)];
            self.refs.push(r);
        }
        removed
    }

    /// Compact deleted triangles and rebuild vertex → triangle references. On the first call
    /// also build quadrics, face normals, edge errors and border flags.
    fn update_mesh(&mut self, iteration: usize) {
        if iteration > 0 {
            self.tris.retain(|t| !t.deleted);
        }
        let rebuild_refs = |s: &mut Self| {
            for v in &mut s.verts {
                v.tstart = 0;
                v.tcount = 0;
            }
            for t in &s.tris {
                for &v in &t.v {
                    s.verts[v].tcount += 1;
                }
            }
            let mut start = 0;
            for v in &mut s.verts {
                v.tstart = start;
                start += v.tcount;
                v.tcount = 0;
            }
            s.refs.clear();
            s.refs.resize(start, Ref { tid: 0, tvertex: 0 });
            for (ti, t) in s.tris.iter().enumerate() {
                for (j, &v) in t.v.iter().enumerate() {
                    let vert = &mut s.verts[v];
                    s.refs[vert.tstart + vert.tcount] = Ref { tid: ti, tvertex: j };
                    vert.tcount += 1;
                }
            }
        };
        if iteration == 0 {
            for t in &mut self.tris {
                let [a, b, c] = t.v.map(|i| self.verts[i].p);
                let n = (b - a).cross(c - a).normalize_or_zero();
                t.n = n;
                let q = Quadric::plane(n, -n.dot(a));
                for &i in &t.v {
                    self.verts[i].q = self.verts[i].q.add(q);
                }
            }
            for ti in 0..self.tris.len() {
                let tv = self.tris[ti].v;
                let e: [f64; 3] = std::array::from_fn(|j| self.edge_error(tv[j], tv[(j + 1) % 3]).0);
                self.tris[ti].err = [e[0], e[1], e[2], e[0].min(e[1]).min(e[2])];
            }
            rebuild_refs(self);
            // Border vertices: an edge used by exactly one triangle.
            let mut nbrs: Vec<(usize, u32)> = vec![];
            for vi in 0..self.verts.len() {
                nbrs.clear();
                let v = &self.verts[vi];
                for k in 0..v.tcount {
                    let t = &self.tris[self.refs[v.tstart + k].tid];
                    for &id in &t.v {
                        if id == vi {
                            continue;
                        }
                        match nbrs.iter_mut().find(|(n, _)| *n == id) {
                            Some((_, c)) => *c += 1,
                            None => nbrs.push((id, 1)),
                        }
                    }
                }
                for &(id, c) in &nbrs {
                    if c == 1 {
                        self.verts[id].border = true;
                        self.verts[vi].border = true;
                    }
                }
            }
        } else {
            // Face normals follow the moved vertices.
            for t in &mut self.tris {
                let [a, b, c] = t.v.map(|i| self.verts[i].p);
                t.n = (b - a).cross(c - a).normalize_or_zero();
            }
            rebuild_refs(self);
        }
    }
}
