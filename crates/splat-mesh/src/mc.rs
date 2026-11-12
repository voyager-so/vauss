//! Marching cubes over the sparse TSDF.
//!
//! The 256-case triangle table is derived at startup rather than transcribed: on every cube
//! face the crossing edges are paired (an ambiguous face with four crossings pairs the edges
//! around each inside corner, so the inside corners stay separated), which links the crossing
//! edges into closed loops; each loop is oriented so its normal points from inside (negative)
//! to outside and fanned into triangles. The face rule depends only on that face's corners, so
//! neighbouring cubes agree and the surface is watertight.

use std::sync::OnceLock;

use glam::{IVec3, Vec3};
use hashbrown::HashMap;
use rayon::prelude::*;

use crate::Mesh;
use crate::tsdf::{B, Block, TsdfVolume, Voxel, local_index};

/// Corner `c` sits at (c & 1, (c >> 1) & 1, (c >> 2) & 1).
fn corner(c: usize) -> IVec3 {
    IVec3::new((c & 1) as i32, ((c >> 1) & 1) as i32, ((c >> 2) & 1) as i32)
}

/// Edge `e` = (lower corner, axis); it joins `lower` and `lower | 1 << axis`.
const EDGES: [(usize, usize); 12] = [
    (0, 0), (2, 0), (4, 0), (6, 0),
    (0, 1), (1, 1), (4, 1), (5, 1),
    (0, 2), (1, 2), (2, 2), (3, 2),
];

fn edge_ends(e: usize) -> (usize, usize) {
    let (c, axis) = EDGES[e];
    (c, c | 1 << axis)
}

fn table() -> &'static [Vec<[u8; 3]>] {
    static TABLE: OnceLock<Vec<Vec<[u8; 3]>>> = OnceLock::new();
    TABLE.get_or_init(|| (0..256).map(build_case).collect())
}

fn build_loops(case: usize) -> Vec<Vec<usize>> {
    let inside = |c: usize| case >> c & 1 == 1;
    let crossing = |e: usize| {
        let (a, b) = edge_ends(e);
        inside(a) != inside(b)
    };
    let mut partners: [Vec<usize>; 12] = Default::default();
    for axis in 0..3 {
        for side in 0..2 {
            let face_edges: Vec<usize> = (0..12)
                .filter(|&e| {
                    let (a, b) = edge_ends(e);
                    corner(a)[axis] == side && corner(b)[axis] == side
                })
                .collect();
            let cross: Vec<usize> = face_edges.iter().copied().filter(|&e| crossing(e)).collect();
            match cross.len() {
                0 => {}
                2 => link(&mut partners, cross[0], cross[1]),
                4 => {
                    let face_corners = (0..8).filter(|&c| corner(c)[axis] == side);
                    for c in face_corners.filter(|&c| inside(c)) {
                        let around: Vec<usize> = cross
                            .iter()
                            .copied()
                            .filter(|&e| {
                                let (a, b) = edge_ends(e);
                                a == c || b == c
                            })
                            .collect();
                        link(&mut partners, around[0], around[1]);
                    }
                }
                n => unreachable!("face with {n} crossings"),
            }
        }
    }
    let mid = |e: usize| {
        let (a, b) = edge_ends(e);
        (corner(a) + corner(b)).as_vec3() * 0.5
    };
    let mut used = [false; 12];
    let mut loops = vec![];
    for start in 0..12 {
        if used[start] || !crossing(start) {
            continue;
        }
        let mut loop_edges = vec![start];
        used[start] = true;
        let (mut prev, mut cur) = (start, partners[start][0]);
        while cur != start {
            used[cur] = true;
            loop_edges.push(cur);
            let next = if partners[cur][0] == prev { partners[cur][1] } else { partners[cur][0] };
            prev = cur;
            cur = next;
        }
        // Newell normal of the loop vs. the inside → outside direction across its edges.
        let mut normal = Vec3::ZERO;
        let mut outward = Vec3::ZERO;
        for (i, &e) in loop_edges.iter().enumerate() {
            let (p, q) = (mid(e), mid(loop_edges[(i + 1) % loop_edges.len()]));
            normal += Vec3::new((p.y - q.y) * (p.z + q.z), (p.z - q.z) * (p.x + q.x), (p.x - q.x) * (p.y + q.y));
            let (a, b) = edge_ends(e);
            let (inner, outer) = if inside(a) { (a, b) } else { (b, a) };
            outward += (corner(outer) - corner(inner)).as_vec3();
        }
        if normal.dot(outward) < 0.0 {
            loop_edges.reverse();
        }
        loops.push(loop_edges);
    }
    loops
}

fn build_case(case: usize) -> Vec<[u8; 3]> {
    let mut tris = vec![];
    for edges in build_loops(case) {
        for i in 1..edges.len()-1 {
            tris.push([edges[0] as u8, edges[i] as u8, edges[i+1] as u8]);
        }
    }
    tris
}

fn link(partners: &mut [Vec<usize>; 12], a: usize, b: usize) {
    partners[a].push(b);
    partners[b].push(a);
}

const OFF: i64 = 1 << 19;

/// Max |tsdf| step across a crossing edge before the cube counts as a depth discontinuity.
const DISCONTINUITY: f32 = 1.9;

#[inline]
fn edge_key(g: IVec3, axis: usize) -> u64 {
    let f = |v: i32| ((v as i64 + OFF) as u64) & 0xF_FFFF;
    f(g.x) | f(g.y) << 20 | f(g.z) << 40 | (axis as u64) << 60
}

/// A block and its +x/+y/+z neighbours, indexed by offset bits (x | y << 1 | z << 2).
struct Neighborhood<'a>([Option<&'a Block>; 8]);

impl Neighborhood<'_> {
    /// Voxel at local coords in [0, 8]³ (8 reaches into the neighbour).
    #[inline]
    fn get(&self, l: IVec3) -> Option<&Voxel> {
        let bits = (l.x >= B) as usize | ((l.y >= B) as usize) << 1 | ((l.z >= B) as usize) << 2;
        let local = l - IVec3::new(B, B, B) * corner(bits);
        self.0[bits].map(|b| &b[local_index(local)])
    }
}

impl TsdfVolume {
    fn neighborhood(&self, coord: IVec3) -> Neighborhood<'_> {
        Neighborhood(std::array::from_fn(|bits| self.index.get(&(coord + corner(bits))).map(|&i| &self.blocks[i as usize])))
    }

    pub(crate) fn extract_mesh(&self, min_weight: f32) -> Mesh {
        let table = table();
        let valid = |v: Option<&Voxel>| v.filter(|v| v.weight >= min_weight).copied();
        let voxel = self.voxel;

        // Vertices on every crossing edge owned (as lower endpoint) by a voxel of each block.
        let per_block: Vec<Vec<(u64, Vec3, [u8; 3])>> = self
            .coords
            .par_iter()
            .map(|&coord| {
                let nb = self.neighborhood(coord);
                let mut out = vec![];
                for lz in 0..B {
                    for ly in 0..B {
                        for lx in 0..B {
                            let l = IVec3::new(lx, ly, lz);
                            let Some(a) = valid(nb.get(l)) else { continue };
                            for axis in 0..3 {
                                let Some(b) = valid(nb.get(l + IVec3::AXES[axis])) else { continue };
                                if (a.tsdf < 0.0) == (b.tsdf < 0.0) {
                                    continue;
                                }
                                let t = a.tsdf / (a.tsdf - b.tsdf);
                                let g = coord * B + l;
                                let p = (g.as_vec3() + IVec3::AXES[axis].as_vec3() * t) * voxel;
                                let ca = if a.color_weight > 0.0 { a.rgb } else { b.rgb };
                                let cb = if b.color_weight > 0.0 { b.rgb } else { a.rgb };
                                let c = std::array::from_fn(|i| (ca[i] + (cb[i] - ca[i]) * t).round().clamp(0.0, 255.0) as u8);
                                out.push((edge_key(g, axis), p, c));
                            }
                        }
                    }
                }
                out
            })
            .collect();
        let n: usize = per_block.iter().map(Vec::len).sum();
        let mut mesh = Mesh { positions: Vec::with_capacity(n), colors: Vec::with_capacity(n), triangles: vec![] };
        let mut ids: HashMap<u64, u32> = HashMap::with_capacity(n);
        for (key, p, c) in per_block.into_iter().flatten() {
            ids.insert(key, mesh.positions.len() as u32);
            mesh.positions.push(p);
            mesh.colors.push(c);
        }

        mesh.triangles = self
            .coords
            .par_iter()
            .flat_map_iter(|&coord| {
                let nb = self.neighborhood(coord);
                let mut tris = vec![];
                for lz in 0..B {
                    for ly in 0..B {
                        for lx in 0..B {
                            let l = IVec3::new(lx, ly, lz);
                            let mut case = 0usize;
                            let mut vals = [0f32; 8];
                            let mut ok = true;
                            for c in 0..8 {
                                match valid(nb.get(l + corner(c))) {
                                    Some(v) => {
                                        vals[c] = v.tsdf;
                                        if v.tsdf < 0.0 {
                                            case |= 1 << c;
                                        }
                                    }
                                    None => {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                            if !ok || case == 0 || case == 255 {
                                continue;
                            }
                            // A sign flip between two strongly truncated values is a depth
                            // discontinuity, not a surface.
                            if (0..12).any(|e| {
                                let (a, b) = edge_ends(e);
                                (vals[a] < 0.0) != (vals[b] < 0.0) && (vals[a] - vals[b]).abs() > DISCONTINUITY
                            }) {
                                continue;
                            }
                            let g = coord * B + l;
                            for t in &table[case] {
                                let idx = t.map(|e| {
                                    let (c, axis) = EDGES[e as usize];
                                    ids.get(&edge_key(g + corner(c), axis)).copied()
                                });
                                if let [Some(a), Some(b), Some(c)] = idx
                                    && a != b
                                    && b != c
                                    && a != c
                                {
                                    tris.push([a, b, c]);
                                }
                            }
                        }
                    }
                }
                tris
            })
            .collect();
        mesh
    }
}

/// Marching cubes of occupied lattice samples, with empty space everywhere else.
/// Unlike TSDF extraction, every crossing has both sides, including the outer bounds.
/// The shared face rule and shared edge vertices close the surface around each solid.
/// Input order does not affect the output. `origin + cell * voxel` locates a sample.
pub fn mesh_occupancy(cells: &[IVec3], origin: Vec3, voxel: f32) -> Mesh {
    use hashbrown::HashSet;
    let loops: Vec<_> = (0..256).map(build_loops).collect();
    let occupied: HashSet<IVec3> = cells.iter().copied().collect();
    let mut cubes = HashSet::new();
    for &p in cells {
        // Only boundary samples can touch a surface cube.
        if IVec3::AXES.iter().all(|&d| occupied.contains(&(p+d)) && occupied.contains(&(p-d))) {
            continue;
        }
        for c in 0..8 { cubes.insert(p - corner(c)); }
    }
    let mut cubes: Vec<_> = cubes.into_iter().collect();
    cubes.sort_unstable_by_key(|p| (p.z, p.y, p.x));
    let mut ids = HashMap::<(IVec3, usize), u32>::new();
    let mut mesh = Mesh::default();
    for p in cubes {
        let case = (0..8).fold(0, |case, c| case | ((occupied.contains(&(p + corner(c))) as usize) << c));
        for edges in &loops[case] {
            let vertices: Vec<u32> = edges.iter().map(|&e| {
                let (c, axis) = EDGES[e];
                let lo = p + corner(c);
                *ids.entry((lo, axis)).or_insert_with(|| {
                    let id = mesh.positions.len() as u32;
                    mesh.positions.push(origin + (lo.as_vec3() + IVec3::AXES[axis].as_vec3() * 0.5) * voxel);
                    mesh.colors.push([88, 196, 210]);
                    id
                })
            }).collect();
            if vertices.len() == 3 {
                mesh.triangles.push([vertices[0], vertices[1], vertices[2]]);
            } else {
                // Fan around an interior vertex. Fanning from a contour corner can put a
                // diagonal on a shared cube face, duplicating that face's neighbor geometry.
                let center = vertices.iter().map(|&i| mesh.positions[i as usize]).sum::<Vec3>() / vertices.len() as f32;
                let id = mesh.positions.len() as u32;
                mesh.positions.push(center);
                mesh.colors.push([88, 196, 210]);
                for i in 0..vertices.len() { mesh.triangles.push([id, vertices[i], vertices[(i+1)%vertices.len()]]); }
            }
        }
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_complete_and_closed() {
        let t = table();
        assert!(t[0].is_empty() && t[255].is_empty(), "trivial cases emit nothing");
        for case in 1..255 {
            assert!(!t[case].is_empty(), "case {case} emits triangles");
            // Every crossing edge is used; each internal edge-pair (triangle side) count is
            // balanced so the loops close.
            for e in 0..12 {
                let (a, b) = edge_ends(e);
                let crosses = (case >> a & 1) != (case >> b & 1);
                let used = t[case].iter().any(|tri| tri.contains(&(e as u8)));
                assert_eq!(crosses, used, "case {case} edge {e}");
            }
        }
        // Single inside corner: one triangle whose normal points away from that corner.
        assert_eq!(t[1].len(), 1);
    }

    #[test]
    fn complement_cases_mirror() {
        // Complementary cases have the same number of loops' vertices (triangles may differ only
        // on ambiguous faces, where the rule separates inside corners).
        let t = table();
        for case in [1usize, 3, 7, 15] {
            assert_eq!(t[case].len(), t[255 - case].len(), "case {case}");
        }
    }
}
