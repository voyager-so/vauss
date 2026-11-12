//! Mesh cleanup: small-component removal and vertex normals.

use glam::Vec3;

use crate::Mesh;

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let p = parent[parent[x as usize] as usize];
        parent[x as usize] = p;
        x = p;
    }
    x
}

/// Drop connected components (sharing vertices) with fewer triangles than `ratio` times the
/// largest component's. Returns how many components were removed. Vertices are not compacted.
pub fn remove_small_components(mesh: &mut Mesh, ratio: f32) -> usize {
    if mesh.triangles.is_empty() || ratio <= 0.0 {
        return 0;
    }
    let mut parent: Vec<u32> = (0..mesh.positions.len() as u32).collect();
    for t in &mesh.triangles {
        let a = find(&mut parent, t[0]);
        for &v in &t[1..] {
            let b = find(&mut parent, v);
            if a != b {
                parent[b as usize] = a;
            }
        }
    }
    let mut count = vec![0u32; mesh.positions.len()];
    let roots: Vec<u32> = mesh.triangles.iter().map(|t| find(&mut parent, t[0])).collect();
    for &r in &roots {
        count[r as usize] += 1;
    }
    let largest = count.iter().copied().max().unwrap_or(0);
    let min = (largest as f32 * ratio).ceil() as u32;
    let removed = count.iter().filter(|&&c| c > 0 && c < min).count();
    let mut keep = roots.iter().map(|&r| count[r as usize] >= min);
    mesh.triangles.retain(|_| keep.next().unwrap_or(false));
    removed
}

/// Area-weighted vertex normals (unit length; zero for unused vertices).
pub fn vertex_normals(mesh: &Mesh) -> Vec<Vec3> {
    let mut n = vec![Vec3::ZERO; mesh.positions.len()];
    for t in &mesh.triangles {
        let [a, b, c] = t.map(|i| mesh.positions[i as usize]);
        let f = (b - a).cross(c - a);
        for &i in t {
            n[i as usize] += f;
        }
    }
    n.iter().map(|v| v.normalize_or_zero()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_small_islands() {
        let mut mesh = Mesh::default();
        // A 10-triangle fan and a lone triangle.
        mesh.positions.push(Vec3::ZERO);
        for i in 0..11 {
            let a = i as f32 * 0.5;
            mesh.positions.push(Vec3::new(a.cos(), a.sin(), 0.0));
        }
        for i in 1..11 {
            mesh.triangles.push([0, i, i + 1]);
        }
        let base = mesh.positions.len() as u32;
        mesh.positions.extend([Vec3::X * 5.0, Vec3::new(5.0, 1.0, 0.0), Vec3::new(6.0, 0.0, 0.0)]);
        mesh.triangles.push([base, base + 1, base + 2]);
        mesh.colors = vec![[0; 3]; mesh.positions.len()];
        assert_eq!(remove_small_components(&mut mesh, 0.2), 1);
        assert_eq!(mesh.triangles.len(), 10);
        mesh.compact();
        assert_eq!(mesh.positions.len(), 12);
    }
}

/// Taubin λ|μ smoothing (Taubin 1995): alternating shrink and inflate Laplacian steps, which
/// removes high-frequency noise without the volume loss of plain Laplacian smoothing.
/// Border vertices stay fixed.
pub fn smooth_taubin(mesh: &mut Mesh, iterations: u32) {
    use rayon::prelude::*;
    if iterations == 0 || mesh.triangles.is_empty() {
        return;
    }
    let n = mesh.positions.len();
    // Undirected edges, deduplicated, as CSR adjacency; edges used once mark border vertices.
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(mesh.triangles.len() * 3);
    for t in &mesh.triangles {
        for j in 0..3 {
            let (a, b) = (t[j], t[(j + 1) % 3]);
            edges.push((a.min(b), a.max(b)));
        }
    }
    edges.par_sort_unstable();
    let mut border = vec![false; n];
    let mut uniq = Vec::with_capacity(edges.len() / 2);
    let mut i = 0;
    while i < edges.len() {
        let mut j = i + 1;
        while j < edges.len() && edges[j] == edges[i] {
            j += 1;
        }
        if j - i == 1 {
            border[edges[i].0 as usize] = true;
            border[edges[i].1 as usize] = true;
        }
        uniq.push(edges[i]);
        i = j;
    }
    let mut start = vec![0u32; n + 1];
    for &(a, b) in &uniq {
        start[a as usize + 1] += 1;
        start[b as usize + 1] += 1;
    }
    for v in 0..n {
        start[v + 1] += start[v];
    }
    let mut fill = start.clone();
    let mut adj = vec![0u32; start[n] as usize];
    for &(a, b) in &uniq {
        adj[fill[a as usize] as usize] = b;
        fill[a as usize] += 1;
        adj[fill[b as usize] as usize] = a;
        fill[b as usize] += 1;
    }
    let (lambda, mu) = (0.5f32, -0.53f32);
    let mut next = mesh.positions.clone();
    for it in 0..iterations * 2 {
        let f = if it % 2 == 0 { lambda } else { mu };
        let cur = &mesh.positions;
        next.par_iter_mut().enumerate().for_each(|(v, out)| {
            let nb = &adj[start[v] as usize..start[v + 1] as usize];
            if border[v] || nb.is_empty() {
                *out = cur[v];
                return;
            }
            let avg = nb.iter().map(|&u| cur[u as usize]).sum::<Vec3>() / nb.len() as f32;
            *out = cur[v] + (avg - cur[v]) * f;
        });
        std::mem::swap(&mut mesh.positions, &mut next);
    }
}
