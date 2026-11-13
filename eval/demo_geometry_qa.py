#!/usr/bin/env python3
"""CPU evidence extraction for localized native occlusions; no renderer/training invocation.

Produces exact-name native render jobs and conservative foreground Gaussian candidates for a
controlled ablation. Projected bounding spheres are a diagnostic proxy, NOT native alpha or a
floater classifier. Only an original/ablated native comparison can establish their contribution.
An optional pose_refine.json scores the same candidates under refined poses; it must belong to
this exact candidate, not an earlier reconstruction. All original files are read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sfm_health import read_bin


def rotation(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                     [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                     [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])


def ply_data(path):
    header, props, count = [], [], None
    with path.open('rb') as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError('truncated PLY header')
            header.append(line)
            words = line.decode('ascii').split()
            if words[:2] == ['format', 'binary_little_endian']:
                pass
            elif words[:1] == ['format']:
                raise ValueError('requires binary little-endian PLY')
            if words[:2] == ['element', 'vertex']:
                count = int(words[2])
            elif words[:1] == ['element']:
                raise ValueError('only vertex PLYs supported')
            if words[:1] == ['property']:
                if words[1] != 'float':
                    raise ValueError('requires float vertex properties')
                props.append((words[2], '<f4'))
            if words == ['end_header']:
                break
    header = b''.join(header)
    data = np.memmap(path, dtype=props, mode='r', offset=len(header), shape=(count,))
    return header, data


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def native_job(ply, dataset, out, names, source_sizes):
    # LoadImage treats zero as a zero cap, not "unlimited". The largest source edge keeps
    # every requested view at its calibrated size, including mixed landscape/portrait inputs.
    if not names or any(min(source_sizes[name]) <= 0 for name in names):
        raise ValueError('native QA requires named views with positive source dimensions')
    return {'ply': str(ply.resolve()), 'dataset': str(dataset.resolve()),
            'max_resolution': max(max(source_sizes[name]) for name in names),
            'names': names, 'orbit': None, 'out': str(out.resolve()), 'depth': True, 'gpu_lock': None}


def analyze(cam, im, xyz, radius, opacity, sparse_xyz, sparse_ids):
    if cam['model'] != 'PINHOLE':
        raise ValueError('This diagnostic requires PINHOLE; native render jobs remain authoritative')
    fx, fy, cx, cy = cam['params']
    r, t = rotation(im['q']), np.array(im['t'])
    pc = xyz @ r.T + t
    z = pc[:, 2]
    observed = sparse_xyz[np.isin(sparse_ids, im['obs_id'])] @ r.T + t
    positive = observed[:, 2][observed[:, 2] > 0]
    if not len(positive):
        raise ValueError('no positive sparse depths')
    depth = float(np.median(positive))
    # Conservative screen footprint (ignores Gaussian orientation, mip filter and compositing).
    # Behind-camera centres are excluded below; do not divide them by a tiny sentinel and
    # manufacture infinities while computing their unused screen bounds.
    safe_z = np.where(z > 0, z, 1.0)
    with np.errstate(over='ignore', invalid='ignore'):
        u, v = fx * pc[:, 0] / safe_z + cx, fy * pc[:, 1] / safe_z + cy
        rx, ry = 3 * fx * radius / safe_z, 3 * fy * radius / safe_z
        intersects = (u + rx >= 0) & (u - rx < cam['width']) & (v + ry >= 0) & (v - ry < cam['height'])
    near = (z > 0) & (z < depth * .15) & intersects & (opacity > .3)
    large = near & (np.maximum(rx / cam['width'], ry / cam['height']) > .1)
    indices = np.flatnonzero(large)
    order = indices[np.argsort(z[indices])[:20]]
    return {'source_name': im['name'], 'center': (-r.T @ t).tolist(), 'K': [fx, fy, cx, cy],
            'size': [cam['width'], cam['height']], 'median_observed_depth': depth,
            'near_foreground_candidates': int(near.sum()), 'large_near_candidates': len(indices),
            'nearest_candidates': [{'index': int(i), 'depth': float(z[i]), 'opacity': float(opacity[i]),
                                     'max_sigma_world': float(radius[i]), 'radius_widths_3sigma': float(rx[i]/cam['width'])}
                                    for i in order]}, indices


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, type=Path, help='SfM artifact with sparse/0 and images/')
    p.add_argument('--ply', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path, help='new output directory')
    p.add_argument('--names', nargs='+', default=[f'frame_{i:06}.jpg' for i in [100, 101, 103, 104, 130, 149, 169]])
    p.add_argument('--pose-report', type=Path)
    p.add_argument('--ablate', action='store_true', help='write a separate PLY with candidate opacity suppressed, never a product cleanup')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sparse = args.dataset / 'sparse/0'
    cams, images, _, _, sparse_xyz, sparse_ids = read_bin(sparse)
    by_name = {im['name']: im for im in images.values()}
    header, data = ply_data(args.ply)
    xyz = np.stack([data[k] for k in ['x', 'y', 'z']], axis=1).astype(float)
    radius = np.exp(np.stack([data[f'scale_{i}'] for i in range(3)], axis=1).max(axis=1).astype(float))
    opacity = 1/(1+np.exp(-np.clip(data['opacity'].astype(float), -80, 80)))
    rows, selected = [], []
    refined = json.loads(args.pose_report.read_text()) if args.pose_report else None
    for name in args.names:
        im = by_name[name]
        row, indices = analyze(cams[im['camera_id']], im, xyz, radius, opacity, sparse_xyz, sparse_ids)
        rows.append(row)
        selected.extend(indices.tolist())
        if refined:
            pose = next((v for v in refined['views'] if v['view'] == name), None)
            row['refined_pose_available'] = pose is not None
            if pose:
                changed = {**im, 'q': pose['qvec'], 't': pose['tvec']}
                row['refined_pose_proxy'], _ = analyze(cams[im['camera_id']], changed, xyz, radius, opacity, sparse_xyz, sparse_ids)
    selected = np.unique(selected)
    source_sizes = {name: [cams[by_name[name]['camera_id']]['width'], cams[by_name[name]['camera_id']]['height']]
                    for name in args.names}
    report = {'scope': 'conservative_projection_proxy_not_native_render', 'cause': 'open_pending_native_pose_and_ablation_comparison',
              'ply_sha256': sha256(args.ply), 'model_sha256': {n: sha256(sparse/n) for n in ['cameras.bin', 'images.bin', 'points3D.bin']},
              'pose_report': str(args.pose_report) if args.pose_report else None,
              'pose_report_note': 'Proxy keeps original K; check focal_scales before interpreting refined pose differences.',
              'views': rows, 'ablation_candidate_count': len(selected), 'ablation_indices': selected.tolist(),
              'expected_render_dimensions': source_sizes}
    (args.out/'diagnosis.json').write_text(json.dumps(report, indent=2)+'\n')
    def job(ply, arm):
        return native_job(ply, args.dataset, args.out/arm, args.names, source_sizes)
    (args.out/'native-original-job.json').write_text(json.dumps(job(args.ply, 'original'), indent=2)+'\n')
    if args.ablate:
        output = args.out/'diagnostic-ablation.ply'
        modified = np.array(data)
        modified['opacity'][selected] = -30.0
        with output.open('wb') as f:
            f.write(header)  # Includes original comments/mode/SH and all original properties.
            modified.tofile(f)
        (args.out/'native-ablated-job.json').write_text(json.dumps(job(output, 'ablated'), indent=2)+'\n')
    print(json.dumps({'report':str(args.out/'diagnosis.json'), 'ablation_candidates':len(selected)}))


if __name__ == '__main__':
    main()
