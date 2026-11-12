#!/usr/bin/env python3
"""Update an exported viewer without rebuilding models (stdlib only).

python3 dev/patch-viewer-roam.py /path/to/copied/viewer
Optional --collision-from /path/to/previous/viewer keeps a known room volume
when replacing the model. Both packages MUST use the same coordinates and scale.
The original page is saved beside (never inside) the served package folder.
"""
import argparse
import html
import json
from pathlib import Path
import re
import shutil


def script_json(value):
    """JSON for an inline <script>: no literal '<', so metadata cannot close or alter the element."""
    return json.dumps(value, separators=(',', ':')).replace('<', '\\u003c')


def config(package):
    text = (package / 'index.html').read_text()
    match = re.search(r'^const CFG = (\{.*\});$', text, re.M)
    if not match:
        raise ValueError(f'{package}: missing viewer CFG')
    return text, json.loads(match[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    parser.add_argument('--collision-from', type=Path,
                        help='copy collision data/up from a viewer in the SAME coordinate frame')
    parser.add_argument('--template', type=Path, default=Path(__file__).resolve().parents[1] /
                        'crates/splat-webexport/src/viewer/index.html')
    parser.add_argument('--backup', type=Path,
                        help='where to save the original page (default: PACKAGE.before-roam.html beside the folder)')
    parser.add_argument('--no-backup', action='store_true', help='fresh packages: keep no copy of the original page')
    args = parser.parse_args()
    package = args.package.resolve()
    old, cfg = config(package)
    imports = json.loads(re.search(r'<script type="importmap">\s*(.*?)\s*</script>', old, re.S)[1])['imports']
    if args.collision_from:
        source = args.collision_from.resolve()
        _, other = config(source)
        if cfg.get('metersPerUnit', 1) != other.get('metersPerUnit', 1):
            parser.error('collision source scale differs from target')
        if not other.get('collide'):
            parser.error('collision source has no collision volume')
        cfg['collide'] = other['collide']
        cfg['up'] = other['up']
        # The floor is measured along that up vector: it belongs to the same frame.
        for key in ('floorHeight', 'floorExtent'):
            if key in other:
                cfg[key] = other[key]
            else:
                cfg.pop(key, None)
        c = cfg['collide']
        names = [f"data/{c['key']}-{i:02}.js" for i in range(c['parts'])]
        for name in names:
            if not (source / name).is_file():
                parser.error(f'missing {source / name}')
        if source != package:
            (package / 'data').mkdir(exist_ok=True)
            for name in names:
                shutil.copyfile(source / name, package / name)
    template = args.template.read_text()
    result = (template.replace('__TITLE__', html.escape(cfg['title']))
              .replace('__CFG__', script_json(cfg))
              .replace('__THREE_ADDONS__', imports.get('three/addons/', 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/'))
              .replace('__THREE__', imports['three']))
    result = re.sub(r'(<script type="importmap">).*?(</script>)', lambda m: m[1] + script_json({'imports': imports}) + m[2], result, flags=re.S)
    # Backups stay outside the served folder: an old page can carry unescaped metadata (an executable endpoint).
    backup = None if args.no_backup else (args.backup or package.parent / f'{package.name}.before-roam.html').resolve()
    if backup is not None and package in backup.parents:
        parser.error(f'--backup must be outside the served package: {backup}')
    legacy = package / 'index.before-roam.html'  # where earlier versions of this script saved it
    if legacy.exists():
        moved = backup or package.parent / f'{package.name}.before-roam.html'
        for n in range(1, 1000):
            if not moved.exists():
                break
            moved = moved.with_name(f'{package.name}.before-roam.{n}.html')
        shutil.move(legacy, moved)
        print(f'Moved the served legacy backup out of the package: {moved}')
    if backup is not None and not backup.exists():
        backup.write_text(old)
    temporary = package / 'index.roam.tmp'
    temporary.write_text(result)
    temporary.replace(package / 'index.html')
    build_path = package / 'build.json'
    if build_path.exists():
        build = json.loads(build_path.read_text())
        build['navigation'] = 'roam'
        build['collide'] = cfg.get('collide')
        build_path.write_text(json.dumps(build, indent=2) + '\n')
    print(f'Updated {package}/index.html' + (f'; backup: {backup}' if backup else ''))


if __name__ == '__main__':
    main()
