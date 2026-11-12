#!/usr/bin/env python3
"""Public package manifest: no local paths in anything the viewer folder serves.

provenance.py PACKAGE PLY [--web WEB] [--private FILE]
Writes PACKAGE/package-source.json with the model's content hash only, scrubs absolute
paths from build.json and the page's CFG, and fails if any served text still has one.
--private FILE (outside PACKAGE) keeps the full local provenance for your own records.
"""
import argparse,hashlib,json,re
from pathlib import Path

# An absolute POSIX or Windows path of at least two components; URLs (scheme://), relative
# paths and markup such as </script> in a title are kept.
ABSOLUTE=re.compile(r'(?<![\w.:/\\<])(?:[A-Za-z]:[\\/]|~?/(?=[\w.~-]))[^\s"\'(),;<>\\/]*[\\/][^\s"\'(),;<>]*')
REMOVED='(local path)'

def scrub(value):
    if isinstance(value,str): return ABSOLUTE.sub(REMOVED,value)
    if isinstance(value,list): return [scrub(v) for v in value]
    if isinstance(value,dict): return {k:scrub(v) for k,v in value.items()}
    return value

def script_json(value):
    return json.dumps(value,separators=(',',':')).replace('<','\\u003c')

def served_metadata(package):
    """(name, text) of the served metadata: every JSON file and each page's inline JSON (CFG, import map)."""
    for p in sorted(package.rglob('*')):
        if not p.is_file() or {'data','vendor'} & set(p.relative_to(package).parts[:-1]): continue
        if p.suffix=='.json': yield p.relative_to(package),p.read_text(errors='replace')
        if p.suffix=='.html':
            text=p.read_text(errors='replace')
            for m in re.finditer(r'^const CFG = (.*);$',text,re.M): yield p.relative_to(package),m[1]
            for m in re.finditer(r'<script type="importmap">(.*?)</script>',text,re.S): yield p.relative_to(package),m[1]

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('package',type=Path); ap.add_argument('ply',type=Path)
    ap.add_argument('--web',type=Path); ap.add_argument('--private',type=Path)
    a=ap.parse_args()
    package=a.package.resolve(); ply=a.ply.resolve()
    if a.private and package in a.private.resolve().parents: ap.error('--private must be outside the served package')
    digest=sha256(ply)
    (package/'package-source.json').write_text(json.dumps({'ply_sha256':digest,'ply_bytes':ply.stat().st_size,
        'coordinate_note':'--web/--collider must describe this exact model coordinate frame'},indent=2)+'\n')
    if a.private:
        a.private.parent.mkdir(parents=True,exist_ok=True)
        a.private.write_text(json.dumps({'ply':str(ply),'web':str(a.web.resolve()) if a.web else None,'ply_sha256':digest,'package':str(package)},indent=2)+'\n')
    build=package/'build.json'
    if build.exists(): build.write_text(json.dumps(scrub(json.loads(build.read_text())),indent=2)+'\n')
    page=package/'index.html'; text=page.read_text(); m=re.search(r'^const CFG = (.*);$',text,re.M)
    if not m: raise SystemExit(f'{page}: missing viewer CFG')
    page.write_text(text[:m.start(1)]+script_json(scrub(json.loads(m[1])))+text[m.end(1):])
    leaks=[(name,hit) for name,text in served_metadata(package) for hit in ABSOLUTE.findall(text)]
    if leaks: raise SystemExit('absolute paths left in the served package: '+'; '.join(f'{p}: {h}' for p,h in leaks[:5]))

if __name__=='__main__':
    main()
