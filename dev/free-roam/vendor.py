#!/usr/bin/env python3
import json,re,sys,urllib.request
from pathlib import Path
p=Path(sys.argv[1]); text=(p/'index.html').read_text();m=re.search(r'(<script type="importmap">)\s*(.*?)\s*(</script>)',text,re.S); obj=json.loads(m[2]);imports=obj['imports']; dest=p/'vendor';dest.mkdir(exist_ok=True)
addons=imports.get('three/addons/', 'https://cdn.jsdelivr.net/npm/three@0.180.0/examples/jsm/')
for key in ['three','three/addons/']:
    url=imports[key]+('controls/OrbitControls.js' if key.endswith('/') else '')
    name='OrbitControls.js' if key.endswith('/') else 'three.module.js'
    data=urllib.request.urlopen(url,timeout=30).read();(dest/name).write_bytes(data)
    if key=='three': imports[key]='./vendor/'+name
    else: imports['three/addons/controls/OrbitControls.js']='./vendor/'+name;del imports[key]
# Three recent versions import their sibling core module.
core=re.findall(r"from ['\"](\./three\.core\.js)['\"]",(dest/'three.module.js').read_text())
if core: (dest/'three.core.js').write_bytes(urllib.request.urlopen(json.loads(m[2])['imports']['three'].rsplit('/',1)[0]+'/three.core.js',timeout=30).read())
(dest/'Pass.js').write_bytes(urllib.request.urlopen(addons+'postprocessing/Pass.js',timeout=30).read())
imports['three/addons/postprocessing/Pass.js']='./vendor/Pass.js'
# Inline-script safe: no literal '<' in the import map.
text=text[:m.start(2)]+json.dumps(obj).replace('<','\\u003c')+text[m.end(2):];(p/'index.html').write_text(text)
