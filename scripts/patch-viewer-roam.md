Update an existing exported viewer without rebuilding its models:

```sh
cp -a /path/to/viewer /path/to/roam
python3 dev/patch-viewer-roam.py /path/to/roam
python3 -m http.server 8013 --bind 127.0.0.1 --directory /path/to/roam
```

The script uses this checkout's viewer template. It preserves the existing model
parts, import URLs, camera views, scale, and collision grid, and saves the original
page beside the folder as `roam.before-roam.html` (`--backup PATH` elsewhere,
`--no-backup` for none), never inside the served folder. A backup an earlier version
left inside the folder is moved out. It can be rerun. No build or GPU is needed.

When a cleaned model package uses the **same coordinates and scale**, keep a known
collision volume and its levelled up vector with:

```sh
python3 dev/patch-viewer-roam.py /path/to/new-viewer --collision-from /path/to/roam
```

This copies the `data/collide-NN.js` parts, their configuration and the floor height
measured along that up vector. Never reuse a
volume after rescaling, rotating, or independently reconstructing the model. Use
the new model's own collision export in that case. `--template /path/index.html`
lets a copied script use a template outside this checkout.

Rooms open in walk mode. Mouse drag looks; WASD/arrows walk; scroll steps along the
view; right-drag slides; Shift accelerates; R resets; O toggles orbit (where Q/E fly
down/up); H hides help. `?mode=orbit` opens in orbit around the named view's look-at
point. Capture paths and view directions never constrain navigation, including with
legacy `?guard=1` URLs.
The near plane is 1 cm (scaled to package units) in both modes; legacy `?near=`
values no longer hide splats. Model-detail crossfades are unchanged.

Walk is a standing capsule (1.5 m eye height) that slides along walls and ceilings
and steps up at most 20 cm. Every spawn (named view, R, switching from orbit) stands
on the floor: the exported floor height where it supports the spot, else the first
support below the eye; an occupied spawn searches nearby for clear space. Movement
itself never passes through solid, and camera sweeps remain shorter than half a voxel
even after long frames. The grid's outer box is storage, not a wall: walking beyond
it is allowed. Geometry accuracy still depends on the exported collision volume.

Run the walk regression (synthetic rooms, level and tilted by 5.5 degrees):

```sh
node scripts/accept/viewer-roam.js
# Or check the exact JavaScript shipped in a patched package:
node scripts/accept/viewer-roam.js /path/to/roam/index.html
```

A named page that does not exist is an error, never a silent template check. It
checks the capsule against furniture below eye level, swept walls without tunnelling,
height-preserving wall sliding (also on a tilted reconstruction), free space beyond
the stored grid, a 15 cm step and a 30 cm obstacle, escape from an occupied spawn,
floor placement from high and low cameras without floor metadata, the startup orbit
pivot, mode switches, and the floor basis for every cardinal up axis.
