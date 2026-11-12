# Free-walk packages

Requires an existing `splat` CLI, Python 3 + NumPy, Node + `playwright-core`, Chrome, and ffmpeg. Exporting downloads the pinned Three.js modules; viewing uses only localhost files. No footage is uploaded.

```bash
SPLAT_BIN=/path/to/splat dev/free-roam/package.sh room/splat.ply room/viewer --data room/colmap
dev/free-roam/serve.sh room/viewer 8024
node dev/free-roam/capture.mjs http://127.0.0.1:8024/ room/walk.mp4
```

`package.sh PLY OUT` also accepts a bare PLY. This preserves the PLY coordinates, reads its vertical-axis comment as the repository's PLY importer does (`x`, `y`, `z` are +X, -Y, -Z; three numbers are an up vector; it must be finite and nonzero), and assumes model units are metres. Use `--data` for solved scale/orientation. The bare fallback supports binary little-endian PLY with scalar vertex properties.

`--web EXISTING_WEB` reuses a web export of **that model** and its source-to-viewer transform; collision is still rebuilt from the supplied PLY. It is useful when a 4M-splat model is already exported. `--collider OTHER_VIEWER` explicitly reuses another package's collider and up axis; it must have exactly the same transform and scale. Equal metres-per-unit alone does not prove this. Output must be a new directory.

Collision uses alpha > 0.8 splat centres, rejects splats with sigma above 0.5 m, closes one-cell sampling gaps supported on opposite sides, and removes isolated components of at most eight cells. It does not carve a corridor around the recorded path. The grid coarsens above 40 million cells; its extents never constrain movement. This is approximate reconstructed geometry: holes in walls or missing/translucent furniture can still be missed. Newly cleaned geometry is preferable to an old collider.

Walk uses a standing capsule, 1.5 m eye height, level movement, swept wall sliding and steps up to 20 cm. Every spawn (named view, R, switching from orbit) stands on the floor: the exported floor height where it supports that spot, else the first support below the eye; an occupied spawn then searches nearby for clear space. Any up vector (including Z-up and X-up sources) gets an orthonormal floor basis. Mouse pitch/yaw are unrestricted. WASD/arrows and scroll move; right-drag slides on the floor. Q/E work in orbit/fly mode. O changes modes and R resets. `?body=0` selects the cheaper legacy head-sphere collision for comparisons; `?collide=0` explicitly disables collision. Neither is used in the acceptance recordings.

The default capture searches reachable supported floor, prioritises distance from capture segments, and visits the middle and four extrema of that reachable region. Planned edges pass through the same movement solver used during recording. The finite search window limits only the planner, never manual movement. A sparse/noisy model may leave a smaller reachable region than the real room.

Supply a JSON array of `[x,y,z]` **floor positions in the viewer frame** as the third argument to capture. The recorder starts at the viewer's spawn and walks toward the supplied points; include the current floor position first. It never teleports along the route. Outputs include a real-time MP4, WebM, final screenshot, requested path, actual positions, GPU identity and timing metrics. Check actual route coverage in the JSON; a supplied route can be blocked by geometry.

`SECONDS=60`, `WIDTH=1280`, `HEIGHT=720` are defaults. `NO_VIDEO=1 SECONDS=20` measures navigation without encoding. `movingFps` excludes idle frames; `fps` is browser animation cadence over the entire sample; `drawCallsPerSecond` counts render calls and can include offscreen passes. The JSON reports model splat count and adaptive rendering remains enabled. The fixed-size capture canvas preserves the output resolution when the renderer changes its own pixel budget. If the renderer becomes idle, the recorder holds the last actual frame through the remaining real-time interval; it never speeds up the recording.

Set `PLAYWRIGHT_MODULE=/absolute/path/to/playwright-core/index.mjs` to reuse an installation; `CHROME=/path/to/chrome` overrides the executable. The tool rejects software GPU renderers. Linux uses Vulkan ANGLE; macOS uses Metal ANGLE. For a private Mac check, copy the script and use `ssh -R 18024:127.0.0.1:8024 USER@MAC_HOST` while running it there against localhost:18024. Nothing is publicly served.

`serve.sh PACKAGE PORT [UNIT]` starts a transient user systemd unit bound to 127.0.0.1. Stop it with `systemctl --user stop free-roam-PORT`; restart after reboot by running the script again. Existing units and occupied ports are not replaced.

The served folder never carries local paths: `package-source.json` records the model's SHA-256 and size, and absolute paths are scrubbed from `build.json` and the page configuration (packaging fails if one remains). `--provenance PRIVATE.json` (outside OUT) keeps the full local paths for your own records. Fresh packages keep no backup page.

Acceptance checks against the actual template, or against the exact JavaScript of an exported / patched page:

```bash
THREE_MODULE=/path/to/viewer/vendor/three.module.js node dev/free-roam/check-navigation.mjs [/path/to/viewer/index.html]
THREE_MODULE=/path/to/viewer/vendor/three.module.js python3 -m unittest discover -s dev/free-roam -p 'test_*.py'
```

The unit tests cover inline-script escaping of generated pages, served-path scrubbing, PLY vertical-axis parsing and the navigation check (skipped without Three.js).
