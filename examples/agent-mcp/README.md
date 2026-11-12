# An agent drives splat over MCP

`vauss serve --run-dir RUN` is an MCP server for one run. Any MCP client can drive the whole pipeline through it:
Claude Code, Codex, or your own code. This example uses [`agent.py`](agent.py), a ~150-line Python client
(standard library only). It plays the agent's part in a fixed order and saves every request, result and image a
real agent would see. That makes the run reproducible and needs no LLM account.

![Held-out frames and renders the agent received from render_views](stills/render-views.jpg)

*An image the agent received: `render_views` returns held-out frames next to the splat's renders.*

## Clip

"Moskvitch-2142 walk around" by [trolleway](https://www.youtube.com/user/trolleway),
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Moskvitch-2142_walk_around.webm),
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0).

## Commands

```sh
curl -fsSL -o moskvitch.webm \
  "https://upload.wikimedia.org/wikipedia/commons/c/c4/Moskvitch-2142_walk_around.webm"
vauss init moskvitch.webm --goal "3D scene of the car, then a slow orbit shot around it" --run-dir runs/agent-car
python3 examples/agent-mcp/agent.py runs/agent-car out/agent-car --shot "slow orbit around the car, 6 seconds"
```

`agent.py` starts `vauss serve`, lists the tools, then calls:

1. `run_default_pipeline`: frames, camera solve, held-out split, training, export, evaluation, in one call;
2. `render_views`: looks at held-out frames next to their renders;
3. `plan_shot` and `render_shot`: a camera move through the result;
4. `finish`: ends the run with a verdict. `finish` checks the claim against the run's gates. If, say, too few
   frames were placed, it refuses "succeeded" and the client reports "partial" with the reason, as a real agent
   must.

`--remove "<object>"` adds `select_objects` and `remove_objects` before the shot. Selection needs the optional
Spirula segmenter (GPL-3.0, a separate program; opt in with `vauss fetch spirula --allow-gpl` or
`SPLAT_ALLOW_GPL=1`, and on Debian/Ubuntu `sudo apt install libopengl0`). It was not part of this run.

To let a real agent do this instead:

```sh
claude mcp add vauss -- vauss serve --run-dir "$PWD/runs/agent-car"   # then ask Claude Code for what you want
vauss run moskvitch.webm --goal "..." --driver claude                  # or let splat start Claude Code itself
```

More in [docs/agents.md](../../docs/agents.md).

## Result

`out/agent-car/stdout.txt` from this run:

```
21 tools: run_default_pipeline, inspect_input, extract_frames, mask_distractors, reconstruct_sfm, define_split, train_splats, evaluate, render_views, extract_mesh, plan_shot, render_shot, generate_shot, export_blender, select_objects, remove_objects, inpaint_removed, job_status, stop_job, record_decision, finish
[  538.8s] run_default_pipeline({}) -> succeeded in 539s
candidate: candidate_2eba4991620c
[  541.1s] render_views({"candidate_id": "candidate_2eba4991620c", "max_views": 4}) -> succeeded in 2s
[  541.2s] plan_shot({"candidate_id": "candidate_2eba4991620c", "prompt": "slow orbit around the car,) -> succeeded in 0s
[  555.2s] render_shot({"shot_id": "shot_ecd1f48bad28"}) -> succeeded in 14s
[  555.2s] finish({"status": "succeeded", "candidate_id": "candidate_2eba4991620c", "explanation":) -> succeeded in 0s
done in 555s; transcript: out/agent-car/transcript.jsonl
```

`out/agent-car/transcript.jsonl` has one line per call (tool, arguments, seconds, the full JSON result). The
images the server returned are saved next to it: eight held-out comparisons from the pipeline, four from
`render_views`, and the first, middle and last frames of the shot.

![The middle frame of the rendered orbit, as returned by render_shot](stills/render-shot.jpg)

## Time and hardware

Measured on 2026-10-24 with splat built from `main` at `bea9145d` on an NVIDIA RTX PRO 6000 Blackwell (96 GB),
6 CPU cores. 555 s in total, 539 s of it in `run_default_pipeline`. Peak GPU memory, sampled every 2 s:
0.8 GiB.

## Notes

- `run_default_pipeline` blocks until done. The step-by-step tools (`reconstruct_sfm`, `train_splats`,
  `evaluate`…) return a `job_id` at once; poll them with `job_status` and `wait_s`, as `agent.py` does for
  `select_objects`.
- The server only gives access to splat's tools, never a shell or the file system. Every call lands in the run's
  `journal.jsonl`.
