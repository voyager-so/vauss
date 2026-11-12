# Agents and MCP

Every pipeline step in Vauss is a typed tool: `(run directory, JSON request) → JSON result + images`. The
same tools are reachable three ways, and they are the same code path:

| Way in | Use it for |
|---|---|
| `vauss tool NAME --run-dir R --input '{…}'` | Scripts and debugging: one step at a time from a shell |
| `vauss serve --run-dir R` | Your own agent: an MCP server over stdio for one run |
| `vauss run … --driver claude` / `--driver codex` | Let Claude Code or Codex run the whole job with those tools |

`vauss tools` lists the tools with a one-line description each; `vauss --json tools` prints their JSON
input schemas.

## The tools

| Stage | Tools |
|---|---|
| Whole pipeline in one call | `run_default_pipeline` (the fixed driver's pipeline: frames, cameras, split, preview, training, export, evaluation) |
| Input | `inspect_input`, `extract_frames`, `mask_distractors` (mask people, cars… by name) |
| Cameras | `reconstruct_sfm`, `define_split` (freeze the held-out frames once, before any training) |
| Model | `train_splats`, `evaluate`, `render_views` |
| Geometry | `extract_mesh` |
| Edits | `select_objects` (by name), `remove_objects`, `inpaint_removed` |
| Shots | `plan_shot`, `render_shot`, `generate_shot` (hosted video models, costs money), `export_blender` |
| Control | `job_status`, `stop_job`, `record_decision`, `finish` |

Selecting objects by name (`select_objects`, `mask_distractors`, and "remove the <object>" goals) uses SAM 2.1
and Grounding DINO through the optional [Spirula](https://github.com/harry7557558/spirula-studio) helper. Spirula
is GPL-3.0 and runs as a separate program; Vauss never links or bundles it. It is not downloaded automatically:
opt in once with `vauss fetch spirula --allow-gpl`, or set `SPLAT_ALLOW_GPL=1`. On Debian/Ubuntu it needs the
OpenGL loader: `sudo apt install libopengl0` (plus `libglx0 libegl1` on minimal systems).

Long tools (camera solve, training, evaluation, meshing, selection) return a `job_id` at once; call
`job_status` with `wait_s` to block until the job ends and get its result. Tools that produce pictures (held-out
comparisons, contact sheets, mesh previews, shot frames) return them as MCP image content, so a multimodal model
actually looks at them. Every call and decision is appended to the run's `journal.jsonl`.

## Use Vauss from Claude Code, Codex or any MCP client

```sh
vauss init room.mp4 --goal "walk-through of my living room" --run-dir runs/room
claude mcp add vauss -- vauss serve --run-dir "$PWD/runs/room"
```

Then ask your agent for what you want ("reconstruct this room, then remove the coffee table and render a slow
orbit"). One server serves one run directory; start another server for another run. The server logs to
stderr; stdout is the protocol channel.

For Codex, add the server to `~/.codex/config.toml`:

```toml
[mcp_servers.vauss]
command = "vauss"
args = ["serve", "--run-dir", "/absolute/path/to/runs/room"]
tool_timeout_sec = 3600
```

## Let Vauss start the agent

```sh
vauss run room.mp4 --goal "walk-through of my living room; remove the coffee table" --driver claude
vauss run room.mp4 --goal "..." --driver codex --model <model> --effort high
```

Vauss starts the `claude` or `codex` CLI headless (it must be installed and logged in), connects it to its own
MCP server for this run, and gives it a system prompt about the tools and the rules: freeze the held-out split
once, look at the comparison images, record decisions, and end with an honest `finish` verdict. The agent has no
shell or file access, only Vauss’s tools. Its event stream is saved in `RUN/agent/`, and a runaway guard ends
it after 300 turns (`SPLAT_AGENT_MAX_TURNS`).

`--driver fixed` runs the same pipeline with no agent.

## Worked example

[examples/agent-mcp](../examples/agent-mcp/README.md) drives `vauss serve` from a ~150-line Python MCP client:
the full pipeline, a held-out review, a rendered camera move and an honest `finish`, with an optional object
removal. It saves every request, result and image the agent would see.
