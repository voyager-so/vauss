# Examples

Each example runs splat end to end on a public, commercially usable clip ([DATA.md](DATA.md)). Every command shown
was run as written, and the times, scores and stills come from those runs.

| Example | From | To | Time |
|---|---|---|---|
| [room-walk](room-walk/README.md) | 50 s iPhone video of an open-plan living room | A splat you walk through in the browser | 27 min |
| [car-mesh](car-mesh/README.md) | 41 s walk around a parked car | A textured mesh (GLB) and a splat | 12 min |
| [camera-moves](camera-moves/README.md) | The car splat | Orbit, push-in, crane and a stabilized re-render, as video | 3 min |
| [room-fill](room-fill/README.md) | The room splat | `vauss fill`: novel views repaired by NVIDIA Fixer and trained back in | 6 min |
| [agent-mcp](agent-mcp/README.md) | The car clip | The pipeline driven by an agent over MCP (`vauss serve`) | 9 min |

All runs used splat built from `main` at `bea9145d` (2026-10-24) on one cloud Linux machine: an NVIDIA RTX PRO 6000
Blackwell (96 GB) and 6 CPU cores. On a smaller GPU, training takes longer. The camera solve runs on the CPU and is
the longest stage in every example, so more CPU cores shorten it and a faster GPU does not.

The clips were downloaded on that machine from their original sources with the commands in each example.
