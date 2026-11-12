# CubeCL channel wake patch

Upstream: `tracel-ai/cubecl` commit
`467522442c7b68f02c7ff0c84b2e249e36d8da91`, crate `cubecl-common`
version `0.11.0-pre.4`. Original MIT/Apache licenses and NOTICE are included.

The only Rust source change is in `src/device/handle/channel.rs`:

- Save the server thread handle in `OnceLock<Thread>`.
- Unpark it when a fully initialized batch is published or shutdown requested.
- Replace the server's idle sleep with `park_timeout` at the same timeout.

The batch size, atomics, spin/yield budgets, client wait, and timeout remain
unchanged. An unpark token persists across the queue-check/park race. The
timeout still lets the server detect the last client being dropped. This uses
portable standard-library threads inside the existing native channel module.

`Cargo.toml` expands upstream workspace inheritance without changing dependency
versions or features. Its `cubecl-environment` dependency stays on the pinned
upstream Git revision, so the rest of CubeCL shares that same crate identity.
The root Git-source patch replaces only `cubecl-common`; no external fork or
local-path dependency outside this repository is required.

Measured source patch: `8182044b9edeefc8bd96dc6b6f6160b36985d8c0` in the
O3 experiment checkout. Full-schedule-warmed, same-runner L40S Truck-clean
7k pairs on engine `1ef021d`, three seeds and rotated order:

| Seed | Main wall (s) | Wake wall (s) |
|---|---:|---:|
| 1 | 36.62 | 33.63 |
| 2 | 36.82 | 35.02 |
| 3 | 36.42 | 34.42 |

Mean wall falls from 36.620 to 34.357 seconds (6.18%). O1 reported neutral
quality: mean PSNR delta +0.04 dB, LPIPS delta at most 0.003, and about
279.7k final splats in all arms. Runner setup and full warmup are excluded
from these walls. These measurements precede the packaging-only vendoring.

When updating CubeCL, re-evaluate this patch against upstream and remove the
vendor override if upstream provides the same wake behavior.

## Full-queue client wait (H1)

`DeviceClient::enqueue` used to retry `available_index.fetch_add` in a spin
loop of 524,288 iterations while the queue was full, so the Burn fusion and
trainer threads burned a core each on a locked add to the cache line the
server writes. A client now spins on a plain load for the server's spin
budget, then registers in `State::waiters` and parks; `Server::fetch` unparks
all waiters right after it resets the index. Both sides order the index and
waiter flag with `SeqCst`, so a wake is never lost; a 1 ms park timeout is a
safety net only.
