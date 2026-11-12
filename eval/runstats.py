#!/usr/bin/env python3
"""Run a command and record wall time, peak GPU memory and peak host RSS.

Usage:
    python eval/runstats.py --json stats.json [--interval 0.5] -- spirula train ...

GPU memory is sampled with nvidia-smi every --interval seconds, two ways:
  peak_vram_proc_mib    summed used_memory of the process tree (works for CUDA processes; Vulkan
                        processes often do not appear in --query-compute-apps)
  peak_vram_device_mib  device-wide used memory minus the value just before launch (works for any
                        API, but includes other jobs on the GPU; only trust it on an idle GPU)
Host RSS is the peak summed RSS of the process tree from /proc.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def smi(args):
    try:
        return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return ""


def device_used():
    out = smi(["--query-gpu=memory.used"]).split()
    return int(out[0]) if out else 0


def tree(pid):
    pids, frontier = {pid}, [pid]
    while frontier:
        p = frontier.pop()
        try:
            kids = Path(f"/proc/{p}/task/{p}/children").read_text().split()
        except OSError:
            continue
        for k in map(int, kids):
            if k not in pids:
                pids.add(k); frontier.append(k)
    return pids


def rss_kib(pids):
    total = 0
    for p in pids:
        try:
            for line in Path(f"/proc/{p}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1])
        except OSError:
            pass
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command given")

    base = device_used()
    peak = {"proc": 0, "device": 0, "rss": 0}
    t0 = time.time()
    proc = subprocess.Popen(cmd)
    done = threading.Event()

    def sample():
        while not done.is_set():
            pids = tree(proc.pid)
            used = 0
            for line in smi(["--query-compute-apps=pid,used_memory"]).splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) in pids and parts[1].isdigit():
                    used += int(parts[1])
            peak["proc"] = max(peak["proc"], used)
            peak["device"] = max(peak["device"], device_used() - base)
            peak["rss"] = max(peak["rss"], rss_kib(pids))
            done.wait(a.interval)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    rc = proc.wait()
    done.set(); th.join()
    stats = {"cmd": cmd, "returncode": rc, "wall_s": round(time.time() - t0, 2),
             "peak_vram_proc_mib": peak["proc"], "peak_vram_device_mib": peak["device"],
             "baseline_device_mib": base, "peak_rss_mib": round(peak["rss"] / 1024, 1)}
    print(json.dumps(stats), file=sys.stderr)
    if a.json:
        a.json.write_text(json.dumps(stats, indent=2))
    return rc


if __name__ == "__main__":
    sys.exit(main())
