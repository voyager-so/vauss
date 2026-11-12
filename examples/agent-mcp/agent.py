#!/usr/bin/env python3
"""A scripted stand-in for an AI agent: drives `splat serve` (MCP over stdio) through a whole run.

It makes the same tool calls Claude Code or Codex make through splat's MCP server, in a fixed order, and
saves every request, result and returned image so you can see exactly what an agent sees.

    python3 agent.py RUN_DIR OUT_DIR [--shot "slow orbit, 6 seconds"] [--remove "dining chairs"] [--splat splat]

RUN_DIR must already exist (`splat init VIDEO --goal ... --run-dir RUN_DIR`). Python 3.8+, standard library only.
"""
import argparse, base64, json, pathlib, subprocess, sys, time


class Mcp:
    def __init__(self, argv, log):
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.log, self.next_id = log, 1

    def send(self, method, params=None, notify=False):
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notify:
            msg["id"] = self.next_id
            self.next_id += 1
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()
        if notify:
            return None
        while True:
            line = self.p.stdout.readline()
            if not line:
                sys.exit(f"server exited (code {self.p.wait()})")
            reply = json.loads(line)
            if reply.get("id") == msg["id"]:
                if "error" in reply:
                    sys.exit(f"{method}: {reply['error']}")
                return reply["result"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--remove", default="", help="objects to select and delete after training, e.g. 'dining chairs'")
    ap.add_argument("--shot", default="slow orbit around the subject, 6 seconds", help="camera move to plan and render")
    ap.add_argument("--splat", default="splat")
    a = ap.parse_args()
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "transcript.jsonl", "w")
    t_start = time.time()
    srv = Mcp([a.splat, "serve", "--run-dir", a.run_dir], log)
    srv.send("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                            "clientInfo": {"name": "splat-example-agent", "version": "1"}})
    srv.send("notifications/initialized", notify=True)
    tools = [t["name"] for t in srv.send("tools/list")["tools"]]
    print(f"{len(tools)} tools: {', '.join(tools)}", flush=True)
    n_img = 0

    def call(name, **args):
        nonlocal n_img
        t0 = time.time()
        res = srv.send("tools/call", {"name": name, "arguments": args})
        dt = time.time() - t0
        texts = [c["text"] for c in res.get("content", []) if c.get("type") == "text"]
        for c in res.get("content", []):
            if c.get("type") == "image":
                n_img += 1
                ext = "png" if c.get("mimeType") == "image/png" else "jpg"
                (out / f"{n_img:02d}-{name}.{ext}").write_bytes(base64.b64decode(c["data"]))
        body = json.loads(texts[0]) if texts and texts[0].lstrip().startswith("{") else {"text": texts[:1]}
        log.write(json.dumps({"tool": name, "args": args, "seconds": round(dt, 1), "result": body}) + "\n")
        log.flush()
        status = body.get("status") or body.get("state") or ("error" if res.get("isError") else "ok")
        print(f"[{time.time() - t_start:7.1f}s] {name}({json.dumps(args)[:80]}) -> {status} in {dt:.0f}s", flush=True)
        if res.get("isError") or status in ("failed", "error"):
            print(json.dumps(body)[:2000])
        return body

    def wait(body):
        # Long tools return a job_id; job_status(wait_s) blocks until the job ends and returns its result.
        job = body.get("job_id")
        while job:
            body = call("job_status", job_id=job, wait_s=600)
            if body.get("status") not in ("queued", "running", "pending"):
                return body.get("result", body)
        return body

    def find(body, key):
        # First value of `key` anywhere in a result (ids sit at different depths in different tools).
        if isinstance(body, dict):
            if body.get(key):
                return body[key]
            body = list(body.values())
        if isinstance(body, list):
            for v in body:
                hit = find(v, key)
                if hit:
                    return hit
        return None

    def artifact(body, kind):
        # Id of the first artifact of this kind (shot, selection, candidate…) a result registered.
        return next((x["id"] for x in body.get("artifacts", []) if x.get("kind") == kind), None)

    # 1. The whole default pipeline in one call: frames, camera solve, split, training, export, evaluation.
    run = wait(call("run_default_pipeline"))
    cand = find(run, "best_candidate_id") or find(run, "candidate_id")
    print(f"candidate: {cand}", flush=True)
    # 2. Look at held-out views (the agent sees these images).
    call("render_views", candidate_id=cand, max_views=4)
    # 3. Optional edit: select objects by name, delete them, look again.
    if a.remove:
        sel = wait(call("select_objects", candidate_id=cand, prompt=a.remove))
        sel_id = artifact(sel, "selection") or find(sel, "selection_id")
        if sel_id:
            edit = wait(call("remove_objects", candidate_id=cand, selection_id=sel_id))
            cand = artifact(edit, "candidate") or cand
            call("render_views", candidate_id=cand, max_views=4)
    # 4. A designed camera move through the result.
    shot = call("plan_shot", candidate_id=cand, prompt=a.shot)
    shot_id = artifact(shot, "shot")
    if shot_id:
        wait(call("render_shot", shot_id=shot_id))
    # 5. End the run with a verdict, as the agent must.
    why = "Scripted example agent: default pipeline, held-out review, optional object removal, one shot."
    done = call("finish", status="succeeded", candidate_id=cand, explanation=why)
    if done.get("status") == "failed":
        # finish checks the claim against the run's gates (e.g. too few frames placed): report it as partial.
        defect = (done.get("error") or {}).get("message", "")
        call("finish", status="partial", candidate_id=cand, explanation=why, remaining_defects=[defect])
    srv.p.stdin.close()
    srv.p.wait(timeout=60)
    print(f"done in {time.time() - t_start:.0f}s; transcript: {out / 'transcript.jsonl'}")


if __name__ == "__main__":
    main()
