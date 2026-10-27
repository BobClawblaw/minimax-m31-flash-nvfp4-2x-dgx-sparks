"""python3 -m spark <command>: MiniMax-M3.1-Flash on two DGX Sparks (TensorFold native CUDA, TP=2).

  download   the release at its pinned revision (on this node)
  convert    the mixed checkpoint from it (NVFP4 experts on the listed layers, 3-bit on the rest), then copy it to the worker
  build      the pinned engine and libtfgrammar, on this node; `up` copies them to the worker
  up         rank 1 on the worker, rank 0 here; waits until it serves
  down       both ranks, gracefully, then for good
  status     each rank's process and log tail, and /v1/models
  logs       rank 0's log (rank 1's with --worker)
  chat TEXT  one request
"""

import json
import os
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .config import ROOT, load

PATTERN = "[t]ensorfold-native serve"


def sh(cmd: str, check=True, capture=False):
    r = subprocess.run(cmd, shell=True, text=True, capture_output=capture)
    if check and r.returncode != 0:
        sys.exit(f"failed ({r.returncode}): {cmd}\n{r.stderr if capture else ''}")
    return r


def on_worker(s, cmd: str, check=True, capture=False):
    return sh(f"ssh {s.worker} {shlex.quote(cmd)}", check, capture)


def download(s):
    sh(f"hf download {s.source_repo} --revision {s.source_revision} --local-dir {s.source} --max-workers 8")


def convert(s):
    conv = s.worktree / "tools/zig/minimax_m3_nvfp4_mix.py"
    if not conv.exists():
        sys.exit(f"{conv} is missing: run `python3 -m spark build` first (the converter ships with the engine)")
    fp4 = " --dense-fp4" if s.dense_fp4 else ""
    sh(f"python3 -I {conv} {s.source} {s.model} --nvfp4-layers {s.nvfp4_layers} --revision {s.source_revision}{fp4}")
    sh(f"python3 -I {conv} --check {s.model}")
    on_worker(s, f"mkdir -p {s.model}")
    sh(f"rsync -a --info=progress2 {s.model}/ {s.worker}:{s.model}/")


def build(s):
    if not s.tensorfold_sha:
        sys.exit("recipe.toml pins no tensorfold_sha yet")
    src = s.path(s.tensorfold_src)
    if not src.exists():
        sh(f"git clone {s.tensorfold_repo} {src}")
    sh(f"git -C {src} fetch -q origin {s.tensorfold_branch}")
    if not s.worktree.exists():
        sh(f"git -C {src} worktree add --detach {s.worktree} {s.tensorfold_sha}")
    z = s.path(s.zig)
    cache = s.build / "zig-cache"
    common = f"-Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dtp-nvcc=/usr/local/cuda/bin/nvcc -Dsm=121 --cache-dir {cache}/local"
    env = f"ZIG_GLOBAL_CACHE_DIR={cache}/global"
    sh(f"cd {s.worktree} && {env} {z} build native {common}")
    if s.xgrammar:
        sh(f"cd {s.worktree} && {env} {z} build grammar -Dxgrammar={s.path(s.xgrammar)} -Doptimize=ReleaseFast --cache-dir {cache}/local")
    print(f"built {s.native}/bin/tensorfold-native")


def served(s) -> bool:
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{s.port}/v1/models", timeout=3)
        return True
    except urllib.error.HTTPError:
        return True  # answering, behind api_key
    except Exception:
        return False


def up(s):
    if served(s):
        print(f"already serving on :{s.port}")
        return
    run = s.run_dir
    exe = run / "bin/tensorfold-native"
    sh(f"mkdir -p {run} && rsync -a {s.native}/ {run}/")
    on_worker(s, f"mkdir -p {run}")
    sh(f"rsync -a {run}/bin {run}/lib {s.worker}:{run}/")
    # the page cache of a fresh load or copy crowds the device's allocations out; NCCL registers at init
    flush = "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null"
    sh(flush, check=False)
    on_worker(s, flush, check=False)
    envs = lambda r: " ".join(f"{k}={shlex.quote(v)}" for k, v in s.env(r).items())
    args = lambda r: " ".join(shlex.quote(a) for a in s.args(r))
    on_worker(s, f"nohup env {envs(1)} {exe} {args(1)} > {run}/rank1.log 2>&1 < /dev/null &")
    # rank 0's lines stamped with Unix seconds (the harness reads a run's finished requests from them)
    stamp = "perl -ne '$|=1; print time(), q( ), $_'"
    sh(f"nohup sh -c {shlex.quote(f'env {envs(0)} {exe} {args(0)} 2>&1 | {stamp}')} > {run}/rank0.log 2>&1 < /dev/null &")
    t0 = time.time()
    while time.time() - t0 < 900:
        if served(s):
            print(f"serving {s.served_name} on :{s.port} after {time.time() - t0:.0f} s")
            return
        # rank 0's lines carry a timestamp; a refusal or a dead rank 0 ends the wait at once
        alive = sh(f"pgrep -f '{PATTERN}'", check=False, capture=True).stdout.strip()
        r = sh(f"grep -E '^[0-9]+ (error|tensorfold:)' {run}/rank0.log | head -3", check=False, capture=True)
        if r.stdout.strip() or (not alive and time.time() - t0 > 10):
            print(r.stdout.strip() or sh(f"tail -5 {run}/rank0.log", check=False, capture=True).stdout.strip())
            on_worker(s, f"grep -E 'error' {run}/rank1.log | head -3", check=False)
            down(s)
            sys.exit(1)
        time.sleep(3)
    down(s)
    sys.exit("not serving after 15 minutes; both ranks stopped: python3 -m spark logs")


def down(s):
    sh(f"pkill -INT -f '{PATTERN}'", check=False)
    on_worker(s, f"pkill -INT -f '{PATTERN}'", check=False)
    time.sleep(8)
    sh(f"pkill -9 -f '{PATTERN}'", check=False)
    on_worker(s, f"pkill -9 -f '{PATTERN}'", check=False)
    print("stopped")


def status(s):
    probe = lambda r: f"pgrep -f '{PATTERN}' >/dev/null && echo running || echo stopped; tail -2 {s.run_dir}/rank{r}.log"
    print("rank 0:", sh(probe(0), check=False, capture=True).stdout.strip())
    print("rank 1:", on_worker(s, probe(1), check=False, capture=True).stdout.strip())
    print("serving" if served(s) else "not serving")


def logs(s, worker: bool):
    if worker:
        on_worker(s, f"tail -50 {s.run_dir}/rank1.log", check=False)
    else:
        sh(f"tail -50 {s.run_dir}/rank0.log", check=False)


def chat(s, text: str):
    body = {"model": s.served_name, "messages": [{"role": "user", "content": text}], "max_tokens": 512, "temperature": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{s.port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    print(d["choices"][0]["message"].get("content"))


def main():
    s = load()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd in ("up", "down", "status", "logs", "convert") and (not s.worker or s.worker.startswith("CHANGE-ME")):
        sys.exit("set [cluster] worker and master_addr in local.toml (copy local.example.toml)")
    if cmd == "download":
        download(s)
    elif cmd == "convert":
        convert(s)
    elif cmd == "build":
        build(s)
    elif cmd == "up":
        up(s)
    elif cmd == "down":
        down(s)
    elif cmd == "status":
        status(s)
    elif cmd == "logs":
        logs(s, "--worker" in sys.argv)
    elif cmd == "chat":
        chat(s, " ".join(sys.argv[2:]) or "Hello!")
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
