#!/usr/bin/env python3
"""Measure exact per-tensor bytes from safetensors headers over HTTP Range.

No weights are downloaded: each shard is range-read (first ~2.5 MB), which holds
the safetensors header (8-byte LE length prefix + JSON of dtype/shape/offsets).
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

PROXY = "http://127.0.0.1:1099"
OUT = Path("/tmp/qwen38-measure")
OUT.mkdir(parents=True, exist_ok=True)


def curl(url: str, out: Path, rng: str | None = None) -> bool:
    cmd = ["curl", "-sS", "--max-time", "180", "-x", PROXY, "-L"]
    if rng:
        cmd += ["-r", rng]
    cmd += ["-o", str(out), url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and out.exists() and out.stat().st_size > 0


def tree(repo: str) -> list[dict]:
    f = OUT / (repo.replace("/", "__") + ".tree.json")
    if not curl(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true", f):
        raise SystemExit(f"tree fetch failed: {repo}")
    return json.loads(f.read_text())


def header(repo: str, path: str) -> dict:
    f = OUT / (repo.replace("/", "__") + "___" + path.replace("/", "_") + ".hdr")
    if not f.exists() or f.stat().st_size == 0:
        if not curl(f"https://huggingface.co/{repo}/resolve/main/{path}", f, "0-2500000"):
            raise SystemExit(f"header fetch failed: {repo}/{path}")
    raw = f.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    if n > len(raw) - 8:
        raise SystemExit(f"header truncated {repo}/{path}: need {n}, have {len(raw)-8}")
    return json.loads(raw[8 : 8 + n])


def measure(repo: str) -> dict:
    files = [e for e in tree(repo) if e["type"] == "file" and e["path"].endswith(".safetensors")]
    files.sort(key=lambda e: e["path"])
    print(f"== {repo}: {len(files)} shards", flush=True)
    tensors: dict[str, dict] = {}
    for e in files:
        h = header(repo, e["path"])
        meta = h.get("__metadata__", {})
        cnt = 0
        for name, t in h.items():
            if name == "__metadata__":
                continue
            off = t["data_offsets"]
            nbytes = off[1] - off[0]
            rec = {
                "dtype": t["dtype"],
                "shape": t["shape"],
                "bytes": nbytes,
                "shard": e["path"],
            }
            if name in tensors:
                raise SystemExit(f"duplicate tensor {name} in {repo}")
            tensors[name] = rec
            cnt += 1
        print(f"   {e['path']:34s} {cnt:5d} tensors  meta={meta.get('total_size','?')}", flush=True)
    payload = sum(t["bytes"] for t in tensors.values())
    disk = sum((e.get("lfs") or {}).get("size") or e.get("size") or 0 for e in files)
    out = {
        "repo": repo,
        "shards": len(files),
        "tensor_count": len(tensors),
        "payload_bytes": payload,
        "disk_bytes": disk,
        "tensors": tensors,
    }
    (OUT / (repo.replace("/", "__") + ".measured.json")).write_text(json.dumps(out, indent=1))
    print(f"   payload {payload/1e9:.3f} GB · disk {disk/1e9:.3f} GB · tensors {len(tensors)}", flush=True)
    return out


if __name__ == "__main__":
    for repo in sys.argv[1:]:
        measure(repo)
