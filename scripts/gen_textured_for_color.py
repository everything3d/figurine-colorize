#!/usr/bin/env python3
"""Watertight colored-3MF prep: generate a textured GLB (color source) +
a convert_model 3MF (watertight geometry) for one image, so colorize_3mf.py can
fuse them. Run, then call colorize_3mf.py <glb> --geometry <3mf>.

Usage: python scripts/gen_textured_for_color.py <image> <out_dir>

Task ids are written to <out_dir>/<stem>_tasks.json as soon as they are submitted and
reused on a re-run, so a client-side timeout or a killed terminal costs nothing — run the
same command again and it resumes polling the task already paid for. Tripo has no
task-listing endpoint, so an id that never reached disk is unrecoverable.
"""
import json
import os
import sys
import time
from pathlib import Path

import requests

API_BASE = "https://openapi.tripo3d.ai/v3"
POLL = 5
# a two-figure sculpture legitimately runs past 15 minutes; the old 900s ceiling
# abandoned tasks that were still progressing
TASK_TIMEOUT = 3600


def H():
    return {"Authorization": f"Bearer {os.environ['TRIPO_API_KEY']}"}


def upload(image: Path) -> str:
    ext = image.suffix.lstrip(".").replace("jpg", "jpeg")
    with open(image, "rb") as f:
        r = requests.post(f"{API_BASE}/files", headers=H(),
                          files={"file": (image.name, f, f"image/{ext}")}, timeout=300)
    r.raise_for_status()
    d = r.json()
    if d.get("code") != 0:
        raise RuntimeError(d)
    return d["data"]["file_token"]


def task(endpoint: str, payload: dict) -> str:
    r = requests.post(f"{API_BASE}{endpoint}", headers={**H(), "Content-Type": "application/json"},
                      json=payload, timeout=30)
    r.raise_for_status()
    d = r.json()
    if d.get("code") != 0:
        raise RuntimeError(d)
    return d["data"]["task_id"]


def wait(task_id: str, label: str, timeout=TASK_TIMEOUT) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{API_BASE}/tasks/{task_id}", headers=H(), timeout=30)
        r.raise_for_status()
        d = r.json()["data"]
        st = d["status"]
        if st == "success":
            print(f"  + {label} done")
            return d
        # V3 folds banned/expired into "failed" (error_code 2008/2018); anything that is
        # not queued/running is terminal, so an unknown status can't poll forever.
        if st not in ("queued", "running"):
            raise RuntimeError(f"{label} {st}: credits_consumed={d.get('credits_consumed')} "
                               f"progress={d.get('progress')} error={d.get('error_code')}: {d.get('error_message')} "
                               f"output={d.get('output')}")
        print(f"  ... {label} {st} ({int(time.time()-start)}s)")
        time.sleep(POLL)
    raise RuntimeError(f"{label} timed out")


def submit_once(store: Path, key: str, endpoint: str, payload: dict) -> str:
    """Submit a task, or reuse the id from a previous run. Every task id hits disk before
    we start polling — Tripo cannot list tasks, so an unsaved id is a paid-for orphan."""
    saved = json.loads(store.read_text()) if store.exists() else {}
    if key in saved:
        print(f"  reusing {key} {saved[key]} from {store.name}")
        return saved[key]
    tid = task(endpoint, payload)
    saved[key] = tid
    store.write_text(json.dumps(saved, indent=2))
    print(f"  {key} {tid}  (saved to {store.name})")
    return tid


def main():
    if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
        sys.exit("usage: gen_textured_for_color.py <image.png> <out_dir>")
    if not os.environ.get("TRIPO_API_KEY"):
        sys.exit("TRIPO_API_KEY is not set (get one at https://platform.tripo3d.ai)")
    image = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    stem = image.stem
    store = out / f"{stem}_tasks.json"

    if store.exists():
        print(f"Resuming from {store.name}")
        token = None
    else:
        print("Uploading image...")
        token = upload(image)

    print("Submitting textured image_to_model (texture=True, pbr=False)...")
    model_task = submit_once(store, "model_task", "/generation/image-to-model", {
        "model_version": "v3.1-20260211",
        "texture": True,
        "pbr": False,
        "file": {"type": image.suffix.lstrip(".").lower().replace("jpeg", "jpg"), "file_token": token},
    })
    model = wait(model_task, "model")

    glb_url = model["output"]["model_url"]
    glb_path = out / f"{stem}_textured.glb"
    if not glb_path.exists():
        print(f"Downloading textured GLB -> {glb_path.name}")
        glb_path.write_bytes(requests.get(glb_url, timeout=600).content)

    print("Submitting convert_model -> 3mf (texture=True) for watertight geometry...")
    conv_task = submit_once(store, "convert_task", "/models/convert", {
        "format": "3mf",
        "texture": True,
        "original_model_task_id": model_task,
    })
    conv = wait(conv_task, "convert")
    mf_url = conv["output"]["model_url"]
    mf_path = out / f"{stem}_geometry.3mf"
    print(f"Downloading watertight 3MF -> {mf_path.name}")
    mf_path.write_bytes(requests.get(mf_url, timeout=600).content)

    print(f"\nDone.\n  color source: {glb_path}\n  geometry:     {mf_path}")
    print(f"  model_task_id={model_task}")


if __name__ == "__main__":
    main()
