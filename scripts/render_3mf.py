#!/usr/bin/env python3
"""Render a colored 3MF exactly as the slicer will show it.

Reads the file's per-triangle `p1` indices against its `<m:colorgroup>` palette and
rasterizes with a real depth test, from several angles.

Use this — NOT `colorize_3mf.py --preview` — to sign off on a colored model. That
flag's point-splat renderer has a weak occlusion test and fakes heavy salt-and-pepper
speckle on front-facing surfaces. Several tuning
runs have been wasted chasing speckle that was never in the model.

Default look is the studio render from render_pretty.py: smooth shading, key/fill/rim lights,
ambient occlusion, soft shadows, bright background. The old flat look read as "too dark". Colours are still the file's own per-triangle colours.
`--flat` gives the old checker: one headlight, no shadows, faceted. Use it when you need to judge
a colour boundary without any shading (AO darkens creases, which can hide a thin bleed). It is
also the automatic fallback when moderngl is not installed.

Usage:
  python3 scripts/render_3mf.py <colored.3mf> -o <out.png> [--angles 0,45,135,180,270] [--flat]
"""

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

NS_COLOR = r'<m:color color="(#[0-9a-fA-F]{6})"'
NS_VERT = r'<vertex x="([-\d.eE+]+)" y="([-\d.eE+]+)" z="([-\d.eE+]+)"'
NS_TRI = r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"(?:\s+p1="(\d+)")?'
NS_TRI_PAINT = r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"(?:\s+paint_color="([^"]*)")?'
# Bambu paint codes map monotonically onto filament slots 1..N (same table as split_painted_3mf.py)
SLOT_CODES = ["4", "8", "0C", "1C", "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C"]   # slots 1..12; state>=3 is 11 + 4 bits LSB-first


def hexes_to_rgb(hexes):
    return np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in hexes], dtype=float)


def load_painted_3mf(path: Path):
    """Bambu project 3MF painted with per-triangle `paint_color`. Palette comes from the
    file's own `filament_colour` — never inferred (see the rank-matching trap in
    the figurine-split skill)."""
    import json
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        entry = next((n for n in names if re.fullmatch(r"3D/Objects/object_\d+\.model", n)), None)
        if entry is None:
            return None
        xml = z.read(entry).decode("utf-8", "ignore")
        # scan the whole file: the triangles sit behind the entire <vertices> block, which on a
        # figurine is tens of MB, so any head-of-file window misses the paint
        if 'paint_color="' not in xml:
            return None
        try:
            cfg = json.loads(z.read("Metadata/project_settings.config").decode())
        except KeyError:
            raise SystemExit(f"{path.name}: painted mesh but no project_settings.config to read the palette from")

    palette = hexes_to_rgb(cfg["filament_colour"])
    verts = np.array(re.findall(NS_VERT, xml), dtype=float)
    tris = re.findall(NS_TRI_PAINT, xml)
    faces = np.array([[int(a), int(b), int(c)] for a, b, c, _ in tris], dtype=np.int64)
    # unpainted faces take the object's extruder, which is slot 1
    slot = {c: i for i, c in enumerate(SLOT_CODES)}
    pidx = np.array([slot.get(p[:2] if len(p) > 2 else p, 0) for *_, p in tris], dtype=np.int64)
    pidx = np.clip(pidx, 0, len(palette) - 1)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh, palette[pidx], palette


def load_colored_3mf(path: Path):
    """Return (mesh, per-face RGB, palette). Handles both encodings this repo produces:
    `<m:colorgroup>` + `p1` (colorize_3mf.py) and Bambu `paint_color` (split_painted_3mf.py,
    colorgroup_to_painted.py). Faces with no color fall back to palette[0]."""
    painted = load_painted_3mf(path)
    if painted is not None:
        return painted

    with zipfile.ZipFile(path) as z:
        xml = z.read("3D/3dmodel.model").decode()

    palette = hexes_to_rgb(re.findall(NS_COLOR, xml))
    if not len(palette):
        raise SystemExit(f"{path.name}: no <m:colorgroup> palette and no paint_color — "
                         "is this a colorized or painted 3MF?")

    verts = np.array(re.findall(NS_VERT, xml), dtype=float)
    tris = re.findall(NS_TRI, xml)
    faces = np.array([[int(a), int(b), int(c)] for a, b, c, _ in tris], dtype=np.int64)
    pidx = np.array([int(p) if p else 0 for *_, p in tris], dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh, palette[pidx], palette


def render(mesh, face_rgb, azimuth, width=460, height=600, supersample=2):
    """Painter's-algorithm rasterizer with a per-face depth test. Camera looks along +Y."""
    m = mesh.copy()
    m.apply_transform(
        trimesh.transformations.rotation_matrix(np.radians(azimuth), [0, 0, 1], m.centroid)
    )
    v = m.vertices
    W, H = width * supersample, height * supersample
    lo, hi = v.min(0), v.max(0)
    scale = min(W * 0.88 / (hi[0] - lo[0]), H * 0.92 / (hi[2] - lo[2]))
    px = (v[:, 0] - (lo[0] + hi[0]) / 2) * scale + W / 2
    py = H - ((v[:, 2] - lo[2]) * scale + H * 0.04)

    # simple headlight from front-top-left so relief reads
    shade = np.clip(m.face_normals @ np.array([-0.35, -0.85, 0.4]), 0, 1) * 0.55 + 0.45
    tri = np.stack([px[m.faces], py[m.faces]], -1)
    depth = v[:, 1][m.faces].mean(1)

    zbuf = np.full((H, W), np.inf)
    img = np.full((H, W, 3), 255, np.uint8)
    for i in np.argsort(-depth):  # far to near
        t = tri[i]
        x0, x1 = int(max(t[:, 0].min(), 0)), int(min(np.ceil(t[:, 0].max()), W - 1))
        y0, y1 = int(max(t[:, 1].min(), 0)), int(min(np.ceil(t[:, 1].max()), H - 1))
        if x1 < x0 or y1 < y0:
            continue
        det = (t[1, 0] - t[0, 0]) * (t[2, 1] - t[0, 1]) - (t[2, 0] - t[0, 0]) * (t[1, 1] - t[0, 1])
        if abs(det) < 1e-9:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        w1 = ((xs - t[0, 0]) * (t[2, 1] - t[0, 1]) - (t[2, 0] - t[0, 0]) * (ys - t[0, 1])) / det
        w2 = ((t[1, 0] - t[0, 0]) * (ys - t[0, 1]) - (xs - t[0, 0]) * (t[1, 1] - t[0, 1])) / det
        hit = (w1 >= -1e-6) & (w2 >= -1e-6) & (w1 + w2 <= 1 + 1e-6)
        hit &= depth[i] < zbuf[y0:y1 + 1, x0:x1 + 1]
        if not hit.any():
            continue
        zbuf[y0:y1 + 1, x0:x1 + 1][hit] = depth[i]
        img[y0:y1 + 1, x0:x1 + 1][hit] = np.clip(face_rgb[i] * shade[i], 0, 255).astype(np.uint8)

    return Image.fromarray(img).resize((width, height), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser(description="Render a colorgroup 3MF as the slicer will show it")
    ap.add_argument("input", type=Path, help="colored 3MF (from colorize_3mf.py)")
    ap.add_argument("-o", "--output", type=Path, required=True, help="output PNG (views tiled)")
    ap.add_argument("--angles", default="0,45,135,180,270",
                    help="comma-separated azimuths in degrees (default 0,45,135,180,270)")
    ap.add_argument("--width", type=int, default=460, help="per-view width (default 460)")
    ap.add_argument("--height", type=int, default=600, help="per-view height (default 600)")
    ap.add_argument("--flat", action="store_true",
                    help="old flat checker look (no shadows/AO) for judging colour edges")
    args = ap.parse_args()

    mesh, face_rgb, palette = load_colored_3mf(args.input)
    counts = np.array([(face_rgb == c).all(1).sum() for c in palette])
    print(f"{args.input.name}: {len(mesh.vertices)} verts, {len(mesh.faces)} tris, "
          f"{len(palette)} colors")
    for c, n in zip(palette, counts):
        print(f"  #{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}  {n:>9d} faces "
              f"({100 * n / len(face_rgb):5.1f}%)")

    angles = [float(a) for a in args.angles.split(",") if a.strip()]
    views = None
    if not args.flat:
        try:
            from render_pretty import render_views
            views = [im for _, im in render_views(args.input, angles=args.angles,
                                                  width=args.width, height=args.height)]
        except ImportError as e:
            print(f"  studio render unavailable ({e}); falling back to --flat. pip install moderngl")
    if views is None:
        views = [render(mesh, face_rgb, a, args.width, args.height) for a in angles]
    canvas = Image.new("RGB", (args.width * len(views), args.height), "white")
    for i, im in enumerate(views):
        canvas.paste(im, (i * args.width, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Wrote {args.output} ({len(views)} views: {', '.join(str(int(a)) + 'deg' for a in angles)})")


if __name__ == "__main__":
    main()
