#!/usr/bin/env python3
"""Clean N-filament colouring: one global graph-cut labelling instead of per-vertex snapping.

Same inputs and output format as colorize_3mf.py (textured GLB + Tripo convert_model 3MF ->
colorgroup 3MF, identical mesh and face order), but the colour decision is made for the whole
surface at once by minimising

    E = sum_faces  area * |colour(face) - palette[label]|                 (match the texture)
      + lam * sum_edges  length * crease(edge) * [label_i != label_j]     (short, creased boundaries)

crease() is ~1 on smooth surface and falls to ~0 at concave creases, the step where a raised tilak,
brow, bead or hem meets the surface. So:
  * boundaries slide onto the sculpted edges instead of wherever the blurry texture put them
    (no skin rim round a raised tilak);
  * the blend zone between two colours can't become a THIRD colour: a band costs two boundaries;
  * specks and pinholes cost their whole perimeter and are absorbed by their surroundings.
Solved with alpha-expansion (PyMaxflow). Leftover islands < --min-island mm2 merge into the
neighbour they share most boundary with. On a test figure this removed red robe-shading bleed, white fringes and the skin speckle inside
a raised tilak, with no per-model fixes.

    python3 scripts/clean_colorize.py <n>_textured.glb --geometry <n>_geometry.3mf \\
        -o <n>_clean.3mf --palette "#E8B08A,#EE6A1F,#3A2216,#B01C1C" --height 127 \\
        --up-axis y --geometry-up-axis z [--obj]

Constraints (repeatable; box = x0:x1,y0:y1,z0:z1 in output mm, same frame as the written 3MF):
    --only   "#B01C1C@-1:9,-3.2:-0.3,90.5:93.7"   colour allowed ONLY inside the box(es)
    --forbid "#EE6A1F@..."                         colour not allowed inside the box
    --prefer "#EE6A1F@...=12"                      nudge toward colour inside the box (default 12)
    --black-max-chroma 14   the darkest slot only where the texture is achromatic (soft
                            replacement for colorize_3mf --black-chroma; use only if it's BLACK)
Knobs: --lam (boundary cost, 6), --sigma (concave crease angle deg, 25), --min-island (0.4 mm2).
Texture colours are cached in <out>.feats.npz, so re-runs with new knobs take ~15 s.
Requires: pip install PyMaxflow
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]

sys.path.insert(0, str(Path(__file__).parent))
from colorize_3mf import hex_to_rgb, reorient, srgb_to_lab, to_feature, write_3mf, write_obj_mtl  # noqa: E402
from sculpt_features import shadow_islands, skin_index  # noqa: E402


def load_colour(glb, geom_path, height, up, gup):
    """Texture RGB per geometry vertex, normalised exactly like colorize_3mf.py (same mesh, same order)."""
    sc = trimesh.load(glb, process=False)
    mesh = sc.to_geometry() if isinstance(sc, trimesh.Scene) else sc
    assert isinstance(mesh, trimesh.Trimesh), "expected a single triangular mesh"
    reorient(mesh, up, "input")
    if height: mesh.apply_scale(height / mesh.extents[2])
    mesh.apply_translation([-mesh.bounds[:, 0].mean(), -mesh.bounds[:, 1].mean(), -mesh.bounds[0, 2]])
    vcol = np.asarray(mesh.visual.to_color().vertex_colors)[:, :3].astype(np.float64)
    gs = trimesh.load(geom_path, process=False)
    geom = gs.to_geometry() if isinstance(gs, trimesh.Scene) else gs
    assert isinstance(geom, trimesh.Trimesh), "expected a single triangular geometry mesh"
    reorient(geom, gup, "geometry")
    geom.apply_scale(mesh.extents[2] / geom.extents[2])
    geom.apply_translation([-geom.bounds[:, 0].mean(), -geom.bounds[:, 1].mean(), -geom.bounds[0, 2]])
    d, idx = cKDTree(mesh.vertices).query(geom.vertices)
    print(f"  colour-transfer distance: mean {d.mean():.3f}mm, max {d.max():.3f}mm")
    return geom, vcol[idx]


def solve(mesh, vcol, palette, lam=6.0, sigma=25.0, convex_sigma=60.0, luma=0.5,
          allowed=None, bias=None, iters=3, min_island=0.4):
    import maxflow
    F = mesh.faces; n = len(F); area = mesh.area_faces
    ei, ej = mesh.face_adjacency.T
    ev = mesh.face_adjacency_edges
    L = np.linalg.norm(mesh.vertices[ev[:, 0]] - mesh.vertices[ev[:, 1]], axis=1)
    ang = np.degrees(mesh.face_adjacency_angles)
    crease = np.where(mesh.face_adjacency_convex, np.exp(-(ang / convex_sigma) ** 2), np.exp(-(ang / sigma) ** 2))
    w = lam * L * crease
    pal = np.array([hex_to_rgb(h) for h in palette], float)
    ff = to_feature(vcol, "lab", luma)[F].mean(1)
    U = np.linalg.norm(ff[:, None, :] - to_feature(pal, "lab", luma)[None], axis=2) * area[:, None]
    if bias is not None: U = U + bias * area[:, None]
    if allowed is not None: U = np.where(allowed, U, 1e9)
    lab = U.argmin(1)
    energy = lambda lb: U[np.arange(n), lb].sum() + w[lb[ei] != lb[ej]].sum()
    print(f"  nearest-colour start: E={energy(lab):.0f}")
    for it in range(iters):
        changed = 0
        for a in range(len(palette)):          # alpha-expansion move for label a (Potts, submodular)
            li, lj = lab[ei], lab[ej]
            A = w * (li != lj); B = w * (li != a); C = w * (lj != a)
            lin = U[:, a] - U[np.arange(n), lab]
            np.add.at(lin, ei, C - A); np.add.at(lin, ej, -C)
            g = maxflow.Graph[float](n, len(ei)); g.add_nodes(n)
            g.add_edges(ei, ej, B + C - A, np.zeros(len(ei)))
            g.add_grid_tedges(np.arange(n), np.maximum(lin, 0), np.maximum(-lin, 0))
            g.maxflow()
            new = np.where(g.get_grid_segments(np.arange(n)), a, lab)
            changed += int((new != lab).sum()); lab = new
        print(f"  expansion pass {it + 1}: E={energy(lab):.0f}, {changed} faces changed")
        if not changed: break
    # islands smaller than min_island mm2 -> the neighbour label they share most boundary with
    moved = 0
    for _ in range(5):
        same = lab[ei] == lab[ej]
        nc, comp = connected_components(coo_matrix((np.ones(same.sum()), (ei[same], ej[same])), shape=(n, n)), directed=False)
        small = np.bincount(comp, weights=area, minlength=nc)[comp] < min_island
        if not small.any(): break
        d_ = ~same; a_, b_ = ei[d_], ej[d_]
        rows = np.r_[comp[a_], comp[b_]]; nb = np.r_[lab[b_], lab[a_]]; ln = np.r_[L[d_], L[d_]]
        k = small[np.r_[a_, b_]]
        Mx = coo_matrix((ln[k], (rows[k], nb[k])), shape=(nc, len(palette))).toarray()
        new = np.where(small & (Mx[comp].sum(1) > 0), Mx.argmax(1)[comp], lab)
        if allowed is not None: new = np.where(allowed[np.arange(n), new], new, lab)
        step = int((new != lab).sum()); moved += step; lab = new
        if not step: break
    print(f"  islands < {min_island} mm2 merged: {moved} faces")
    return lab


def lab_to_srgb(lab):
    fy = (lab[:, 0] + 16) / 116
    f = np.stack([fy + lab[:, 1] / 500, fy, fy - lab[:, 2] / 200], 1)
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (116 * f - 16) / 903.3) * np.array([0.95047, 1.0, 1.08883])
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    lin = np.clip(xyz @ np.linalg.inv(M).T, 0, 1)
    return np.clip(np.where(lin > 0.0031308, 1.055 * lin ** (1 / 2.4) - 0.055, lin * 12.92) * 255, 0, 255)


def vertex_ao(mesh, rays=32):
    """per-vertex ambient occlusion (1 open, 0 buried); render_pretty caches it by mesh hash. None without embree."""
    sys.path.insert(0, str(Path(__file__).parent))
    from render_pretty import bake_ao
    try:
        return bake_ao(mesh.vertices, mesh.faces, np.asarray(mesh.vertex_normals, np.float64), rays)
    except ImportError as e:
        print(f"  no ambient occlusion ({e}): pip install embreex -- deshade and shadow fix skipped")
        return None


def isotonic(x, y, w):
    """weighted least-squares non-decreasing fit (pool adjacent violators); x sorted"""
    blocks = [[float(y[k]), float(w[k]), 1] for k in range(len(y))]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2, n2 = out.pop(); v1, w1, n1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    return np.concatenate([[v] * n for v, _, n in out])


def deshade(vcol, ao, pal_rgb, strength=1.0):
    """Tripo bakes ambient occlusion into its texture: shaded skin and cloth go dark and desaturated, which is
    exactly what pulls them onto the black filament ("black patches wherever there is shadow"). Measure how much
    darker each mid-tone palette colour gets as our own AO drops (one test figure: skin L 63 open -> 34 buried,
    khaki 59 -> 36, while real black stays at 11) and divide that back out: L / r(ao), chroma / r^0.6."""
    lab = srgb_to_lab(vcol); pal = srgb_to_lab(pal_rgb)
    bins = np.linspace(0, 1.001, 11); mids = (bins[:-1] + bins[1:]) / 2
    f = lab * [0.5, 1, 1]; lab_i = np.argmin(((f[:, None] - pal[None] * [0.5, 1, 1]) ** 2).sum(2), 1)
    R, W = [], []
    for i, (L, a_, b_) in enumerate(pal):
        if not 22 < L < 88 or (np.hypot(a_, b_) < 6 and not 30 < L < 80):
            continue                            # black / white / near-greys: nothing to calibrate on
        sel = lab_i == i; op = sel & (ao > 0.85)
        if op.sum() < 500:
            continue
        L0 = np.median(lab[op, 0])
        R.append([np.median(lab[sel & (ao >= b0) & (ao < b1), 0]) / L0 if (sel & (ao >= b0) & (ao < b1)).sum() > 100
                  else np.nan for b0, b1 in zip(bins[:-1], bins[1:])]); W.append(op.sum())
    if R:
        R = np.array(R); Wt = np.array(W, float)[:, None] * np.isfinite(R)
        curve = np.nansum(np.nan_to_num(R) * Wt, 0) / np.maximum(Wt.sum(0), 1e-9)
        ok = Wt.sum(0) > 0
        curve = isotonic(mids[ok], curve[ok], Wt.sum(0)[ok])      # darker with more occlusion, never a noisy bin
        curve = np.clip(np.interp(mids, mids[ok], curve), 0.4, 1.0)
    else:
        curve = 0.55 + 0.45 * mids
    print("  deshade L(ao)/L(open): " + " ".join(f"{c:.2f}" for c in curve) + "  (ao 0.05 .. 0.95)")
    r = 1 - min(max(strength, 0.0), 1.0) * (1 - np.interp(ao, mids, curve))
    out = lab.copy(); out[:, 0] = np.minimum(lab[:, 0] / r, 100); out[:, 1:] = lab[:, 1:] / r[:, None] ** 0.6
    return lab_to_srgb(out)


def parse_box(spec):
    hexcol, rest = spec.split("@", 1)
    val = None
    if "=" in rest: rest, val = rest.split("=", 1)
    (x0, x1), (y0, y1), (z0, z1) = [map(float, r.split(":")) for r in rest.split(",")]
    return hexcol.strip().upper(), (x0, x1, y0, y1, z0, z1), (float(val) if val else None)


def in_box(C, b):
    return (C[:, 0] >= b[0]) & (C[:, 0] <= b[1]) & (C[:, 1] >= b[2]) & (C[:, 1] <= b[3]) & (C[:, 2] >= b[4]) & (C[:, 2] <= b[5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="textured GLB (colour source)")
    ap.add_argument("--geometry", required=True, help="Tripo convert_model 3MF (watertight geometry)")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--palette", required=True)
    ap.add_argument("--height", type=float)
    ap.add_argument("--up-axis", default="auto", choices=["auto", "x", "y", "z"])
    ap.add_argument("--geometry-up-axis", default="auto", choices=["auto", "x", "y", "z"])
    ap.add_argument("--lam", type=float, default=6.0)
    ap.add_argument("--sigma", type=float, default=25.0)
    ap.add_argument("--luma-weight", type=float, default=0.5)
    ap.add_argument("--min-island", type=float, default=0.4)
    ap.add_argument("--black-max-chroma", type=float)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--forbid", action="append", default=[])
    ap.add_argument("--prefer", action="append", default=[])
    ap.add_argument("--deshade", type=float, default=1.0,
                    help="divide the texture's baked ambient occlusion back out (0 = off, 1 = full; default 1)")
    ap.add_argument("--no-shadow-fix", action="store_true",
                    help="keep dark islands that sit in deep shadow (default: re-label them, black forbidden there)")
    ap.add_argument("--skin", help="skin hex (auto-detected) -- skin-bordered shadow islands must be >= 20 mm2")
    ap.add_argument("--obj", action="store_true")
    a = ap.parse_args()

    out = Path(a.output)
    cache = out.with_suffix(".feats.npz")
    if cache.exists():
        z = np.load(cache); mesh = trimesh.Trimesh(z["V"], z["F"], process=False); vcol = z["vcol"].astype(float)
        print(f"  using cached texture colours {cache.name} (delete it if the inputs changed)")
    else:
        mesh, vcol = load_colour(a.input, a.geometry, a.height, a.up_axis, a.geometry_up_axis)
        np.savez_compressed(cache, V=mesh.vertices, F=mesh.faces, vcol=vcol.astype(np.float32))
    palette = [h.strip().upper() for h in a.palette.split(",") if h.strip()]
    pal_rgb = np.array([hex_to_rgb(h) for h in palette], float)
    ao = vertex_ao(mesh) if (a.deshade > 0 or not a.no_shadow_fix) else None
    vtex = vcol
    if a.deshade > 0 and ao is not None:
        vcol = deshade(vcol, ao, pal_rgb, a.deshade)
    C = mesh.triangles_center
    allowed = np.ones((len(mesh.faces), len(palette)), bool); bias = np.zeros_like(allowed, float)
    boxes = {}
    for s in a.only:
        h, b, _ = parse_box(s); boxes.setdefault(h, np.zeros(len(C), bool)); boxes[h] |= in_box(C, b)
    for h, m in boxes.items(): allowed[~m, palette.index(h)] = False
    for s in a.forbid:
        h, b, _ = parse_box(s); allowed[in_box(C, b), palette.index(h)] = False
    for s in a.prefer:
        h, b, v = parse_box(s); bias[in_box(C, b), palette.index(h)] -= (v if v is not None else 12.0)
    L_ = [srgb_to_lab(np.array([hex_to_rgb(h)], float))[0, 0] for h in palette]
    dark = int(np.argmin(L_))
    if a.black_max_chroma is not None:
        lab_v = srgb_to_lab(vtex); chroma = np.hypot(lab_v[:, 1], lab_v[:, 2])[mesh.faces].mean(1)
        allowed[chroma > a.black_max_chroma, dark] = False

    lab = solve(mesh, vcol, palette, lam=a.lam, sigma=a.sigma, luma=a.luma_weight,
                allowed=allowed, bias=bias, min_island=a.min_island)
    if not a.no_shadow_fix and ao is not None:
        skin = palette.index(a.skin.upper()) if a.skin else skin_index(palette)
        fao = ao[mesh.faces].mean(1)
        light = np.array(L_)
        isl, k, area = shadow_islands(mesh, lab, dark, fao, skin, light=light)
        if k:
            print(f"  shadow fix: {k} shadow islands ({area:.1f} mm2) -> re-solve with their colour forbidden there")
            f_ = np.where(isl)[0]
            others = allowed[f_].sum(1) > 1           # never strip a face's last allowed colour (--only / --forbid win)
            allowed[f_[others], lab[f_[others]]] = False
            lab = solve(mesh, vcol, palette, lam=a.lam, sigma=a.sigma, luma=a.luma_weight,
                        allowed=allowed, bias=bias, min_island=a.min_island)
    write_3mf(mesh, lab, pal_rgb, out, encoding="colorgroup")
    if a.obj: write_obj_mtl(mesh, lab, pal_rgb, out.with_suffix(".obj"))
    cnt = np.bincount(lab, weights=mesh.area_faces, minlength=len(palette))
    print("Area per colour: " + "  ".join(f"{h} {c / cnt.sum() * 100:.1f}%" for h, c in zip(palette, cnt)))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
