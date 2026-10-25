#!/usr/bin/env python3
"""Paint face details (eyes, eyebrows, teeth) onto an already-coloured 3MF, by hand,
AFTER the automatic colouring is done.

Why a separate pass: eyes, brows and teeth are 1-3 mm features. The texture is too blurry there for
any automatic labeller to get them right (clean_colorize.py turns a whole eye into one black blob,
and a pale brow vanishes into skin). So we place them deliberately, like painting in the slicer, but
reproducibly and from the command line.

How it works: look at the model from one direction (--view, same azimuth convention as
render_3mf.py; the camera looks along +Y after rotating the model by AZ about the z axis). In that
view, u = horizontal mm and v = height mm (v is the model's z). Each shape is drawn in those (u, v)
mm and colours every face that is visible from the view, faces the camera, and has its centroid
inside the shape. Shapes apply in the order given.

  ellipse:HEX@u,v,rx,ry[,deg]            filled ellipse (rotation in degrees)
  stroke:HEX@w;u1,v1;u2,v2;...           polyline brush of width w mm (eyebrows, teeth bands)
  ...[|on=HEX+HEX]                       only repaint faces currently one of these colours
  ...[|smooth=N]                         poly only: round the outline (N Chaikin passes), so a traced
                                         beard / hair line is a curve, not straight segments
  ...[|refine=0]                         leave this shape out of --refine (big cleanup polys: refining a
                                         whole turban multiplies its triangles 16x for nothing)
  ...[|depth=MM]                         also paint faces within MM of the front surface whatever way they
                                         face (fills a sculpted crease / a beard's side edge); lips: 0.6
  poly:HEX@u1,v1;u2,v2;...              filled polygon (trace a turban edge / beard line off the sculpt;
                                         boxes leave straight stray-colour edges)
  lips:HEX@up,low[,bow];u1,v1;u2,v2;... upper + lower lip from the sculpted MOUTH LINE traced corner to
        corner (u1,v1 ... un,vn). Upper lip up mm thick at the centre with a cupid's bow (dip = bow x up,
        default 0.3), lower lip low mm and rounder; both taper to points at the corners. The two meet on
        the mouth crease, so the sculpt's own shadow parts them. A single stroke band looks awful.
        Upper lips face down and sit in shadow: at 0.5 mm they vanish in a render, use ~0.6-0.75.
  mouth:LIP,TEETH,INSIDE@up,low,open[,teeth];u1,v1;...  an OPEN (laughing) mouth on the same traced mouth
        line: dark INSIDE opening open mm tall (30% above the line, 70% below), TEETH band along its top
        (teeth mm, default 0.3), upper lip up mm above it and lower lip low mm below it. Use this when
        the render's mouth is open even if the sculpt closed it.
  eye:WHITE,IRIS,SKIN@u,v,rx,ry[,deg[,iris[,lid]]]
        a whole cartoon eye, the way the renders draw them: clears the auto-coloured eye blob
        (IRIS-coloured faces just outside the eye go back to SKIN), an almond WHITE, a big IRIS
        that fills the eye height (radius = iris * ry, default 1.05, clipped to the eye so white
        only shows at the corners), and an IRIS upper-lid line of width lid mm (default 0.3, 0 = none).
        A small iris floating in a white ring reads as a horror stare -- don't.

SCULPT-FOLLOWING shapes (scripts/sculpt_features.py; one seed point, the geometry decides the outline -- the way a person
repaints these files by hand in the slicer). Run before the 2D shapes, unrefined mesh:
  eye3d:WHITE,IRIS,SKIN@u,v[,liner][|clear=0|iris=1.1|catch=0.43|catchmin=2|liner=1.45|dome=1]
        white = the visible sculpted eyeball (sphere + rim crease), iris 1.1 x its height clipped by the lids,
        catchlight up-right; `liner` (women) = the whole upper lid to the lid fold, with an outer wing. Men: none.
  lips3d:LIP,SKIN@u1,v1;u2,v2;...[|keep=TEETH+INSIDE|teeth=TEETH|w=MM]
        both sculpted lip bulges round the mouth line (traced corner to corner, or one point to auto-trace),
        ends in the corner pits; the opening between the lips -> teeth colour.
  grow:DARK>INTO[+INTO][@u0,u1,v0,v1][|d=1.2|eps=0.5|skip=u,v,r;...]
        brows / beard / moustache / hairline out to the concave crease at the foot of the sculpted feature
        (skip: front-view discs left alone, e.g. eyes kept as the texture painted them).
  shadowfix:DARK[,SKIN][|ao=0.25|min=2|skinmin=20|all=1]
        islands of a darker colour lying in deep shadow -> the lighter colour around them (all=0: black only).
--auto-face [--women 0,2] [--lip HEX]  finds every face and writes these shapes to <output>.features.txt.

--refine N   split the triangles under (and 0.4 mm around) every shape N times before painting
             (red-green refinement: the mesh stays watertight). Tripo meshes are ~0.18 mm per edge
             with slivers to 0.7 mm, i.e. ~8 triangles across an iris, so without this every edge is
             spiky. 2 levels -> ~0.05 mm. Changes the face count, so the output is a new mesh.

Global relabels (before painting): --remap "#OLD>#NEW" (face assignment untouched -- remap, don't
re-quantise). A HEX not in the palette is appended as a new colour.

Shapes can come from --paint (repeatable) and/or --spec FILE (one shape per line, # comments), so a
project keeps them in one file and build scripts can reuse them via apply_features().

Find coordinates first with a grid preview of a window (u0,u1,v0,v1 in mm):
  paint_features.py in.3mf --view 270 --preview head.png --window=5,30,120,145 [--spec ...]
(--window= with '=' : a negative first number otherwise parses as a flag.) The preview draws the
paint, a 1 mm grid (labels every 5 mm) and every shape's outline.

Writes a colorgroup 3MF + OBJ/MTL, like clean_colorize.py.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from colorize_3mf import write_3mf, write_obj_mtl  # noqa: E402
from render_3mf import load_colored_3mf  # noqa: E402


def hex_rgb(h):
    h = h.strip().lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float)


def view_coords(V, az):
    """u (horizontal), v (= z), depth (smaller = nearer the camera), as in render_3mf but about the origin."""
    a = np.radians(az)
    x = V[:, 0] * np.cos(a) - V[:, 1] * np.sin(a)
    y = V[:, 0] * np.sin(a) + V[:, 1] * np.cos(a)
    return x, V[:, 2], y


# ---------------------------------------------------------------- shapes

def _ellipse(hexcol, u, v, rx, ry, deg=0.0, on=None):
    return dict(kind="ellipse", hex=hexcol, on=on, u=u, v=v, rx=rx, ry=ry, deg=deg)


def _stroke(hexcol, w, pts, on=None):
    return dict(kind="stroke", hex=hexcol, on=on, w=w, pts=np.asarray(pts, float))


def _poly(hexcol, pts, on=None):
    return dict(kind="poly", hex=hexcol, on=on, pts=np.asarray(pts, float))


def expand_lips(hexcol, up, low, line, bow=0.3, n=80):
    L = np.asarray(line, float)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(L, axis=0), axis=1))]
    t = np.linspace(0, 1, n); s = t * seg[-1]
    P = np.stack([np.interp(s, seg, L[:, 0]), np.interp(s, seg, L[:, 1])], 1)
    tan = np.gradient(P, axis=0); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
    nrm *= np.sign(nrm[:, 1:2] + 1e-9)                                      # always pointing up (+v)
    env = np.sin(np.pi * t)
    hu = up * env ** 0.7 * (1 - bow * np.exp(-((t - 0.5) / 0.07) ** 2)) \
        * (1 + 0.12 * np.exp(-((np.abs(t - 0.5) - 0.13) / 0.06) ** 2))     # cupid's bow: dip + two peaks
    hl = low * env ** 0.9                                                  # fuller, rounder lower lip
    upper = np.vstack([P, (P + nrm * hu[:, None])[::-1]])
    lower = np.vstack([P, (P - nrm * hl[:, None])[::-1]])
    return [_poly(hexcol, upper), _poly(hexcol, lower)]


def _line(line, n=80):
    L = np.asarray(line, float)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(L, axis=0), axis=1))]
    t = np.linspace(0, 1, n); s = t * seg[-1]
    P = np.stack([np.interp(s, seg, L[:, 0]), np.interp(s, seg, L[:, 1])], 1)
    tan = np.gradient(P, axis=0); tan /= np.linalg.norm(tan, axis=1, keepdims=True)
    nrm = np.stack([-tan[:, 1], tan[:, 0]], 1); nrm *= np.sign(nrm[:, 1:2] + 1e-9)
    return t, P, nrm, np.sin(np.pi * t)


def _band(P, nrm, a, b):
    """polygon between offsets a and b (per point, mm along the up-normal)"""
    return np.vstack([P + nrm * a[:, None], (P + nrm * b[:, None])[::-1]])


def expand_mouth(lip, teeth, inside, up, low, opn, th=0.3, line=None):
    t, P, nrm, env = _line(line)
    top, bot = 0.3 * opn * env, -0.7 * opn * env ** 0.8
    bow = (1 - 0.3 * np.exp(-((t - 0.5) / 0.07) ** 2))
    return [_poly(inside, _band(P, nrm, bot, top)),
            _poly(teeth, _band(P, nrm, top - th * env ** 0.6, top)),
            _poly(lip, _band(P, nrm, top, top + up * env ** 0.7 * bow)),
            _poly(lip, _band(P, nrm, bot - low * env ** 0.9, bot))]


def chaikin(pts, n):
    for _ in range(n):
        q = np.roll(pts, -1, axis=0)
        pts = np.stack([0.75 * pts + 0.25 * q, 0.25 * pts + 0.75 * q], 1).reshape(-1, 2)
    return pts


def expand_eye(white, iris, skin, u, v, rx, ry, deg=0.0, k_iris=1.05, lid=0.3):
    t = np.radians(deg)
    rot = lambda a, b: (u + a * np.cos(t) - b * np.sin(t), v + a * np.sin(t) + b * np.cos(t))
    out = [_ellipse(skin, u, v, rx + 0.45, ry + 0.45, deg, on=[iris]),      # clear the auto blob
           _ellipse(white, u, v, rx, ry, deg),
           _ellipse(iris, u, v, k_iris * ry, k_iris * ry, 0.0, on=[white])]  # full-height iris
    if lid > 0:
        s = np.radians(np.linspace(185, -5, 24))
        out.append(_stroke(iris, lid, [rot(rx * np.cos(a), ry * np.sin(a)) for a in s]))
    return out


def parse_paint(spec):
    out = _parse_shape(spec)
    mods = dict(m.split("=", 1) for m in spec.strip().split("|")[1:])
    for s in out:
        if "depth" in mods: s["depth"] = float(mods["depth"])
        if "smooth" in mods and s["kind"] == "poly": s["pts"] = chaikin(s["pts"], int(mods["smooth"]))
        if mods.get("refine") == "0": s["norefine"] = True
    return out


def _parse_shape(spec):
    kind, rest = spec.strip().split(":", 1)
    on = None
    mods = rest.split("|"); rest = mods[0]
    for m in mods[1:]:
        if m.startswith("on="):
            on = [x.strip().upper() for x in m[3:].split("+")]
    kv = dict(m.split("=", 1) for m in mods[1:] if "=" in m)
    if kind in ("eye3d", "lips3d", "grow", "shadowfix"):
        return [_parse_3d(kind, rest, kv)]
    hexcol, body = rest.split("@", 1)
    hexcol = hexcol.strip().upper()
    if kind == "ellipse":
        n = [float(x) for x in body.split(",")]
        return [_ellipse(hexcol, n[0], n[1], n[2], n[3], n[4] if len(n) > 4 else 0.0, on)]
    if kind == "stroke":
        parts = body.split(";")
        return [_stroke(hexcol, float(parts[0]), [[float(a) for a in p.split(",")] for p in parts[1:]], on)]
    if kind == "poly":
        return [_poly(hexcol, [[float(a) for a in p.split(",")] for p in body.split(";")], on)]
    if kind == "lips":
        parts = body.split(";"); k = [float(x) for x in parts[0].split(",")]
        out = expand_lips(hexcol, k[0], k[1], [[float(a) for a in p.split(",")] for p in parts[1:]],
                          k[2] if len(k) > 2 else 0.3)
        for s in out: s["on"] = on; s["depth"] = 0.6    # also fill the mouth crease (see apply_features)
        return out
    if kind == "mouth":
        lip, teeth, inside = [x.strip() for x in hexcol.split(",")]
        parts = body.split(";"); k = [float(x) for x in parts[0].split(",")]
        out = expand_mouth(lip, teeth, inside, k[0], k[1], k[2], k[3] if len(k) > 3 else 0.3,
                           [[float(a) for a in p.split(",")] for p in parts[1:]])
        out[1]["on"] = [inside]                       # teeth only inside the opening
        for s in (out[0], out[2], out[3]): s["on"] = on
        for s in out: s["depth"] = 0.6
        return out
    if kind == "eye":
        w, i, s = [x.strip() for x in hexcol.split(",")]
        n = [float(x) for x in body.split(",")]
        return expand_eye(w, i, s, *n)
    raise SystemExit(f"unknown shape {kind!r} (ellipse | stroke | poly | lips | mouth | eye)")


def _parse_3d(kind, rest, kv):
    """sculpt-following shapes (scripts/sculpt_features.py), seeded by one front-view point"""
    if kind == "shadowfix":       # shadowfix:DARK[,SKIN]   dark islands in deep shadow -> the colour around them
        cols = [x.strip().upper() for x in rest.split(",")]
        return dict(kind="shadowfix", hex=cols[0], skin=cols[1] if len(cols) > 1 else None, on=None,
                    max_ao=float(kv.get("ao", 0.25)), min_area=float(kv.get("min", 2.0)),
                    skin_min_area=float(kv.get("skinmin", 20.0)), all=kv.get("all", "1") == "1", norefine=True)
    if kind == "grow":            # grow:DARK>INTO[+INTO]@u0,u1,v0,v1   (window optional)
        cols, _, win = rest.partition("@")
        dark, into = cols.split(">")
        return dict(kind="grow", hex=dark.strip().upper(), on=None, into=[x.strip().upper() for x in into.split("+")],
                    win=[float(x) for x in win.split(",")] if win.strip() else None,
                    skip=[[float(x) for x in q.split(",")] for q in kv["skip"].split(";")] if "skip" in kv else [],
                    d=float(kv.get("d", 1.2)), eps=float(kv.get("eps", 0.5)), norefine=True)
    cols, body = rest.split("@", 1)
    cols = [x.strip().upper() for x in cols.split(",")]
    pts = [[float(a) for a in p.split(",")[:2]] for p in body.split(";")]
    b = body.split(";")[0].split(",")
    u, v = float(b[0]), float(b[1])
    if kind == "eye3d":           # eye3d:WHITE,IRIS,SKIN@u,v[,liner]
        return dict(kind="eye3d", hex=cols[0], white=cols[0], iris=cols[1], skin=cols[2], on=None, u=u, v=v,
                    liner=len(b) > 2 and b[2].strip() in ("liner", "woman", "f"),
                    iris_k=float(kv.get("iris", 1.1)), catch_k=float(kv.get("catch", 0.43)),
                    liner_max=float(kv.get("liner", 1.45)), clear=float(kv.get("clear", 0.6)),
                    dome=kv.get("dome", "0") == "1", catch_min=float(kv.get("catchmin", 2.0)), norefine=True)
    # lips3d:LIP,SKIN@u1,v1;u2,v2;...  the mouth line traced corner to corner (one point = auto-trace from there)
    #   |keep=#TEETH+#MOUTH (labels left alone: the opening of an open mouth)  |w=half width mm
    if len(pts) > 1:
        u, v = np.mean(pts, 0)
    return dict(kind="lips3d", hex=cols[0], skin=cols[1], on=None, u=float(u), v=float(v), line=pts if len(pts) > 1 else None,
                keep=[x.strip().upper() for x in kv["keep"].split("+")] if "keep" in kv else [],
                teeth=kv["teeth"].strip().upper() if "teeth" in kv else None,
                half_w=float(kv["w"]) if "w" in kv else None, norefine=True)


def read_spec(path):
    """One shape per line. Comments: a line starting with '#', or '  #' (two spaces) after a shape --
    a bare '#' can't mark a comment because every colour is written #RRGGBB."""
    shapes = []
    for line in Path(path).read_text().splitlines():
        line = line.split("  #", 1)[0].strip()
        if line and not line.startswith("#"):
            shapes += parse_paint(line)
    return shapes


def inside(sh, u, v, grow=0.0):
    if sh["kind"] == "ellipse":
        t = np.radians(sh["deg"]); du, dv = u - sh["u"], v - sh["v"]
        a = du * np.cos(t) + dv * np.sin(t); b = -du * np.sin(t) + dv * np.cos(t)
        return (a / (sh["rx"] + grow)) ** 2 + (b / (sh["ry"] + grow)) ** 2 <= 1
    if sh["kind"] == "poly":
        from matplotlib.path import Path as MPath
        pts = np.stack([u, v], 1)
        if grow:   # refinement mask only: bounding box is plenty
            lo, hi = sh["pts"].min(0) - grow, sh["pts"].max(0) + grow
            return ((pts >= lo) & (pts <= hi)).all(1)
        return MPath(sh["pts"]).contains_points(pts)
    P = np.stack([u, v], 1); best = np.full(len(u), np.inf)
    for p, q in zip(sh["pts"][:-1], sh["pts"][1:]):
        d = q - p; t = np.clip(((P - p) @ d) / max(d @ d, 1e-12), 0, 1)
        best = np.minimum(best, np.linalg.norm(P - (p + t[:, None] * d), axis=1))
    return best <= sh["w"] / 2 + grow


def outline(sh, n=64):
    if sh["kind"] == "ellipse":
        s = np.linspace(0, 2 * np.pi, n); t = np.radians(sh["deg"])
        a, b = sh["rx"] * np.cos(s), sh["ry"] * np.sin(s)
        return np.stack([sh["u"] + a * np.cos(t) - b * np.sin(t), sh["v"] + a * np.sin(t) + b * np.cos(t)], 1)
    if sh["kind"] == "poly":
        return np.vstack([sh["pts"], sh["pts"][:1]])
    return sh["pts"]


# ---------------------------------------------------------------- geometry

def refine(V, F, lab, mask):
    """One level of red-green refinement: every edge of a masked face is split at its midpoint;
    neighbours that get 1 or 2 split edges are split into 2 or 3 so the mesh stays conforming."""
    E = np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1)          # (n,3,2) edge i = (f[i], f[i+1])
    key = np.sort(E, 2).reshape(-1, 2)
    uniq, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.reshape(-1, 3)
    split = np.zeros(len(uniq), bool); split[inv[mask].ravel()] = True
    mid = -np.ones(len(uniq), int); ns = int(split.sum())
    mid[split] = np.arange(len(V), len(V) + ns)
    V2 = np.vstack([V, (V[uniq[split, 0]] + V[uniq[split, 1]]) / 2])
    M = mid[inv]                                                         # (n,3) midpoint id or -1
    k = (M >= 0).sum(1)
    out_f, out_l = [F[k == 0]], [lab[k == 0]]
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    s = k == 3
    if s.any():
        m0, m1, m2 = M[s, 0], M[s, 1], M[s, 2]
        out_f += [np.stack(t, 1) for t in ((a[s], m0, m2), (m0, b[s], m1), (m2, m1, c[s]), (m0, m1, m2))]
        out_l += [lab[s]] * 4
    for r in range(3):                                                  # rotate so split edges lead
        A, B, C = F[:, r], F[:, (r + 1) % 3], F[:, (r + 2) % 3]
        mAB, mBC = M[:, r], M[:, (r + 1) % 3]
        s1 = (k == 1) & (mAB >= 0)
        out_f += [np.stack((A[s1], mAB[s1], C[s1]), 1), np.stack((mAB[s1], B[s1], C[s1]), 1)]
        out_l += [lab[s1]] * 2
        s2 = (k == 2) & (mAB >= 0) & (mBC >= 0)
        out_f += [np.stack((mAB[s2], B[s2], mBC[s2]), 1), np.stack((A[s2], mAB[s2], mBC[s2]), 1),
                  np.stack((A[s2], mBC[s2], C[s2]), 1)]
        out_l += [lab[s2]] * 3
    return V2, np.vstack(out_f), np.concatenate(out_l)


def visible_faces(mesh, u, v, d, px=0.05, tol=0.25, occ=None):
    """Face is visible if its centroid is within tol mm of the nearest surface at its pixel
    (centroid splat z-buffer, holes closed with a 3x3 min filter). occ = (u, v, d) face centroids of
    other geometry that also blocks the view -- the rest of the figure when painting one split piece."""
    uc, vc, dc = u[mesh.faces].mean(1), v[mesh.faces].mean(1), d[mesh.faces].mean(1)
    ou, ov, od = occ if occ is not None else (uc[:0], vc[:0], dc[:0])
    keep = (ou >= uc.min()) & (ou <= uc.max()) & (ov >= vc.min()) & (ov <= vc.max())
    i = ((uc - uc.min()) / px).astype(int); j = ((vc - vc.min()) / px).astype(int)
    zb = np.full((i.max() + 3, j.max() + 3), np.inf)
    np.minimum.at(zb, (i + 1, j + 1), dc)
    np.minimum.at(zb, (((ou[keep] - uc.min()) / px).astype(int) + 1, ((ov[keep] - vc.min()) / px).astype(int) + 1), od[keep])
    pad = np.pad(zb, 1, constant_values=np.inf)
    zmin = np.min([pad[a:a + zb.shape[0], b:b + zb.shape[1]] for a in range(3) for b in range(3)], 0)
    return dc <= zmin[i + 1, j + 1] + tol


def apply_features(mesh, lab, palette, shapes, view=270.0, refine_levels=0, verbose=True, occluder=None):
    """Library entry point. mesh: trimesh; lab: per-face palette index; palette: list of '#RRGGBB'
    (extended in place). occluder: the whole assembled figure, when mesh is one split piece -- without
    it a piece's hidden surfaces count as visible (e.g. her hair behind the baby's face takes the
    baby's eye paint). Returns (mesh, lab) -- a new mesh if refine_levels > 0."""
    def idx(h):
        h = h.upper()
        if h not in palette:
            palette.append(h)
        return palette.index(h)

    lab = np.asarray(lab).copy()
    k3 = ("eye3d", "lips3d", "grow", "shadowfix")
    shapes3d = [s for s in shapes if s["kind"] in k3]
    shapes = [s for s in shapes if s["kind"] not in k3]
    eyes, lenses = apply_sculpt(mesh, lab, palette, shapes3d, view, idx, verbose, occluder) if shapes3d else ([], [])
    # labels under the liners / lips before they were painted, per unrefined face (to restore outside the edge)
    under = {}
    for e in eyes:
        for f_, p_ in zip(e.liner, e.liner_prev):
            under[int(f_)] = int(p_)
    for ln in lenses:
        for f_, p_ in zip(ln.faces, ln.prev["faces"]):
            under[int(f_)] = int(p_)

    V, F = mesh.vertices, mesh.faces
    parent = np.arange(len(F))
    for _ in range(refine_levels):
        u, v, _ = view_coords(V, view)
        uc, vc = u[F].mean(1), v[F].mean(1)
        m = np.zeros(len(F), bool)
        for s in shapes:
            if not s.get("norefine"):
                m |= inside(s, uc, vc, grow=0.4)
        if eyes or lenses:                # round iris + catchlight outlines, smooth liner and lip edges
            Cn = V[F].mean(1)
            for e in eyes:
                inn = np.isin(parent, e.reg)
                m[inn] |= e.near_circles(Cn[inn])
                if len(e.liner):
                    m |= np.isin(parent, e.liner) | np.isin(parent, e.liner_ring)
            for ln in lenses:
                m |= np.isin(parent, ln.faces) | np.isin(parent, ln.ring)
        V, F, par = refine(V, F, np.arange(len(F)), m)
        parent, lab = parent[par], lab[par]
    if refine_levels:
        mesh = trimesh.Trimesh(V, F, process=False)
        if verbose: print(f"  refined x{refine_levels}: {len(F):,} faces")
        Cm = mesh.triangles_center
        for e in eyes:
            f = np.where(np.isin(parent, e.reg))[0]
            lab[f] = e.paint_points(Cm[f])
            if len(e.liner):                  # re-cut the liner edge on the small triangles
                f = np.where(np.isin(parent, e.liner) | np.isin(parent, e.liner_ring))[0]
                inside = e.liner_contains(Cm[f])
                was = np.array([under.get(int(p_), lab[k]) for p_, k in zip(parent[f], f)])
                lab[f] = np.where(inside, e.black, np.where(np.isin(parent[f], e.liner), was, lab[f]))
        for ln in lenses:
            f = np.where(np.isin(parent, ln.faces) | np.isin(parent, ln.ring))[0]
            inside = ln.contains(Cm[f])
            was = np.array([under.get(int(p_), lab[k]) for p_, k in zip(parent[f], f)])
            lab[f] = np.where(inside, ln.lip, np.where(np.isin(parent[f], ln.faces), was, lab[f]))

    u, v, d = view_coords(mesh.vertices, view)
    uc, vc = u[mesh.faces].mean(1), v[mesh.faces].mean(1)
    an = np.radians(view)
    ny = mesh.face_normals[:, 0] * np.sin(an) + mesh.face_normals[:, 1] * np.cos(an)
    occ = None
    if occluder is not None:
        ou, ov, od = view_coords(occluder.vertices, view)
        occ = (ou[occluder.faces].mean(1), ov[occluder.faces].mean(1), od[occluder.faces].mean(1))
    vis = visible_faces(mesh, u, v, d, occ=occ) & (ny < -0.05)
    deep = {}
    for s in shapes:
        if s.get("depth"):     # concave detail (mouth crease, beard edge): any face near the front surface
            if s["depth"] not in deep:
                deep[s["depth"]] = visible_faces(mesh, u, v, d, tol=s["depth"], occ=occ) & (ny < 0.5)
            sel = deep[s["depth"]] & inside(s, uc, vc)
        else:
            sel = vis & inside(s, uc, vc)
        if s["on"]:
            sel &= np.isin(lab, [idx(h) for h in s["on"]])
        lab[sel] = idx(s["hex"])
        if verbose:
            where = f"@{s['u']:.2f},{s['v']:.2f}" if s["kind"] == "ellipse" else f"{len(s['pts'])} pts"
            print(f"  {s['kind']} {s['hex']} {where}: {int(sel.sum()):,} faces")
    return mesh, lab


def apply_sculpt(mesh, lab, palette, shapes3d, view, idx, verbose=True, occluder=None):
    """eye3d / lips3d / grow shapes, in order, on the unrefined mesh (they follow the sculpt, see
    sculpt_features.py). Edits lab in place; returns the Eye objects so refinement can re-round the irises."""
    from scipy.spatial import cKDTree
    from sculpt_features import Eye, face_graph, fix_shadow_islands, grow_to_crease, lips_region, seed_from_uv, view_dir
    u, v, d = view_coords(mesh.vertices, view)
    an = np.radians(view)
    ny = mesh.face_normals[:, 0] * np.sin(an) + mesh.face_normals[:, 1] * np.cos(an)
    occ = None
    if occluder is not None:
        ou, ov, od = view_coords(occluder.vertices, view)
        occ = (ou[occluder.faces].mean(1), ov[occluder.faces].mean(1), od[occluder.faces].mean(1))
    vis = visible_faces(mesh, u, v, d, occ=occ) & (ny < -0.05)
    look = view_dir(view)
    G, tree = face_graph(mesh), cKDTree(mesh.triangles_center)
    # build every eye first: a liner needs to know which corner is the outer one (away from the other eye)
    built = {}
    for k, s in enumerate(shapes3d):
        if s["kind"] == "eye3d":
            seed = seed_from_uv(mesh, s["u"], s["v"], view, vis)
            if seed is None:
                continue                      # this piece doesn't contain that eye
            built[k] = Eye(mesh, seed, idx(s["white"]), idx(s["iris"]), idx(s["skin"]), look, liner=s["liner"],
                           iris_k=s["iris_k"], catch_k=s["catch_k"], liner_max=s["liner_max"], clear=s["clear"],
                           dome=s["dome"], catch_min=s["catch_min"], G=G, tree=tree)
    ok_eyes = [e for e in built.values() if e.ok]
    for e in ok_eyes:
        others = [o for o in ok_eyes if o is not e]
        pc = e._C[e.reg].mean(0)
        near = min(others, key=lambda o: np.linalg.norm(o._C[o.reg].mean(0) - pc), default=None)
        same_face = near is not None and np.linalg.norm(near._C[near.reg].mean(0) - pc) < 3.5 * max(e.w, near.w)
        e.make_liner(int(np.sign(e.u_screen - near.u_screen)) if same_face else 0, lab=lab)
    # grow must not touch the eyes: iris black would spread onto the lids (a liner on a man), and a woman's
    # liner is already drawn to the lid fold by the eye itself
    no_grow = np.zeros(len(mesh.faces), bool)
    for e in ok_eyes:
        # keep grow off the lid: men get no liner at all, women's liner is drawn by the eye itself (and hand-painted
        # references leave skin between the liner and the brow)
        Tmax = float(e.liner_T.max()) if len(e.liner) and hasattr(e, "liner_T") else 0.0
        r_ = Tmax + 1.0 if e.want_liner else 1.6
        d_, _ = cKDTree(mesh.triangles_center[e.reg]).query(mesh.triangles_center, distance_upper_bound=r_)
        no_grow |= np.isfinite(d_)
    eyes, lenses = [], []
    for k, s in enumerate(shapes3d):
        if s["kind"] == "shadowfix":
            light = None
            if s["all"]:
                from colorize_3mf import srgb_to_lab
                light = srgb_to_lab(np.array([hex_rgb(h) for h in palette], float))[:, 0]
            lab[:] = fix_shadow_islands(mesh, lab, idx(s["hex"]), idx(s["skin"]) if s["skin"] else None,
                                        max_ao=s["max_ao"], min_area=s["min_area"], skin_min_area=s["skin_min_area"],
                                        verbose=verbose, light=light)
            continue
        if s["kind"] == "grow":
            region = ~no_grow
            uc, vc = u[mesh.faces].mean(1), v[mesh.faces].mean(1)
            if s["win"]:
                u0, u1, v0, v1 = s["win"]
                region &= (uc >= u0) & (uc <= u1) & (vc >= v0) & (vc <= v1)
            for su, sv, sr in s["skip"]:             # eyes kept as the texture painted them
                region &= np.hypot(uc - su, vc - sv) > sr
            new, _ = grow_to_crease(mesh, lab, idx(s["hex"]), [idx(h) for h in s["into"]], region,
                                    d_out=s["d"], eps=s["eps"], verbose=verbose)
            lab[:] = new
            continue
        seed = seed_from_uv(mesh, s["u"], s["v"], view, vis)
        if seed is None:
            if verbose:
                print(f"  {s['kind']} @{s['u']:.2f},{s['v']:.2f}: not on this mesh -- skipped")
            continue
        if s["kind"] == "eye3d":
            e = built[k]
            if not e.ok:
                print(f"  eye3d @{s['u']:.2f},{s['v']:.2f}: NO sculpted eyeball there (lens in front of it, or the eye is "
                      f"only painted, or the seed sits on a raised iris dome: try |dome=1) -- skipped; use the 2D eye: shape")
                continue
            e.apply(lab)
            eyes.append(e)
            if verbose:
                print(f"  eye3d @{s['u']:.2f},{s['v']:.2f}: opening {e.area:.1f} mm2 ({e.w:.1f} x {e.h:.1f} mm), "
                      f"eyeball R {e.R:.1f}, iris d {2 * e.r_iris:.2f}, liner {len(e.liner):,} faces")
        else:
            line = None
            if s["line"]:   # densify the traced polyline to ~0.2 mm and drop it onto the visible surface
                P = np.asarray(s["line"], float)
                seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
                t = np.linspace(0, seg[-1], max(int(seg[-1] / 0.2), 2))
                P = np.c_[np.interp(t, seg, P[:, 0]), np.interp(t, seg, P[:, 1])]
                line = [seed_from_uv(mesh, a_, b_, view, vis) for a_, b_ in P]
                line = np.array([q for q in line if q is not None])
                if len(line) < 3:
                    continue
            # lip-like colours already on the mouth (an old rose lip slot, the texture's red) get repainted, so the
            # new lips are one colour instead of two
            from colorize_3mf import srgb_to_lab
            labp = srgb_to_lab(np.array([hex_rgb(h) for h in palette], float))
            hue = np.degrees(np.arctan2(labp[:, 2], labp[:, 1])) % 360
            reds = [k for k in range(len(palette)) if np.hypot(*labp[k, 1:]) > 25 and (hue[k] < 40 or hue[k] > 330)]
            keep = [idx(h) for h in s["keep"]]
            rep = [k for k in set(reds) | {idx(s["hex"])} if k != idx(s["skin"]) and k not in keep]
            f, teeth, lens = lips_region(mesh, lab, seed, look, idx(s["skin"]), keep_out=keep,
                                         half_w=s["half_w"], tree=tree, verbose=verbose, line_pts=line, repaint=rep)
            if lens is not None:
                lens.lip = idx(s["hex"])
                lens.prev = {"faces": lab[lens.faces].copy(), "ring": lab[lens.ring].copy()}
                lenses.append(lens)
            lab[f] = idx(s["hex"])
            if s["teeth"]:
                lab[teeth] = idx(s["teeth"])
    return eyes, lenses


def preview(mesh, lab, palette, shapes, view, window, ppmm, path, grid=True):
    pal_rgb = np.array([hex_rgb(h) for h in palette])
    u, v, d = view_coords(mesh.vertices, view)
    an = np.radians(view)
    ny = mesh.face_normals[:, 0] * np.sin(an) + mesh.face_normals[:, 1] * np.cos(an)
    u0, u1, v0, v1 = window
    k = ppmm; W, H = int((u1 - u0) * k), int((v1 - v0) * k)
    img = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(img)
    F = mesh.faces
    fu, fv, fd = u[F], v[F], d[F].mean(1)
    win = (fu.max(1) > u0) & (fu.min(1) < u1) & (fv.max(1) > v0) & (fv.min(1) < v1) & (ny < 0)
    light = np.clip(-ny, 0, 1) * 0.45 + 0.55
    for f in np.where(win)[0][np.argsort(-fd[win])]:
        dr.polygon([((fu[f, q] - u0) * k, (v1 - fv[f, q]) * k) for q in range(3)],
                   fill=tuple(int(x) for x in np.clip(pal_rgb[lab[f]] * light[f], 0, 255)))
    if grid:
        for g in np.arange(np.ceil(u0), u1, 1.0):
            x = (g - u0) * k; dr.line([(x, 0), (x, H)], fill=(255, 0, 0) if g % 5 == 0 else (255, 170, 170), width=1)
            if g % 5 == 0: dr.text((x + 2, 2), f"{g:g}", fill=(200, 0, 0))
        for g in np.arange(np.ceil(v0), v1, 1.0):
            y = (v1 - g) * k; dr.line([(0, y), (W, y)], fill=(0, 0, 255) if g % 5 == 0 else (170, 170, 255), width=1)
            if g % 5 == 0: dr.text((2, y + 2), f"{g:g}", fill=(0, 0, 200))
        for s in shapes:
            if s["kind"] in ("eye3d", "lips3d"):
                x, y = (s["u"] - u0) * k, (v1 - s["v"]) * k
                dr.line([(x - 6, y), (x + 6, y)], fill=(0, 200, 0)); dr.line([(x, y - 6), (x, y + 6)], fill=(0, 200, 0))
                continue
            if s["kind"] in ("grow", "shadowfix"):
                continue
            o = outline(s); dr.line([((p[0] - u0) * k, (v1 - p[1]) * k) for p in o], fill=(0, 200, 0), width=1)
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="colorgroup or paint_color 3MF")
    ap.add_argument("-o", "--output", help="colorgroup 3MF to write (omit for preview only)")
    ap.add_argument("--view", type=float, default=270.0, help="azimuth in render_3mf convention (default 270)")
    ap.add_argument("--remap", action="append", default=[], help='"#OLD>#NEW", applied before painting')
    ap.add_argument("--paint", action="append", default=[], help="shape spec, see top of file; applied in order")
    ap.add_argument("--spec", action="append", default=[], help="file of shape specs, one per line (applied before --paint)")
    ap.add_argument("--refine", type=int, default=0, help="refinement levels under the shapes (2 recommended)")
    ap.add_argument("--preview", help="PNG of --window with 1 mm grid and shape outlines")
    ap.add_argument("--no-grid", action="store_true", help="preview without grid/outlines")
    ap.add_argument("--window", help="u0,u1,v0,v1 in mm for --preview")
    ap.add_argument("--ppmm", type=float, default=40, help="preview pixels per mm")
    ap.add_argument("--auto-face", action="store_true",
                    help="find every face (sculpted eyes, mouth line) and paint it from the sculpt: shadowfix, eye3d, "
                         "grow, lips3d for --women. The generated lines are saved as <output>.features.txt -- "
                         "read them, fix any by hand and re-run with --spec. Applied before --spec/--paint shapes")
    ap.add_argument("--women", default="", help="--auto-face: face numbers (left to right in the view) that get "
                                                "liner + lips, e.g. 0,2. Run once without it to see the numbering")
    ap.add_argument("--lip", default="#CE0000", help="--auto-face lip filament (default #CE0000)")
    ap.add_argument("--skin", help="skin hex for --auto-face (auto-detected)")
    ap.add_argument("--white", help="eye-white hex for --auto-face (auto: the palette's near-white, else #F0ECE6 is added)")
    a = ap.parse_args()

    mesh, rgb, pal = load_colored_3mf(Path(a.input))
    palette = ["#%02X%02X%02X" % tuple(int(c) for c in p) for p in pal]
    lab = np.argmin(((rgb[:, None] - pal[None]) ** 2).sum(2), 1)
    for r in a.remap:
        old, new = [x.strip().upper() for x in r.split(">")]
        for h in (old, new):
            if h not in palette: palette.append(h)
        o, n = palette.index(old), palette.index(new)
        print(f"  remap {old} -> {new}: {int((lab == o).sum()):,} faces"); lab[lab == o] = n

    shapes = [s for f in a.spec for s in read_spec(f)] + [s for p in a.paint for s in parse_paint(p)]
    if a.auto_face:
        from sculpt_features import skin_index, suggest_features
        skin = palette.index(a.skin.upper()) if a.skin else skin_index(palette)
        if skin is None:
            raise SystemExit("--auto-face: no skin colour recognised in the palette -- pass --skin #HEX")
        lum = [sum(int(h[k:k + 2], 16) for k in (1, 3, 5)) for h in palette]
        women = [int(x) for x in a.women.split(",") if x.strip()]
        white = palette.index(a.white.upper()) if a.white else None
        lines = [f"shadowfix:{palette[int(np.argmin(lum))]},{palette[skin]}"] + \
            suggest_features(mesh, lab, palette, a.view, white=white, skin=skin, women=women, lip_hex=a.lip.upper())
        print("\n".join("  | " + l for l in lines))
        if a.output:
            Path(a.output).with_suffix(".features.txt").write_text("\n".join(lines) + "\n")
        shapes = [s for l in lines if l.strip() and not l.startswith("#") for s in parse_paint(l)] + shapes
    mesh, lab = apply_features(mesh, lab, palette, shapes, a.view, a.refine)

    if a.preview:
        preview(mesh, lab, palette, shapes, a.view, [float(x) for x in a.window.split(",")], a.ppmm, a.preview,
                grid=not a.no_grid)
        print(f"  preview -> {a.preview}")
    if a.output:
        out = Path(a.output); pal_rgb = np.array([hex_rgb(h) for h in palette])
        write_3mf(mesh, lab, pal_rgb, out, encoding="colorgroup")
        write_obj_mtl(mesh, lab, pal_rgb, out.with_suffix(".obj"))
        A = np.bincount(lab, weights=mesh.area_faces, minlength=len(palette))
        print(f"  {len(mesh.faces):,} faces, watertight={mesh.is_watertight}")
        print("Area per colour: " + "  ".join(f"{h} {x / A.sum() * 100:.1f}%" for h, x in zip(palette, A) if x > 0))


if __name__ == "__main__":
    main()
