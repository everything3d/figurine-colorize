#!/usr/bin/env python3
"""Sculpt-following face paint: eyes, lips and dark features placed from the GEOMETRY, not the blurry texture.

Why: auto-coloured figurines usually get repainted by hand in the slicer before printing. Diffing hand-finished
3MFs against the auto-coloured ones (a dozen-plus figures) showed the same fixes every time,
and each one is a sculpted shape that the texture-driven colouring ignores:
  * eyes: the white is exactly the visible eyeball. Tripo sculpts the eyeball as a sphere sitting behind the
    lids, and the lid rim around it is a sharp concave crease. Our old `eye:` ellipse was drawn in 2D and
    never matched that rim.
  * women get a thick black liner: the whole upper lid, from the eyeball rim up to the lid-fold crease
    (~1.4 mm at 120 mm). Men get no lid line at all (hand-painters delete it).
  * iris ~1.1x the opening height, clipped by the lids, with a white catchlight 0.43x its size, up and right.
  * brows / beards / moustaches / liners: black must cover the whole raised feature up to the concave crease at
    its foot. The texture is lighter on their lit top surfaces, so the colouring stopped short ("missing colour
    at the top").
  * women's lips: red over exactly the two sculpted lip bulges, from the mouth crease out to the concave valleys
    above (philtrum) and below (chin fold).

All of it is one tool: a 2-label min cut on the face graph whose edge cost is length * exp(-(concavity/k0)^2),
so a boundary along a concave crease is nearly free. Seeds say what must be inside / outside; the cut finds the
sculpted outline between them.

Library (used by paint_features.py shape kinds eye3d / lips3d / grow, see there), or CLI to suggest seeds:
    python3 scripts/sculpt_features.py <colored>.3mf --view 270 --detect [--preview faces.png]
prints eye3d / lips3d lines (front-view u,v mm) for features.txt.
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree  # pyright: ignore[reportAttributeAccessIssue]


# ---------------------------------------------------------------- geometry

def submesh(m, faces):
    """(trimesh of the given faces, vertex ids) -- vertices re-indexed, face order kept"""
    F = m.faces[faces]
    u, inv = np.unique(F, return_inverse=True)
    return trimesh.Trimesh(m.vertices[u], inv.reshape(-1, 3), process=False), u


def _vertex_adjacency(F, n):
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.vstack([E, E[:, ::-1]])
    A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A


def smooth_vertex_normals(m, iters=3):
    A = _vertex_adjacency(m.faces, len(m.vertices)) + sp.eye(len(m.vertices))
    N = np.asarray(m.vertex_normals, float).copy()
    for _ in range(iters):
        N = A @ N
        N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    return N


def edge_curvature(m, iters=3):
    """normal curvature (1/mm) across every face-adjacency edge, from normals smoothed over `iters` rings:
    + convex, - concave. Per-edge dihedrals are useless on Tripo's 0.18 mm triangles (a real crease is
    spread over several edges); the smoothed normal field sees it."""
    N = smooth_vertex_normals(m, iters)
    Nf = N[m.faces].mean(1)
    Nf /= np.linalg.norm(Nf, axis=1, keepdims=True) + 1e-12
    C = m.triangles_center
    ei, ej = m.face_adjacency.T
    d = C[ej] - C[ei]
    return ((Nf[ej] - Nf[ei]) * d).sum(1) / ((d * d).sum(1) + 1e-12)


def face_mean_curvature(m, iters=6):
    """per-face mean curvature (1/mm, + convex) from smoothed vertex normals"""
    N = smooth_vertex_normals(m, iters)
    V, F = m.vertices, m.faces
    E = np.stack([V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]], 2)
    D = np.stack([N[F[:, 1]] - N[F[:, 0]], N[F[:, 2]] - N[F[:, 0]]], 2)
    EtE = np.einsum("nik,nil->nkl", E, E)
    EtD = np.einsum("nik,nil->nkl", E, D)
    G = np.einsum("nkl,nlm->nkm", EtD, np.linalg.inv(EtE + 1e-12 * np.eye(2)))
    return 0.5 * np.trace(G, axis1=1, axis2=2)


def face_graph(m):
    ei, ej = m.face_adjacency.T
    n = len(m.faces)
    return sp.coo_matrix((np.ones(2 * len(ei)), (np.r_[ei, ej], np.r_[ej, ei])), shape=(n, n)).tocsr()


def crease_cut(m, src, sink, unary=None, k0=1.0, kappa=None, curv_iters=3, smooth=1.0):
    """2-label min cut on the face graph of (sub)mesh m. src/sink: bool masks, hard. unary: optional (n,2)
    [cost if src side, cost if sink side]. Boundary cost = edge length * exp(-(max(0,-kappa)/k0)^2).
    Returns bool mask of the src side."""
    import maxflow
    if len(m.faces) == 0:                      # seed with no surface in reach (e.g. between two figures)
        return np.zeros(0, bool)
    if kappa is None:
        kappa = edge_curvature(m, curv_iters)
    ei, ej = m.face_adjacency.T
    ev = m.face_adjacency_edges
    L = np.linalg.norm(m.vertices[ev[:, 0]] - m.vertices[ev[:, 1]], axis=1)
    w = smooth * L * np.exp(-(np.maximum(0, -kappa) / k0) ** 2) + 1e-7
    n = len(m.faces)
    g = maxflow.Graph[float](n, len(ei))
    g.add_nodes(n)
    g.add_edges(ei, ej, w, w)
    s_cap = np.where(src, 1e6, 0.0)
    t_cap = np.where(sink, 1e6, 0.0)
    if unary is not None:
        s_cap = s_cap + unary[:, 1]
        t_cap = t_cap + unary[:, 0]
    g.add_grid_tedges(np.arange(n), s_cap, t_cap)
    g.maxflow()
    return ~g.get_grid_segments(np.arange(n))


# ---------------------------------------------------------------- eyes

def fit_sphere(P, w=None, N=None):
    """least-squares sphere. With normals N, solve c + R n_i = p_i (stable on small or coarse patches, where the
    purely algebraic fit degenerates); otherwise the algebraic fit."""
    if N is not None:
        n = len(P)
        A = np.zeros((3 * n, 4))
        A[:, :3] = np.tile(np.eye(3), (n, 1))
        A[:, 3] = N.reshape(-1)
        b = P.reshape(-1)
        if w is not None:
            ww = np.repeat(w, 3)
            A, b = A * ww[:, None], b * ww
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        return x[:3], float(abs(x[3]))
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    if w is not None:
        A, b = A * w[:, None], b * w
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    c = x[:3]
    return c, float(np.sqrt(max(x[3] + c @ c, 1e-9)))


def _flood(G, ok, f0):
    seen = np.zeros(len(ok), bool)
    seen[f0] = True
    q, out = deque([f0]), []
    ip, ix = G.indptr, G.indices
    while q:
        f = q.popleft()
        out.append(f)
        for g in ix[ip[f]:ip[f + 1]]:
            if ok[g] and not seen[g]:
                seen[g] = True
                q.append(g)
    return np.array(out)


def grow_sphere(m, seed, G=None, tree=None, tol=0.035, search=1.5, max_r=12.0):
    """The sculpted eyeball is a sphere (hand-painted eye whites fit one to +-0.03 mm). Try sphere fits on
    small patches round the seed (a patch on the lid, the rim or a raised iris dome gives a wrong sphere), flood
    the faces lying on each, keep the one with the largest plausible core. Returns (faces, centre, radius)."""
    C, N = m.triangles_center, m.face_normals
    tree = tree if tree is not None else cKDTree(C)
    G = G if G is not None else face_graph(m)
    seed = np.asarray(seed, float)
    near = np.array(tree.query_ball_point(seed, search), dtype=int)
    if not len(near):
        return np.array([], int), seed, 0.0
    rng = np.random.default_rng(0)
    starts = [tree.query(seed)[1]] + list(rng.choice(near, min(10, len(near)), replace=False))
    best = (0.0, np.array([], int), seed, 0.0)
    dseed = np.linalg.norm(C - seed, axis=1)
    for f0 in starts:
        for r in (0.7, 1.0):
            reg = np.array(tree.query_ball_point(C[f0], r), dtype=int)
            reg = reg[(N[reg] @ N[f0]) > 0.5]
            if len(reg) < 12:
                continue
            c, R = fit_sphere(C[reg])
            if not 1.5 <= R <= 25:
                continue
            for _ in range(3):
                rad = C - c
                dist = np.linalg.norm(rad, axis=1)
                tl = tol * max(1.0, R / 4.0)           # sculpt deviations grow with the figure
                ok = (np.abs(dist - R) < tl) & ((N * rad).sum(1) / (dist + 1e-12) > 0.85) & (dseed < max_r)
                if not ok[f0]:
                    break
                reg = _flood(G, ok, f0)
                if len(reg) < 12:
                    break
                c, R = fit_sphere(C[reg], np.sqrt(m.area_faces[reg]))
                if not 1.5 <= R <= 25:
                    break
            else:
                area = m.area_faces[reg].sum()
                if area > best[0]:
                    best = (area, reg, c, R)
            break
    return best[1], best[2], best[3]


def eye_opening(m, seed, G=None, tree=None, reach=4.5, with_dome=False, dome=False):
    """faces of the visible eyeball around seed: a min cut to the concave rim crease where the lids meet it.
    When the eyeball fits a sphere convincingly, the sphere also bounds and steers the cut (needed on coarse
    meshes, where the rim crease reads weak). Some sculpts carry a raised iris dome with its own crease: the
    first cut then returns just the dome, so step across it to the rim and report the dome as the iris.
    Returns (faces, centre, R) or, with_dome, (faces, centre, R, dome faces)."""
    seed = np.asarray(seed, float)
    core, c, R = grow_sphere(m, seed, G=G, tree=tree)
    C = m.triangles_center
    if len(core) and 1.5 <= R <= 25:
        # big figures have big eyes: the cut window must hold the whole opening (the visible part of the
        # eyeball reaches ~1.2 R from the core's centre)
        reach = max(reach, 1.3 * R + 0.5, 1.4 * float(np.linalg.norm(C[core] - C[core].mean(0), axis=1).max()))
    fsel = np.where(np.linalg.norm(C - seed, axis=1) < reach + 0.5)[0]
    sm, _ = submesh(m, fsel)
    Cs = sm.triangles_center
    ds = np.linalg.norm(Cs - seed, axis=1)
    incore = np.isin(fsel, core)
    sphere = incore.sum() >= 10 and 1.5 <= R <= 25 and m.area_faces[core].sum() > 4.0
    un = None
    if sphere:
        sR = max(1.0, R / 4.0)                     # tolerances scale with the eyeball (big figures, big eyes)
        res = np.linalg.norm(Cs - c, axis=1) - R
        cc = Cs[incore].mean(0)
        rc = np.percentile(np.linalg.norm(Cs[incore] - cc, axis=1), 90)
        src = incore & (np.linalg.norm(Cs - cc, axis=1) < max(0.8, 0.6 * rc))
        sink = (np.abs(res) > 0.3 * sR) | (ds > reach)
        a = sm.area_faces
        un = np.zeros((len(Cs), 2))
        un[:, 1] = np.where(np.abs(res) < 0.04 * sR, 2.0 * a, 0.0)
        un[:, 0] = np.where(np.abs(res) > 0.12 * sR, 2.0 * a, 0.0)
    else:
        src = ds < 0.6
        sink = ds > reach
    kap = edge_curvature(sm, 3)
    O = crease_cut(sm, src, sink, unary=un, kappa=kap)
    dm = np.zeros(len(Cs), bool)
    if dome and sm.area_faces[O].sum() < 4.0:        # seed on a raised iris dome: go out across its base crease
        dm = O.copy()
        dd, _ = cKDTree(Cs[O]).query(Cs)
        O = crease_cut(sm, dd < 0.35, sink, unary=un, kappa=kap)
        dm &= O
    if sphere and O.sum() > 12:
        O |= _eye_corners(sm, O, c, R, kap, ds)
    if not sphere:
        c, R = fit_sphere(Cs[O]) if O.sum() > 12 else (seed, 0.0)
    if with_dome:
        return fsel[O], c, R, fsel[dm]
    return fsel[O], c, R


def _eye_corners(sm, O, c, R, kap, ds, res_tol=0.3, dmax=1.2, xfrac=0.5, block=1.5):
    """The eyeball tucks in at the corners (0.2 mm off the sphere), so the cut stops short of them and hand-painted
    white runs on into the corner (1.5-2.6 mm2 per eye). Add faces beside the opening, at its two ends only,
    still roughly on the sphere and facing out, reachable without crossing the lid-rim crease. 89-93% of what this
    adds is white in hand-painted files; on an eye that is already complete it adds ~nothing."""
    from scipy.sparse.csgraph import connected_components
    Cs, N = sm.triangles_center, sm.face_normals
    sR = max(1.0, R / 4.0)
    look = -(Cs[O].mean(0) - c)
    to2 = eye_chart(c, Cs[O], look / (np.linalg.norm(look) + 1e-12))
    P = to2(Cs)
    w = sm.area_faces[O]
    oc = (P[O] * w[:, None]).sum(0) / w.sum()
    hw = np.ptp(P[O][:, 0]) / 2
    d = cKDTree(Cs[O]).query(Cs)[0]
    rad = Cs - c
    dist = np.linalg.norm(rad, axis=1)
    cand = ~O & (d < dmax) & (np.abs(dist - R) < res_tol * sR) & ((N * rad).sum(1) / (dist + 1e-12) > 0.6) & \
        (np.abs(P[:, 0] - oc[0]) > xfrac * hw)
    ei, ej = sm.face_adjacency.T
    ok = (O | cand)[ei] & (O | cand)[ej] & (kap > -block)
    n = len(Cs)
    _, comp = connected_components(sp.coo_matrix((np.ones(ok.sum()), (ei[ok], ej[ok])), shape=(n, n)), directed=False)
    return cand & np.isin(comp, np.unique(comp[O]))


def view_dir(az):
    """direction the camera looks in, for render_3mf / paint_features azimuth az"""
    a = np.radians(az)
    return np.array([np.sin(a), np.cos(a), 0.0])


def eye_chart(c, pts, look):
    """2D chart (mm) of the eyeball round the opening: u = screen right, v = up, as seen from the view."""
    g = pts.mean(0) - c
    g /= np.linalg.norm(g)
    up = np.array([0, 0, 1.0])
    u = np.cross(up, g)
    u /= np.linalg.norm(u) + 1e-12
    if np.dot(np.cross(look, up), u) < 0:
        u = -u
    v = np.cross(g, u)
    if v[2] < 0:
        v = -v
    return lambda P: np.stack([(P - c) @ u, (P - c) @ v], 1)


LINER_ABSORB, LINER_ABSORB_ITERS = 0.6, 3     # how far (mm) / how many times the liner grows through old lash paint
LINER_PCT = 75                                # band reach per arc bin: this percentile of the lid flood
LINER_PROFILE = (0.18, 0.30, 0.42)            # minimum band, x opening height, inner corner / mid / outer corner


class Eye:
    """A painted eye. `apply(lab)` paints the unrefined mesh; `circles(chart_pts)` re-evaluates iris and
    catchlight on refined faces so their outlines come out round."""

    def __init__(self, m, seed, white, black, skin, look, liner=False, iris_k=1.1, catch_k=0.43,
                 catch_off=(0.2, 0.18), iris_up=0.08, liner_max=1.45, arc_lo=-12.0, crease_block=2.0,
                 clear=0.6, dome=False, catch_min=2.0, G=None, tree=None):
        self.white, self.black, self.skin = white, black, skin
        self.reg, self.c, self.R, self.dome = eye_opening(m, seed, G=G, tree=tree, with_dome=True, dome=dome)
        self.ok = len(self.reg) >= 20
        if not self.ok:
            return
        P0 = eye_chart(self.c, m.triangles_center[self.reg], look)(m.triangles_center[self.reg])
        # a real opening is a few mm across; anything tiny means the seed is not on a sculpted eyeball
        # (painted-on eye, or a lens surface in front of it) -> caller falls back to the 2D eye: shape
        if m.area_faces[self.reg].sum() < 0.8 or np.ptp(P0[:, 0]) < 1.0 or np.ptp(P0[:, 1]) < 0.4:
            self.ok = False
            return
        C, ar = m.triangles_center, m.area_faces
        self._C = C
        self.to2 = eye_chart(self.c, C[self.reg], look)
        P = self.to2(C[self.reg])
        w = ar[self.reg]
        self.w, self.h = float(np.ptp(P[:, 0])), float(np.ptp(P[:, 1]))
        self.oc = (P * w[:, None]).sum(0) / w.sum()
        self.r_iris = min(iris_k * self.h, 0.85 * self.w) / 2
        self.ic = self.oc + np.r_[0.0, iris_up * self.h]
        if len(self.dome) and ar[self.dome].sum() > 0.6:       # the sculpt has its own iris: use it
            Pd = self.to2(C[self.dome])
            self.ic = (Pd * ar[self.dome][:, None]).sum(0) / ar[self.dome].sum()
            self.r_iris = float(np.sqrt(ar[self.dome].sum() / np.pi)) * 1.05
        self.cc = self.ic + np.asarray(catch_off) * self.r_iris
        # catchlight: one white disc, up and to the viewer's right, same in both eyes. Small realistic eyes
        # (iris under catch_min mm) get none -- hand-painters paint those solid
        self.r_catch = catch_k * self.r_iris if 2 * self.r_iris >= catch_min else 0.0
        self.area = float(w.sum())
        # faces round the opening: old eye paint within `clear` mm goes back to skin
        inO = np.zeros(len(C), bool)
        inO[self.reg] = True
        tr = tree if tree is not None else cKDTree(C)
        nb = np.array(tr.query_ball_point(C[self.reg].mean(0), max(self.w, self.h) * 0.75 + max(liner_max, 0.5 * self.h) + 1.6), dtype=int)
        nb = nb[~inO[nb]]
        d, _ = cKDTree(C[self.reg]).query(C[nb])
        # old eye paint round the opening goes back to skin: black (lid lines) within `clear`, white (an old
        # ellipse spilling onto the lids) a little further
        self.clear = nb[d < clear]
        self.eye_cols = None                  # colours the old eye used (set in apply)
        cw = nb[d < max(clear, 1.2 * (clear > 0))]
        # faces just outside the cut that still lie on the eyeball sphere are eyeball the cut missed (a corner):
        # they become white; white on the lids (an old ellipse spilling over) goes back to skin
        rad = C[cw] - self.c
        dist = np.linalg.norm(rad, axis=1)
        on_ball = (np.abs(dist - self.R) < 0.06) & ((m.face_normals[cw] * rad).sum(1) / (dist + 1e-12) > 0.9) \
            if self.R > 1.0 else np.zeros(len(cw), bool)
        self.extra_white = cw[on_ball & (d[d < max(clear, 1.2 * (clear > 0))] < 0.4)]
        self.clear_white = cw[~on_ball]
        self.liner = np.array([], int)
        self.liner_prev = np.array([], int)
        self.liner_ring = np.array([], int)
        self.below_corner = None
        self.want_liner = liner
        self._m, self._nb, self._d = m, nb, d
        self._lp = (liner_max, arc_lo, crease_block)
        self.u_screen = float(C[self.reg].mean(0) @ np.cross(look, [0, 0, 1.0]))

    def make_liner(self, outer=0, lab=None):
        """outer: screen direction of the outer (temple) corner, +1 right / -1 left / 0 unknown.
        lab: current labels -- the old texture lash line is absorbed into the liner instead of leaving a skin gap"""
        if self.ok and self.want_liner:
            self.liner = self._liner(self._m, self._nb, self._d, *self._lp, outer=outer, lab=lab)
        return self

    def _liner(self, m, nb, d, liner_max, arc_lo, crease_block, outer=0, lab=None):
        """women's liner = one solid band over the upper lid, the way it is painted by hand (measured:
        1.1-1.6 mm mid-lid, 1.4-1.95 mm at the outer corner, wing 0.7-2.3 mm, sculpted lashes inside it):
        - reach: flood from the eyeball rim over the lid, stopped by the lid-fold crease;
        - old texture lash / lid line within reach of the rim is absorbed (no skin gap, no double line);
        - never thinner than 0.18 / 0.30 / 0.42 x opening height at the inner corner / mid-lid / outer corner;
        - smoothed along the arc, a wing ~0.28 x eye width past the outer corner, rising.
        The thickness profile is kept (angle -> mm) so refinement can re-cut the edge smoothly."""
        from scipy.sparse.csgraph import dijkstra
        from scipy.ndimage import gaussian_filter1d
        cap = max(liner_max, 0.5 * self.h)
        keep = d < cap + 1.0
        fsel = np.r_[nb[keep], self.reg]
        sm, _ = submesh(m, fsel)
        n1 = int(keep.sum())
        Cs = sm.triangles_center
        Ps = self.to2(Cs)
        isO = np.r_[np.zeros(n1, bool), np.ones(len(self.reg), bool)]
        # band width is measured as seen from the front (eye chart), not in 3D: the lid fold curls inward, so
        # the bottom of the fold is further away in 3D than the lid beyond it, and would drop out as a sliver
        self._open2 = cKDTree(Ps[isO])
        ds = self._open2.query(Ps)[0]
        ds[isO] = 0.0
        th = self._angle(Ps)
        arc = (th > arc_lo) & (th < 180 - arc_lo)
        kap = edge_curvature(sm, 3)
        ei, ej = sm.face_adjacency.T
        ok = (~isO) & arc & (ds < cap)
        start = np.where(ok & (ds < 0.15))[0]
        if not len(start):
            return np.array([], int)
        good = ok[ei] & ok[ej] & ~(kap < -crease_block)
        Lg = np.linalg.norm(Cs[ei] - Cs[ej], axis=1)
        Gr = sp.coo_matrix((Lg[good], (ei[good], ej[good])), shape=(len(Cs), len(Cs))).tocsr()
        reach = np.isfinite(dijkstra(Gr, directed=False, indices=start, min_only=True)) & ok
        nbin = 31
        bins = np.linspace(arc_lo, 180 - arc_lo, nbin)
        b = np.clip(np.digitize(th, bins) - 1, 0, nbin - 2)
        T = np.array([np.percentile(ds[reach & (b == k)], LINER_PCT) if (reach & (b == k)).sum() > 3 else np.nan
                      for k in range(nbin - 1)])
        if np.isnan(T).all():
            T = np.zeros(nbin - 1)
        T = np.where(np.isnan(T), np.nanmedian(T), T)
        if lab is not None:              # the texture's own lash line: take it into the band
            old = lab[fsel] == self.black
            Tb = np.array([np.percentile(ds[ok & old & (b == k)], 95) if (ok & old & (b == k)).sum() > 3 else 0.0
                           for k in range(nbin - 1)])
            T = np.maximum(T, np.minimum(Tb, cap))
        mid = (bins[:-1] + bins[1:]) / 2
        x = (mid - arc_lo) / (180 - 2 * arc_lo)            # 0 = screen-right end, 1 = screen-left end
        pos = x if outer < 0 else (1 - x) if outer > 0 else 0.5 + 0 * x   # 0 = inner corner .. 1 = outer
        prof = np.interp(pos, [0, 0.5, 1], LINER_PROFILE) * self.h
        if not outer:
            prof = np.full_like(mid, LINER_PROFILE[1] * self.h)
        T = np.maximum(T, prof)
        T = np.array([np.median(T[max(0, k - 2):k + 3]) for k in range(len(T))])
        T = gaussian_filter1d(T, 1.5, mode="nearest")
        # thin out into the inner corner (hand-painted liner hooks round the tear duct, it doesn't stop square)
        if outer:
            T *= np.clip(0.45 + pos / 0.18, 0.45, 1.0)
        T = np.clip(T, 0.15, cap)
        if lab is not None:
            # close the gap to the texture's lash line: old black starting within 0.6 mm of the band's edge is
            # taken in (no skin sliver between liner and lashes, the outer-corner lash flicks join the wing end)
            old_all = lab[fsel] == self.black

            def reach_old(T):
                out = np.zeros(nbin - 1)
                for k in range(nbin - 1):
                    nbk = (np.abs(b - k) <= 1) & arc & old_all & (ds > T[k] - 0.05) & (ds <= T[k] + LINER_ABSORB)
                    if nbk.sum() > 3:
                        out[k] = np.percentile(ds[nbk], 97)
                return np.minimum(out, cap + 0.6)

            for _ in range(LINER_ABSORB_ITERS):
                ext = reach_old(T)
                if not (ext > T + 0.02).any():
                    break
                T = np.maximum(T, ext)
            T = gaussian_filter1d(T, 1.0, mode="nearest")
            T = np.maximum(T, reach_old(T))          # smoothing must not reopen a sliver under the lash line
        self.liner_T, self.liner_bins, self.liner_arc = T, bins, (arc_lo, 180 - arc_lo)
        self._open_tree = cKDTree(m.triangles_center[self.reg])
        self._cap3 = cap + 1.0
        L = (~isO) & arc & (ds <= T[b])
        self.below_corner = None
        if outer:
            PO = Ps[isO]
            K = PO[np.argmax(outer * PO[:, 0])]
            # past the outer corner hand-painted liner only goes UP (wing); nothing at or below the corner height
            self.below_corner = (outer, K.copy(), 0.1 * self.w, 0.08 * self.h)
            L &= ~self._below_corner(Ps)
        self.liner_ring = fsel[(~isO) & arc & (ds > T[b]) & (ds <= T[b] + 0.3)]
        self.wing = None
        if outer:
            t_out = T[0] if outer > 0 else T[-1]          # T[0] is the screen-right end
            # hand-painted wing is a flick off the TOP of the outer corner, rising; the corner itself stays white
            # below it (a wing that starts at the corner point paints a black block over the eye corner)
            tip = K + np.r_[outer * 0.28 * self.w, 0.32 * self.h]
            top = K + np.r_[-outer * 0.05 * self.w, t_out]
            base = K + np.r_[-outer * 0.08 * self.w, 0.12 * self.h]
            from matplotlib.path import Path as MPath
            self.wing = MPath(np.array([top, tip, base]))
            self.wing_reach = 0.45 * self.w
            L |= (~isO) & self.wing.contains_points(Ps) & (ds < self.wing_reach)
        # stray old black under the eye (lower lid lines, lash ends in the corners) -> skin; hand-painters never
        # paint the lower lid
        low = (~isO) & ~arc & (ds < 1.2)
        self.lower_clear = fsel[low]
        return fsel[L]

    def _below_corner(self, P):
        if self.below_corner is None:
            return np.zeros(len(P), bool)
        outer, K, dx, dy = self.below_corner
        return (outer * (P[:, 0] - K[0]) > -dx) & (P[:, 1] < K[1] + dy)

    def _angle(self, P):
        q = (P - self.oc) / np.r_[self.w / 2, self.h / 2]
        th = np.degrees(np.arctan2(q[:, 1], q[:, 0]))
        return np.where(th < -90, th + 360, th)

    def liner_contains(self, P3):
        """True where a point (refined face centroid) is inside the liner band or wing"""
        if not len(self.liner) or not hasattr(self, "liner_T"):
            return np.zeros(len(P3), bool)
        P = self.to2(P3)
        th = self._angle(P)
        d, _ = self._open2.query(P)
        d3, _ = self._open_tree.query(P3)
        b = np.clip(np.digitize(th, self.liner_bins) - 1, 0, len(self.liner_T) - 1)
        a0, a1 = self.liner_arc
        near = d3 < self._cap3                   # stay on the lid (not a surface far behind it)
        inside = (th > a0) & (th < a1) & (d <= self.liner_T[b]) & (d > 0.02) & near & ~self._below_corner(P)
        if self.wing is not None:
            inside |= self.wing.contains_points(P) & (d < self.wing_reach) & (d > 0.02) & near
        return inside

    def apply(self, lab):
        if not self.ok:
            return lab
        C = self._C
        # every colour the old eye used (white, black, a brown/blue iris slot) goes; skin and the rest stay
        old = np.unique(lab[self.reg])
        cols = [c for c in old if c != self.skin and (lab[self.reg] == c).sum() > 0.05 * len(self.reg)]
        cols = sorted(set(cols) | {self.white, self.black})
        lab[self.clear[np.isin(lab[self.clear], cols)]] = self.skin
        lab[self.clear_white[lab[self.clear_white] == self.white]] = self.skin
        lab[self.extra_white[np.isin(lab[self.extra_white], [self.white, self.black, self.skin])]] = self.white
        lab[self.reg] = self.white
        P = self.to2(C[self.reg])
        lab[self.reg[np.linalg.norm(P - self.ic, axis=1) <= self.r_iris]] = self.black
        lab[self.reg[np.linalg.norm(P - self.cc, axis=1) <= self.r_catch]] = self.white
        if hasattr(self, "lower_clear"):
            lc = self.lower_clear[lab[self.lower_clear] == self.black]
            lab[lc] = self.skin
        self.liner_prev = lab[self.liner].copy()
        lab[self.liner] = self.black
        return lab

    def near_circles(self, P3, tol=0.25):
        """points within tol mm of the iris or catchlight outline (where refinement pays off)"""
        P = self.to2(P3)
        return (np.abs(np.linalg.norm(P - self.ic, axis=1) - self.r_iris) < tol) | \
               (np.abs(np.linalg.norm(P - self.cc, axis=1) - self.r_catch) < tol)

    def paint_points(self, P3):
        """colour (palette idx) for points on the eyeball: white / iris / catchlight"""
        P = self.to2(P3)
        out = np.full(len(P), self.white)
        out[np.linalg.norm(P - self.ic, axis=1) <= self.r_iris] = self.black
        out[np.linalg.norm(P - self.cc, axis=1) <= self.r_catch] = self.white
        return out


# ---------------------------------------------------------------- dark features

def grow_to_crease(m, lab, grow, into, region=None, d_out=0.8, eps=0.5, k0=1.0, verbose=True):
    """Grow `grow` (palette idx, e.g. black) into `into` (list of idx, e.g. [skin]) up to the concave crease at
    the foot of each sculpted feature (brow, beard, moustache, liner, hairline), within d_out mm. Joining costs
    eps per mm2; a boundary costs its length * exp(-(concavity/k0)^2), so it moves out to a crease when one is
    near and otherwise stays where it is. Never shrinks anything."""
    import maxflow
    lab = lab.copy()
    C = m.triangles_center
    G = lab == grow
    cand = np.isin(lab, into)
    if region is not None:
        G &= region
        cand &= region
    if not G.any() or not cand.any():
        return lab, 0
    d, _ = cKDTree(C[G]).query(C[cand], distance_upper_bound=d_out + 0.3)
    band = np.where(cand)[0][d <= d_out]
    if not len(band):
        return lab, 0
    gnear = np.where(G)[0]
    d2, _ = cKDTree(C[band]).query(C[gnear], distance_upper_bound=0.5)
    gnear = gnear[np.isfinite(d2)]
    sel = np.r_[band, gnear]
    dc, _ = cKDTree(C[sel]).query(C, distance_upper_bound=0.4)
    ctx = np.where(np.isfinite(dc))[0]
    ctx = ctx[~np.isin(ctx, sel)]
    allf = np.r_[sel, ctx]
    sm, _ = submesh(m, allf)
    nb_, ng = len(band), len(gnear)
    isB = np.zeros(len(allf), bool)
    isB[:nb_] = True
    src = np.zeros(len(allf), bool)
    src[nb_:nb_ + ng] = True
    src[nb_ + ng:] = lab[ctx] == grow
    sink = ~isB & ~src
    un = np.zeros((len(allf), 2))
    un[:, 0] = np.where(isB, eps * sm.area_faces, 0.0)
    S = crease_cut(sm, src, sink, unary=un, k0=k0)
    new = isB & S
    lab[allf[new]] = grow
    if verbose:
        print(f"  grow {int(new.sum()):,} faces ({sm.area_faces[new].sum():.1f} mm2) up to sculpted creases")
    return lab, int(new.sum())


# ---------------------------------------------------------------- shadows

def skin_index(palette):
    """the palette colour that looks like skin -- warm hue, moderate chroma, fair to deep -- or None"""
    lab = _srgb_to_lab(np.array([[int(h.lstrip('#')[k:k + 2], 16) for k in (0, 2, 4)] for h in palette], float))
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1])); ch = np.hypot(lab[:, 1], lab[:, 2])
    # skin from fair to deep sits at hue 50-70 deg with a* 7-26 (khaki: a* 5, hue 73; pinks and reds: hue < 40)
    ok = (lab[:, 0] > 25) & (lab[:, 0] < 92) & (ch > 12) & (ch < 60) & (hue > 45) & (hue < 72.5) & (lab[:, 1] > 6)
    if not ok.any():
        return None
    d = np.hypot(lab[:, 1] - 18, lab[:, 2] - 32)
    return int(np.where(ok)[0][np.argmin(d[ok])])


def shadow_islands(mesh, lab, dark, fao, skin=None, max_ao=0.25, min_area=2.0, skin_min_area=20.0, light=None):
    """Islands that are only there because of shadow: a patch of a DARKER colour lying in deep shadow (mean AO <
    max_ao) and bordered mostly (>= 80%) by one lighter colour -- black under drapes, arms and collars, black shoe
    soles, a skin band under a mint hem, skin in a shaded white collar. Hand-painters repainted every such patch on
    the reference set (repaints median AO 0.05-0.3 vs 0.93 for paint they kept). `light` = Lab L per palette index; without
    it only the darkest colour (`dark`) is considered. Skin-bordered dark islands must be >= skin_min_area (nostrils,
    mouth corners and ear holes stay dark); the biggest region of each colour is never touched.
    Returns (face mask, island count, area mm2)."""
    ei, ej = mesh.face_adjacency.T
    ev = mesh.face_adjacency_edges
    L = np.linalg.norm(mesh.vertices[ev[:, 0]] - mesh.vertices[ev[:, 1]], axis=1)
    n = len(mesh.faces)
    ar = mesh.area_faces
    K = int(lab.max()) + 1
    colours = [dark] if light is None else [c for c in range(K) if (lab == c).any()]
    out = np.zeros(n, bool)
    count, area = 0, 0.0
    for c in colours:
        isD = lab == c
        same = isD[ei] & isD[ej]
        nc, comp = connected_components(sp.coo_matrix((np.ones(same.sum()), (ei[same], ej[same])), shape=(n, n)),
                                        directed=False)
        A = np.bincount(comp[isD], weights=ar[isD], minlength=nc)
        aoK = np.bincount(comp[isD], weights=(ar * fao)[isD], minlength=nc) / np.maximum(A, 1e-12)
        bd = isD[ei] ^ isD[ej]
        kc = np.where(isD[ei[bd]], comp[ei[bd]], comp[ej[bd]])
        other = np.where(isD[ei[bd]], lab[ej[bd]], lab[ei[bd]])
        Bm = sp.coo_matrix((L[bd], (kc, other)), shape=(nc, K)).toarray()
        X = Bm.argmax(1)
        share = Bm.max(1) / np.maximum(Bm.sum(1), 1e-12)
        big = A.max() if len(A) else 0
        pick = (A >= min_area) & (aoK < max_ao) & (A < 0.25 * A.sum()) & (A < big) & (Bm.sum(1) > 0)
        if light is not None and c != dark:      # mid tones: only inside ONE lighter colour (the black rule stays as tuned)
            pick &= (share >= 0.8) & (np.asarray(light)[X] > light[c] + 5)
        if skin is not None and c != skin:
            pick &= (X != skin) | (A >= skin_min_area)
        # a sandwich is design, not shadow: a dark sole under a white midsole band whose other side is the same
        # dark again (a black outsole that hand-painters kept). Exempt an island whose surrounding
        # lighter region is small (< 10x the island) and borders this colour elsewhere at least half as long.
        for k in np.where(pick)[0]:
            xk = X[k]
            isX = lab == xk
            sx = isX[ei] & isX[ej]
            _, compX = connected_components(sp.coo_matrix((np.ones(sx.sum()), (ei[sx], ej[sx])), shape=(n, n)),
                                            directed=False)
            e_isl = bd & (((comp[ei] == k) & isD[ei] & isX[ej]) | ((comp[ej] == k) & isD[ej] & isX[ei]))
            xf = np.where(isD[ei[e_isl]], ej[e_isl], ei[e_isl])
            xc = np.unique(compX[xf])
            inxc = isX & np.isin(compX, xc)
            if ar[inxc].sum() > 10 * A[k]:
                continue
            e_other = (inxc[ei] & isD[ej] & (comp[ej] != k)) | (inxc[ej] & isD[ei] & (comp[ei] != k))
            if L[e_other].sum() >= 0.5 * L[e_isl].sum():
                pick[k] = False
        out |= isD & pick[comp]
        count += int(pick.sum())
        area += float(A[pick].sum())
    return out, count, area


def fix_shadow_islands(mesh, lab, dark, skin=None, ao=None, verbose=True, light=None, **kw):
    """label-only version (no texture): every deep-shadow dark island takes the colour around it"""
    sys.path.insert(0, str(Path(__file__).parent))
    from colorize_3mf import fill_from_surroundings
    if ao is None:
        from render_pretty import bake_ao
        ao = bake_ao(mesh.vertices, mesh.faces, np.asarray(mesh.vertex_normals, np.float64), 32)
    isl, k, area = shadow_islands(mesh, lab, dark, ao[mesh.faces].mean(1), skin, light=light, **kw)
    lab = lab.copy()
    if k:
        fill_from_surroundings(lab, mesh.face_adjacency, isl.copy())
    if verbose:
        print(f"  shadow fix: {k} shadow islands ({area:.1f} mm2) -> colour around them")
    return lab


def _srgb_to_lab(rgb):
    a = rgb / 255.0
    lin = np.where(a > 0.04045, ((a + 0.055) / 1.055) ** 2.4, a / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = (lin @ M.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), (903.3 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


# ---------------------------------------------------------------- lips

def trace_mouth_line(U, V, fk, facing, step=0.25, band=2.0, fade=-0.8):
    """the mouth crease, column by column from the seed (U=V=0) out to both corners: in each 0.25 mm column
    take the most concave face near the previous column's height; stop where the crease fades out."""
    keep = facing & (np.abs(V) < band + 1.0)
    idx = np.where(keep)[0]
    out = []
    for direction in (1, -1):
        v_ref, weak = 0.0, 0
        for c in range(0, 60):
            u0 = direction * c * step
            sel = idx[(np.abs(U[idx] - u0) < step * 0.6) & (np.abs(V[idx] - v_ref) < (band if c == 0 else 0.6))]
            if not len(sel):
                break
            f = sel[np.argmin(fk[sel])]
            if fk[f] > fade:
                weak += 1
                if weak >= 3:
                    break
                continue
            weak = 0
            v_ref = V[f]
            out.append(f)
    if not out:
        return np.array([], int)
    out = np.unique(out)
    # thicken to a connected strip: faces within 0.15 mm of the traced points
    d, _ = cKDTree(np.c_[U[out], V[out]]).query(np.c_[U, V])
    return np.where(keep & (d < 0.15))[0]


def _polyline(P, step=0.1):
    """resample a 2D polyline to ~step spacing; returns points, unit tangents, unit normals (+V side)"""
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.linspace(0, seg[-1], max(int(seg[-1] / step), 2))
    Q = np.c_[np.interp(t, seg, P[:, 0]), np.interp(t, seg, P[:, 1])]
    T = np.gradient(Q, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    Nn = np.c_[-T[:, 1], T[:, 0]]
    Nn *= np.sign(Nn[:, 1:2] + 1e-9)
    return Q, T, Nn, seg[-1]


def lips_region(m, lab, seed, look, skin, keep_out=(), half_w=None, up_k=0.23, low_k=0.30, k0=0.5,
                tree=None, verbose=True, line_pts=None, repaint=()):
    """The two sculpted lip bulges round the mouth line. line_pts: the mouth line traced corner to corner (3D
    points on the surface, front view); without it the crease is traced from `seed`. Works in the line's own
    frame (arc length along it, distance across it), so tilted heads and smiling corners are fine.
    Returns (lip faces, opening faces). Only skin is painted lip; `keep_out` labels (teeth, mouth inside) form
    the opening of an open mouth and get the teeth colour."""
    C = m.triangles_center
    tr = tree if tree is not None else cKDTree(C)
    seed = np.asarray(seed, float)
    R0 = 9.0 if half_w is None else 2.2 * half_w
    if line_pts is not None and len(line_pts) > 1:
        R0 = max(R0, 0.75 * float(np.linalg.norm(np.diff(line_pts, axis=0), axis=1).sum()) + 3.0)
    fsel = np.array(tr.query_ball_point(seed, R0), dtype=int)
    sm, _ = submesh(m, fsel)
    Cs = sm.triangles_center
    kap = edge_curvature(sm, 4)
    ei, ej = sm.face_adjacency.T
    fk = np.zeros(len(Cs))
    np.minimum.at(fk, ei, kap)
    np.minimum.at(fk, ej, kap)
    up_ax = np.array([0, 0, 1.0])
    ux = np.cross(look, up_ax)
    ux /= np.linalg.norm(ux)
    U, Vv = (Cs - seed) @ ux, (Cs - seed) @ up_ax
    P2 = np.c_[U, Vv]
    sl = lab[fsel]
    facing = (sm.face_normals @ -look) > 0.0
    opening = np.isin(sl, list(keep_out)) & facing
    if opening.any() and len(keep_out):
        # an open mouth holds teeth (the lightest keep label); dark-only components are glasses frames,
        # nostrils or a moustache and must never be repainted
        from scipy.sparse.csgraph import connected_components as _cc
        ok_ = opening[ei] & opening[ej]
        _, oc_ = _cc(sp.coo_matrix((np.ones(ok_.sum()), (ei[ok_], ej[ok_])), shape=(len(Cs),) * 2), directed=False)
        light = sl == keep_out[0]
        has = np.bincount(oc_[opening & light], minlength=oc_.max() + 1) > 20
        opening &= has[oc_]
    if line_pts is not None:
        L2 = np.c_[(line_pts - seed) @ ux, (line_pts - seed) @ up_ax]
    else:
        tl = trace_mouth_line(U, Vv, fk, facing)
        if len(tl) < 5:
            return np.array([], int), np.array([], int), None
        o = np.argsort(U[tl])
        L2 = np.array([P2[tl][o][k] for k in np.linspace(0, len(tl) - 1, 12).astype(int)])
    # the mouth line = cheapest path between the two traced corners through a 0.8 mm tube round the trace,
    # cheap on concave faces (the sculpted crease) and on the opening (teeth / mouth inside)
    from scipy.sparse.csgraph import dijkstra
    Q0, _, _, _ = _polyline(L2)
    dtube, _ = cKDTree(Q0).query(P2)
    tube = (dtube < 0.8) & facing
    cost = np.where(opening, 0.05, np.exp(np.clip(fk, -6, 3) / 0.8))
    okE = tube[ei] & tube[ej]
    wE = np.linalg.norm(Cs[ei] - Cs[ej], axis=1) * (cost[ei] + cost[ej]) / 2
    Gp = sp.coo_matrix((wE[okE], (ei[okE], ej[okE])), shape=(len(Cs),) * 2).tocsr()
    tv = np.where(tube)[0]
    if not len(tv):
        return np.array([], int), np.array([], int), None
    a0 = tv[np.argmin(np.linalg.norm(P2[tv] - Q0[0], axis=1))]
    a1 = tv[np.argmin(np.linalg.norm(P2[tv] - Q0[-1], axis=1))]
    dist_, pred = dijkstra(Gp, directed=False, indices=a0, return_predecessors=True)
    path = [a1]
    while np.isfinite(dist_[a1]) and path[-1] != a0 and pred[path[-1]] >= 0 and len(path) < 20000:
        path.append(pred[path[-1]])
    if path[-1] != a0:                     # no connected tube: fall back to the trace itself
        path_pts = Q0
    else:
        path_pts = P2[np.array(path[::-1])]
    ker = np.ones(9) / 9
    path_pts = np.c_[np.convolve(np.pad(path_pts[:, 0], 4, mode="edge"), ker, "valid"),
                     np.convolve(np.pad(path_pts[:, 1], 4, mode="edge"), ker, "valid")]
    # run each end out along the crease to the corner pit (hand-painted lips end in the commissure dimples),
    # or trim it back where the trace overshot onto smooth skin
    skin_ok = facing & ~opening
    cand = np.where(skin_ok)[0]
    tcand = cKDTree(P2[cand])

    def follow(p0, d0, max_len=3.0, step=0.2):
        out, p, d = [], p0.copy(), d0 / (np.linalg.norm(d0) + 1e-12)
        weak = 0
        for _ in range(int(max_len / step)):
            q = p + d * step
            idx = cand[tcand.query_ball_point(q, 0.45)]
            if not len(idx):
                break
            rel = P2[idx] - q
            idx = idx[np.abs(rel @ d) < 0.12]
            if not len(idx):
                break
            f = idx[np.argmin(fk[idx])]
            if fk[f] > -1.0:
                weak += 1
                if weak >= 2:
                    break
                p = q
                continue
            weak = 0
            nd = P2[f] - p
            if np.linalg.norm(nd) > 1e-6:
                d = 0.7 * d + 0.3 * nd / np.linalg.norm(nd)
                d /= np.linalg.norm(d)
            p = P2[f]
            out.append(p.copy())
        return out

    def trim(P):            # drop end samples that sit on no crease (an overshooting trace)
        def on(k):
            idx = cand[tcand.query_ball_point(P[k], 0.3)]
            return len(idx) and fk[idx].min() < -1.0
        i0, i1 = 0, len(P) - 1
        while i1 - i0 > 4 and not on(i0) and not (opening[np.argmin(np.linalg.norm(P2 - P[i0], axis=1))]):
            i0 += 1
        while i1 - i0 > 4 and not on(i1) and not (opening[np.argmin(np.linalg.norm(P2 - P[i1], axis=1))]):
            i1 -= 1
        return P[i0:i1 + 1]

    def a_open_ok(o):
        return sm.area_faces[o].sum() > 0.5

    Qt, _, _, _ = _polyline(path_pts, 0.2)
    Qt = trim(Qt)
    core = float(np.linalg.norm(np.diff(Qt, axis=0), axis=1).sum())
    # open mouths: the corner pits sit ~1-2 mm past the teeth; closed: the trace already ran along the crease
    ext = 2.0 if (opening.any() and a_open_ok(opening)) else 0.8
    head = follow(Qt[0], Qt[0] - Qt[min(4, len(Qt) - 1)], max_len=ext)
    tail = follow(Qt[-1], Qt[-1] - Qt[max(-5, -len(Qt))], max_len=ext)
    if verbose:
        print(f"  lips: mouth line {core:.1f} mm + corners {0.2 * len(head):.1f} / {0.2 * len(tail):.1f} mm")
    path_pts = np.vstack([np.array(head[::-1]).reshape(-1, 2), Qt, np.array(tail).reshape(-1, 2)])
    Q, T, Nn, length = _polyline(path_pts)
    length = max(core, 0.85 * length)
    half = half_w or length / 2
    # line frame: arc position t along the mouth line, side (above / below), and a corner-to-corner envelope
    dq, jq = cKDTree(Q).query(P2)
    t = jq / max(len(Q) - 1, 1)
    above = ((P2 - Q[jq]) * Nn[jq]).sum(1) > 0
    env = np.sqrt(np.clip(1 - (2 * t - 1) ** 2, 0, 1))
    beyond = ((jq == 0) | (jq == len(Q) - 1)) & (dq > 0.3)
    # the mouth band: the line itself, plus the open mouth (teeth / inside labels) along it
    band = (dq < 0.2) & facing
    band |= opening & (dq < 0.8 * low_k * half + 0.2) & (env > 0.15)
    dl, _ = cKDTree(P2[band]).query(P2)
    thick = np.where(above, up_k, low_k) * half * env + 1e-3
    src = (dl < 0.25) & facing & (env > 0.5) & (sl == skin)     # the cut decides where the ends taper off
    soft = [skin] + list(repaint)        # repaint: lip-like colours already there (an old rose lip slot)
    # the teeth core (open mouth along the line) is never lip; the rest of the opening is up to the cut: the
    # texture often paints the inner face of the lower lip white ("lower teeth"), which hand-painters paint red
    core = opening & (dq < 0.3)
    sink = (dl > 1.35 * thick) | beyond | (~np.isin(sl, soft + list(keep_out))) | core
    sink &= ~src
    a = sm.area_faces
    H = face_mean_curvature(sm, 6)
    un = np.zeros((len(Cs), 2))
    r = np.clip(dl / thick, 0, 2)
    # the lips are the two CONVEX bulges either side of the line; the philtrum / chin-fold valleys are concave
    un[:, 1] = 2.0 * a * np.clip(H / 0.3, 0, 1) * np.clip(1.2 - r, 0, 1)      # convex near the line: want lip
    un[:, 0] = 2.0 * a * np.clip(-H / 0.3, 0, 1) + 1.0 * a * r ** 2           # concave or far: want skin
    un[:, 0] += np.where(opening, 1.0 * a, 0.0)                               # teeth stay teeth unless clearly lip
    L = crease_cut(sm, src, sink, unary=un, k0=k0, kappa=kap, smooth=1.5)   # hand-painted lips are smoother than the sculpt
    teeth_keep = opening & ~L
    lip = L & np.isin(sl, soft + list(keep_out)) & ~teeth_keep
    # one smooth lens, the way hand-painters paint lips: measure the upper / lower extent of the cut along the
    # mouth line, keep only the stretch where BOTH lips are present (drops one-sided spurs past the corners),
    # smooth it and taper it to points at the ends. The profile is kept so refinement re-cuts a smooth edge.
    from scipy.ndimage import gaussian_filter1d
    side = ((P2 - Q[jq]) * Nn[jq]).sum(1)
    nQ = len(Q)
    wnd = max(2, nQ // 25)
    upE, loE = np.zeros(nQ), np.zeros(nQ)
    for j in range(nQ):
        sj = lip & (np.abs(jq - j) <= wnd)
        a_, b_ = side[sj & (side > 0)], -side[sj & (side < 0)]
        upE[j] = np.percentile(a_, 90) if len(a_) > 3 else 0.0
        loE[j] = np.percentile(b_, 90) if len(b_) > 3 else 0.0
    both = (upE > 0.15) & (loE > 0.15)
    lens_obj = None
    if both.sum() >= 5:
        ja, jb = np.where(both)[0][[0, -1]]
        # fill notches (a bite the cut took out of a lip) before smoothing: grey closing over ~2.5 mm
        from scipy.ndimage import grey_closing
        step = length / max(nQ - 1, 1)
        wc = max(3, int(round(2.5 / max(step, 1e-3))))
        upE[ja:jb + 1] = grey_closing(upE[ja:jb + 1], size=wc, mode="nearest")
        loE[ja:jb + 1] = grey_closing(loE[ja:jb + 1], size=wc, mode="nearest")
        sig = max(1.0, 0.05 * (jb - ja))
        tt = np.clip((np.arange(nQ) - ja) / max(jb - ja, 1), 0, 1)
        env = np.sin(np.pi * tt) ** 0.7
        up = gaussian_filter1d(upE, sig, mode="nearest") * env
        lo = gaussian_filter1d(loE, sig, mode="nearest") * env
        up[:ja] = lo[:ja] = 0
        up[jb + 1:] = lo[jb + 1:] = 0
        lens = facing & np.isin(sl, soft + list(keep_out)) & ~core & ~(teeth_keep & (dq < 0.6)) & \
            (jq >= ja) & (jq <= jb) & \
            (((side >= 0) & (side <= up[jq])) | ((side < 0) & (-side <= lo[jq])))
        if lens.sum() > 20:
            lip = lens
            lens_obj = LipLens(seed, ux, up_ax, Q, Nn, up, lo)
    # the opening: teeth / dark-inside labels in the band, plus skin the lips fully enclose -> teeth colour
    # (hand-painters remove every dark line between teeth and lips and leaves no skin gap)
    win = (dl < 2.5) & ~beyond
    rest = win & ~lip
    from scipy.sparse.csgraph import connected_components
    ok = rest[ei] & rest[ej]
    _, comp = connected_components(sp.coo_matrix((np.ones(ok.sum()), (ei[ok], ej[ok])), shape=(len(Cs),) * 2),
                                   directed=False)
    edge = np.r_[ei[rest[ei] & ~win[ej]], ej[rest[ej] & ~win[ei]]]
    enclosed = rest & ~np.isin(comp, comp[edge]) & np.isin(sl, [skin] + list(keep_out))
    inside = band & opening & ~lip
    teeth = fsel[(enclosed | inside) & ~lip]
    out = fsel[lip]
    if lens_obj is not None:            # faces the refined edge may move into: soft faces just outside the lens
        dlens, _ = cKDTree(Cs[lip]).query(Cs, distance_upper_bound=0.35)
        lens_obj.ring = fsel[np.isfinite(dlens) & ~lip & np.isin(sl, soft) & facing & ~opening]
        lens_obj.faces = out
    if verbose:
        print(f"  lips: mouth line {length:.1f} mm, {m.area_faces[out].sum():.1f} mm2 lips, "
              f"{m.area_faces[teeth].sum():.1f} mm2 opening")
    return out, teeth, lens_obj


class LipLens:
    """the smooth lip lens: point -> inside? (mouth-line frame: arc sample, signed distance across the line)"""

    def __init__(self, seed, ux, up_ax, Q, Nn, up, lo):
        self.seed, self.ux, self.up_ax, self.Q, self.Nn, self.up, self.lo = seed, ux, up_ax, Q, Nn, up, lo
        self.tree = cKDTree(Q)
        self.faces = self.ring = np.array([], int)

    def contains(self, P3):
        P = np.c_[(P3 - self.seed) @ self.ux, (P3 - self.seed) @ self.up_ax]
        _, j = self.tree.query(P)
        sd = ((P - self.Q[j]) * self.Nn[j]).sum(1)
        return ((sd >= 0) & (sd <= self.up[j])) | ((sd < 0) & (-sd <= self.lo[j]))


# ---------------------------------------------------------------- seeds from the front view

def seed_from_uv(m, u, v, az, vis=None):
    """3D point of the visible surface at front-view (u, v) mm (paint_features convention)"""
    a = np.radians(az)
    C = m.triangles_center
    uc = C[:, 0] * np.cos(a) - C[:, 1] * np.sin(a)
    dc = C[:, 0] * np.sin(a) + C[:, 1] * np.cos(a)
    vc = C[:, 2]
    dist = np.hypot(uc - u, vc - v)
    if vis is not None:                     # nearest visible face
        cand = np.where((dist < 1.0) & vis)[0]
        if len(cand):
            return C[cand[np.argmin(dist[cand])]]
    cand = np.where(dist < 0.3)[0]          # else the front-most surface at that point
    if not len(cand):
        cand = np.where(dist < 1.0)[0]
    if not len(cand):
        return None                         # nothing of this mesh there (a split piece without that face)
    return C[cand[np.argmin(dc[cand])]]


def detect_eyes(m, lab, white, black, az=270, G=None, tree=None, skin=None, darks=None, verbose_eyes=False):
    """Candidate eyes: dark+light texture blobs facing the view that sit on a sculpted eyeball (sphere bounded
    by a concave rim). white may be None (palette without a white: bare dark eyes only). darks: every palette
    index an iris may carry (black, a brown or navy slot); default [black]. Returns [(seed3d, info)] best first."""
    C, N = m.triangles_center, m.face_normals
    look = view_dir(az)
    facing = (N @ -look) > 0.2
    ar = m.area_faces
    from scipy.sparse.csgraph import connected_components
    tree = tree if tree is not None else cKDTree(C)
    darks = [black] if darks is None else list(darks)
    isdark = np.isin(lab, darks)
    iswhite = (lab == white) if white is not None else np.zeros(len(lab), bool)

    def blobs(cmask, nmask, dark_seed):
        f_ = np.where(facing & cmask)[0]
        if not len(f_):
            return []
        pr = cKDTree(C[f_]).query_pairs(0.3, output_type="ndarray")
        nc_, cl_ = connected_components(sp.coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])),
                                                      shape=(len(f_), len(f_))), directed=False)
        A_ = np.bincount(cl_, weights=ar[f_], minlength=nc_)
        nf = np.where(facing & nmask)[0]
        ntree = cKDTree(C[nf]) if len(nf) else None
        out = []
        for k in np.argsort(-A_):
            if not 0.3 <= A_[k] <= 150:
                continue
            f = f_[cl_ == k]
            # a textured eye has its dark and its white next to each other; heads coloured WITHOUT white (the
            # split-piece workflow, or no white filament) have only the dark blob -- kept, but it must pass
            # stricter checks below
            touch = ntree is not None and not np.isinf(ntree.query(C[f], distance_upper_bound=0.8)[0]).all()
            if not touch:
                if not dark_seed or A_[k] > 40 or skin is None:
                    continue
                # cheap check before the expensive eyeball fit: a bare dark eye sits in skin
                ring = np.array(tree.query_ball_point(C[f].mean(0), np.sqrt(A_[k]) + 1.2), dtype=int)
                ring = ring[~cmask[ring]]
                if not len(ring) or ar[ring][lab[ring] == skin].sum() / ar[ring].sum() < 0.7:
                    continue
            out.append((A_[k], C[f].mean(0)))
        return out

    G = G if G is not None else face_graph(m)
    # seeds: dark blobs (the iris) and light blobs (the whites -- needed when the iris merged with a glasses
    # frame or brow into one big dark region)
    cands = blobs(isdark, iswhite, True) + (blobs(iswhite, isdark, False) if white is not None else [])
    cands = sorted(cands, key=lambda x: -x[0])
    found = []
    seen = np.zeros(len(C), bool)
    for _, seed in cands:
        f0 = tree.query(seed)[1]
        if seen[f0]:
            continue                       # already tested an opening containing this seed
        reg, c, R = eye_opening(m, seed, G=G, tree=tree)
        if len(reg) < 50:
            continue
        # a texture blob's centroid is often off the eyeball's centre (gaze, lid shadow) and the cut then leaks:
        # re-seed at the centre of what the first cut found, keep whichever fits the sphere better
        cq0, Rq0 = fit_sphere(C[reg], np.sqrt(ar[reg]))
        q0 = float(np.percentile(np.abs(np.linalg.norm(C[reg] - cq0, axis=1) - Rq0), 90))
        if q0 > 0.05:
            s2 = C[reg][np.argmin(np.linalg.norm(C[reg] - C[reg].mean(0), axis=1))]
            reg2, c2, R2 = eye_opening(m, s2, G=G, tree=tree)
            if len(reg2) >= 50:
                cq2, Rq2 = fit_sphere(C[reg2], np.sqrt(ar[reg2]))
                q2 = float(np.percentile(np.abs(np.linalg.norm(C[reg2] - cq2, axis=1) - Rq2), 90))
                if q2 < q0:
                    reg, c, R = reg2, c2, R2
        seen[reg] = True
        a_open = ar[reg].sum()
        wfrac = ar[reg][iswhite[reg]].sum() / a_open
        bfrac = ar[reg][isdark[reg]].sum() / a_open
        if not (1.5 < a_open < 300 and 1.0 < R < 25 and bfrac > 0.1 and wfrac + bfrac > 0.6):
            continue
        no_white = wfrac <= 0.05           # a black-only eye (head coloured without white) -- or hair / a shoe
        if skin is not None:               # an eye sits in skin (allowing for liner, lashes, glasses)
            inO = np.zeros(len(C), bool)
            inO[reg] = True
            nb = np.array(tree.query_ball_point(C[reg].mean(0), 4.0), dtype=int)
            nb = nb[~inO[nb]]
            d, _ = cKDTree(C[reg]).query(C[nb])
            ring = nb[d < 1.2]
            if not len(ring):
                continue
            rs = ar[ring][lab[ring] == skin].sum() / ar[ring].sum()
            rb = ar[ring][isdark[ring]].sum() / ar[ring].sum()
            if rs < 0.15 or rs + rb < 0.6:    # skin round it, maybe with liner / lashes / a glasses frame
                continue
            if no_white and rs < 0.5:
                continue
        elif no_white:
            continue
        # quality: a real eyeball fits a sphere; an opening that leaked onto the lid or lashes does not
        cq, Rq = fit_sphere(C[reg], np.sqrt(ar[reg]))
        q = float(np.percentile(np.abs(np.linalg.norm(C[reg] - cq, axis=1) - Rq), 90))
        if no_white and (q > 0.03 or not 1.0 < Rq < 25):
            continue                       # without white, only a convincing eyeball counts
        if q > 0.15 * max(1.0, Rq / 4.0):
            continue                       # leaked onto the lid or lashes: real eyeballs fit to ~0.05 mm
        found.append((q, reg, dict(area=a_open, R=R, white=wfrac, black=bfrac, fit=q)))
        if len(found) >= 24:
            break
    # best fits first; drop anything overlapping an eye already kept
    taken = np.zeros(len(C), bool)
    out = []
    if verbose_eyes:
        for q, reg, info in sorted(found, key=lambda x: x[0]):
            print("   cand", np.round(C[reg].mean(0), 2), {k: round(float(v), 3) for k, v in info.items()})
    for q, reg, info in sorted(found, key=lambda x: x[0]):
        if taken[reg].mean() > 0.1:
            continue
        taken[reg] = True
        out.append((C[reg].mean(0), info))
    return out


def loose_eyes(m, lab, darks, skin, az=270, tree=None, near=()):
    """Texture-only eye candidates: compact dark blobs that face the viewer and sit in skin. Used to FIND a face
    whose eyes are not sculpted eyeballs (small or realistic eyes, eyes behind lenses), so the face still gets its
    lips and grow; the eyes themselves are then left as the texture painted them. `near`: points already taken."""
    C, N = m.triangles_center, m.face_normals
    look = view_dir(az)
    ar = m.area_faces
    tree = tree if tree is not None else cKDTree(C)
    isd = np.isin(lab, darks)
    f_ = np.where(isd & ((N @ -look) > 0.0))[0]
    if not len(f_):
        return []
    from scipy.sparse.csgraph import connected_components
    pr = cKDTree(C[f_]).query_pairs(0.3, output_type="ndarray")
    nc, cl = connected_components(sp.coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(f_),) * 2),
                                  directed=False)
    A = np.bincount(cl, weights=ar[f_], minlength=nc)
    an = np.radians(az)
    out = []
    for k in np.where((A >= 0.4) & (A <= 25))[0]:
        f = f_[cl == k]
        p = (C[f] * ar[f][:, None]).sum(0) / ar[f].sum()
        if any(np.linalg.norm(p - q) < 2.5 for q in near):
            continue
        nrm = (N[f] * ar[f][:, None]).sum(0)
        if nrm @ -look / (np.linalg.norm(nrm) + 1e-12) < 0.6:          # nostrils / mouth insides face down
            continue
        u = C[f][:, 0] * np.cos(an) - C[f][:, 1] * np.sin(an)
        P2 = np.c_[u, C[f][:, 2]] - [u.mean(), C[f][:, 2].mean()]
        ev = np.sqrt(np.maximum(np.linalg.eigvalsh(np.cov(P2.T, aweights=ar[f])), 1e-9))
        if ev[1] > 2.2 * ev[0]:                                    # elongated: brow, lash line, mouth line
            continue
        ring = np.array(tree.query_ball_point(p, np.sqrt(A[k]) + 1.0), dtype=int)
        ring = ring[~isd[ring]]
        if not len(ring) or ar[ring][lab[ring] == skin].sum() / ar[ring].sum() < 0.6:
            continue
        out.append((p, dict(area=float(A[k]), fit=np.inf, loose=True)))
    return out


def suggest_features(m, lab, palette, view=270, white=None, black=None, skin=None, women=(), lip_hex="#CE0000",
                     verbose=True):
    """Find the faces (pairs of sculpted eyes), trace each mouth line, and return features.txt lines:
    eye3d per eye (liner for faces listed in `women`), lips3d + teeth for women, and one grow over each head.
    Faces are numbered left to right in the view. Check the preview before trusting it."""
    lum = np.array([sum(int(h[k:k + 2], 16) for k in (1, 3, 5)) for h in palette])
    if white is None:
        labp = _srgb_to_lab(np.array([[int(h.lstrip('#')[k:k + 2], 16) for k in (0, 2, 4)] for h in palette], float))
        wc = np.where((labp[:, 0] > 82) & (np.hypot(labp[:, 1], labp[:, 2]) < 12))[0]
        white = int(wc[np.argmax(labp[wc, 0])]) if len(wc) else None
    black = int(np.argmin(lum)) if black is None else black
    if skin is None:
        skin = skin_index(palette)
    if skin is None:
        raise SystemExit("no skin colour recognised in the palette -- pass --skin #HEX")
    look = view_dir(view)
    labp = _srgb_to_lab(np.array([[int(h.lstrip('#')[k:k + 2], 16) for k in (0, 2, 4)] for h in palette], float))
    darks = [k for k in range(len(palette)) if k not in (skin, white) and labp[k, 0] < 40] or [black]
    white_hex = palette[white] if white is not None else "#F0ECE6"
    if verbose:
        print(f"  skin {palette[skin]}, eye white {white_hex}" + ("" if white is not None else
              " (no white filament in the palette: eyes will ADD one)") + f", iris colours {[palette[k] for k in darks]}")
    G, tree = face_graph(m), cKDTree(m.triangles_center)
    eyes = detect_eyes(m, lab, white, black, view, G=G, tree=tree, skin=skin, darks=darks)
    an = np.radians(view)
    uv = lambda p: (p[0] * np.cos(an) - p[1] * np.sin(an), p[2])
    pts = [p for p, _ in eyes]
    infos = [i for _, i in eyes]

    def pair(cands, used, faces, lo_=0.5, hi_=2.0, dmin=1.2, dmax=5.0, typical=None):
        """greedy pairing. typical: preferred distance / blob width (texture-only eyes sit ~5.5 widths apart,
        so a blob next to an eye -- a lash end, a brow tip -- doesn't win just by being closest)"""
        order = sorted(cands, key=lambda k: (infos[k]["fit"], -infos[k]["area"]))   # best eyeballs pair first
        for k in order:
            if k in used:
                continue
            w = np.sqrt(infos[k]["area"])
            best = None
            for j in order:
                if j == k or j in used:
                    continue
                d = np.linalg.norm(pts[j] - pts[k])
                ratio = infos[j]["area"] / infos[k]["area"]
                ww = max(w, np.sqrt(infos[j]["area"]))
                if not (dmin * ww < d < dmax * ww and abs(pts[j][2] - pts[k][2]) < 0.6 * d and lo_ < ratio < hi_):
                    continue
                score = d if typical is None else abs(np.log(d / (typical * ww))) + abs(np.log(ratio))
                if best is None or score < best[0]:
                    best = (score, j)
            if best:
                used |= {k, best[1]}
                faces.append((k, best[1]))

    used, faces = set(), []
    pair(range(len(pts)), used, faces)                      # 1. sculpted eyes
    # 2. texture-only eyes, for heads with no sculpted pair: away from every face already found
    taken = [p for f in faces for p in ((pts[f[0]] + pts[f[1]]) / 2,)]
    rad = [1.8 * np.linalg.norm(pts[f[0]] - pts[f[1]]) for f in faces]
    lo = loose_eyes(m, lab, darks, skin, view, tree=tree, near=pts)
    lo = [(p, i) for p, i in lo if all(np.linalg.norm(p - q) > r for q, r in zip(taken, rad))]
    first = len(pts)
    pts += [p for p, _ in lo]
    infos += [i for _, i in lo]
    singles = [k for k in range(first) if k not in used and all(np.linalg.norm(pts[k] - q) > r for q, r in zip(taken, rad))]
    pair(singles + list(range(first, len(pts))), used, faces, lo_=0.33, hi_=3.0, dmin=3.5, dmax=9.0, typical=5.5)
    # a face found only from texture blobs must be the TOP pair on its head (not mouth corners or buttons below
    # the real eyes)
    def mid_(f):
        return (pts[f[0]] + pts[f[1]]) / 2
    keep = []
    for f in faces:
        if all(infos[x].get("loose") for x in f):
            mf, iod_f = mid_(f), np.linalg.norm(pts[f[0]] - pts[f[1]])
            # anything below another face, roughly under it, is not a face (mouth corners, jacket buttons)
            if any(g is not f and mid_(g)[2] > mf[2] + 0.5 * iod_f and
                   np.linalg.norm((mid_(g) - mf)[:2]) < max(3.0 * iod_f, 1.5 * np.linalg.norm(pts[g[0]] - pts[g[1]]))
                   for g in faces):
                continue
        keep.append(f)
    faces = keep
    faces.sort(key=lambda f: uv((pts[f[0]] + pts[f[1]]) / 2)[0])
    lines = []
    up = np.array([0, 0, 1.0])
    for n, (a, b) in enumerate(faces):
        pa, pb = pts[a], pts[b]
        mid, iod = (pa + pb) / 2, np.linalg.norm(pa - pb)
        woman = n in women
        tag = "woman" if woman else "man"
        lines.append(f"# face {n} ({tag}): eyes {iod:.1f} mm apart")
        keep_tex = infos[a].get("loose") or infos[b].get("loose")      # repaint both eyes or neither
        for x in sorted((a, b), key=lambda q: uv(pts[q])[0]):
            u, v = uv(pts[x])
            if keep_tex:
                lines.append(f"#   eye at {u:.2f},{v:.2f}: no sculpted eyeball -- kept as the texture painted it")
                continue
            lines.append(f"eye3d:{white_hex},{palette[black]},{palette[skin]}@{u:.2f},{v:.2f}" + (",liner" if woman else ""))
        u0, v0 = uv(mid)
        r = 1.3 * iod
        kept = [a, b] if keep_tex else []
        skip = ("|skip=" + ";".join(f"{uv(pts[x])[0]:.2f},{uv(pts[x])[1]:.2f},{0.7 * np.sqrt(infos[x]['area']) + 0.4:.2f}"
                                     for x in kept)) if kept else ""
        lines.append(f"grow:{palette[black]}>{palette[skin]}@{u0 - r:.1f},{u0 + r:.1f},{v0 - r:.1f},{v0 + 0.9 * r:.1f}" + skip)
        if woman:
            # mouth: ~0.75 x the eye distance below the eyes, along the face's own up (heads tilt)
            (ua, va), (ub, vb) = uv(pa), uv(pb)
            e = np.r_[ub - ua, vb - va]
            e /= np.linalg.norm(e) + 1e-12
            nrm = np.r_[-e[1], e[0]]
            nrm *= np.sign(nrm[1] + 1e-9)
            mu, mv = np.r_[(ua + ub) / 2, (va + vb) / 2] - 0.75 * np.hypot(ub - ua, vb - va) * nrm
            seed = seed_from_uv(m, mu, mv, view)
            fsel = np.array(tree.query_ball_point(seed, max(9.0, 0.9 * iod)), dtype=int)
            sm, _ = submesh(m, fsel)
            Cs = sm.triangles_center
            kap = edge_curvature(sm, 4)
            ei, ej = sm.face_adjacency.T
            fk = np.zeros(len(Cs))
            np.minimum.at(fk, ei, kap)
            np.minimum.at(fk, ej, kap)
            ux = np.cross(look, up)
            ux /= np.linalg.norm(ux)
            U, Vv = (Cs - seed) @ ux, (Cs - seed) @ up
            facing = (sm.face_normals @ -look) > 0
            box = facing & (np.abs(U) < 0.6 * iod) & (np.abs(Vv) < 0.35 * iod)
            P = None
            # 1. an open mouth: teeth / dark-inside labels, wider than tall, nearest the seed
            op = np.where(box & np.isin(lab[fsel], [white, black]))[0] if white is not None else np.array([], int)
            if len(op) > 30:
                from scipy.sparse.csgraph import connected_components
                pr = cKDTree(Cs[op]).query_pairs(0.25, output_type="ndarray")
                _, cl = connected_components(sp.coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])),
                                                           shape=(len(op),) * 2), directed=False)
                best = None
                for c_ in np.unique(cl):
                    f = op[cl == c_]
                    A_ = sm.area_faces[f].sum()
                    wA = sm.area_faces[f][lab[fsel][f] == white].sum()
                    if A_ > 0.5 and wA > 0.3 * A_ and np.ptp(U[f]) > 1.5 * np.ptp(Vv[f]) and np.ptp(U[f]) > 0.3 * iod:
                        dd = np.min(np.hypot(U[f], Vv[f]))
                        if best is None or dd < best[0]:
                            best = (dd, f)
                if best is not None:
                    f = best[1]
                    cols = np.linspace(U[f].min(), U[f].max(), 7)
                    P = []
                    for cu in cols:
                        q = f[np.abs(U[f] - cu) < 0.4]
                        if len(q):
                            P.append(Cs[q].mean(0))
                    ext = 0.06 * (U[f].max() - U[f].min())            # lips run a little past the teeth
                    P = [P[0] - ux * ext] + P + [P[-1] + ux * ext]
            # 2. a closed mouth: the longest crease traced from heights round the seed
            if P is None:
                best = None
                sk = lab[fsel] == skin
                nsk = np.where(~sk)[0]
                dn = cKDTree(Cs[nsk]).query(Cs)[0] if len(nsk) else np.full(len(Cs), np.inf)
                okm = facing & sk & (dn > 0.5)
                # trace in the face's own frame (eye line horizontal): heads tilt, mouths follow them
                ang = np.arctan2(e[1], e[0])
                Ur = U * np.cos(ang) + Vv * np.sin(ang)
                Vr = -U * np.sin(ang) + Vv * np.cos(ang)
                for dv in np.arange(-0.25 * iod, 0.25 * iod + 1e-6, 0.4):
                    tl = trace_mouth_line(Ur, Vr - dv, fk, okm, band=0.3)
                    if len(tl) < 5:
                        continue
                    score = np.ptp(Ur[tl]) - 1.5 * abs(np.median(Vr[tl]))   # long, and near where a mouth goes
                    if best is None or score > best[0]:
                        best = (score, tl)
                if best is None or np.ptp(Ur[best[1]]) < 0.35 * iod:
                    lines.append(f"# lips: no mouth line found below the eyes -- trace it by hand (lips3d:{lip_hex},...)")
                    continue
                tl = best[1]
                o = np.argsort(Ur[tl])
                P = list(Cs[tl][o][np.linspace(0, len(tl) - 1, 7).astype(int)])
            poly = ";".join(f"{uv(q)[0]:.2f},{uv(q)[1]:.2f}" for q in P)
            keep = f"{white_hex}+{palette[black]}" if white is not None else palette[black]
            lines.append(f"lips3d:{lip_hex},{palette[skin]}@{poly}|keep={keep}|teeth={white_hex}")
    if verbose:
        print(f"  found {len(eyes)} eyes -> {len(faces)} faces")
    return lines


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    from render_3mf import load_colored_3mf
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--view", type=float, default=270)
    ap.add_argument("--detect", action="store_true", help="print suggested features.txt lines")
    ap.add_argument("--women", default="", help="face numbers (left to right) that get liner + lips, e.g. 0,2")
    ap.add_argument("--lip", default="#CE0000", help="lip filament (default #CE0000)")
    a = ap.parse_args()
    mesh, rgb, pal = load_colored_3mf(Path(a.input))
    lab = np.argmin(((rgb[:, None] - pal[None]) ** 2).sum(2), 1)
    palette = ["#%02X%02X%02X" % tuple(int(c) for c in p) for p in pal]
    lum = pal.sum(1)
    white, black = int(np.argmax(lum)), int(np.argmin(lum))
    if a.detect:
        women = [int(x) for x in a.women.split(",") if x.strip()]
        print("\n".join(suggest_features(mesh, lab, palette, a.view, white, black, women=women, lip_hex=a.lip)))


if __name__ == "__main__":
    main()
