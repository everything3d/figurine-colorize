#!/usr/bin/env python3
"""Client-facing "product photo" render of a coloured figurine 3MF.

render_3mf.py is the CHECKER: flat, shadowless, deliberately boring, so colour boundaries can be
judged honestly. This is the SHOWCASE: bright high-key studio lighting for sending to clients.

    python3 scripts/render_pretty.py <model>.3mf -o <out>.png                 # 4 views, tiled
    python3 scripts/render_pretty.py <model>.3mf -o hero.png --angles 270 --width 1600 --height 2000
    python3 scripts/render_pretty.py a.3mf -o out.png --separate              # also out_<angle>.png

Reads either encoding:
  * colorgroup 3MF (colorize_3mf.py / clean_colorize.py output, per-triangle p1)
  * Bambu/Orca project with per-triangle paint_color (colorgroup_to_painted.py, split parts, U1 plates)

Look: smooth shading; warm key light with soft shadows, cool fill, rim light that outlines the
silhouette; baked ambient occlusion (creases, under arms, feet on the floor); faint satin sheen like
PLA; soft contact + cast shadow on a seamless light-grey sweep. Exposure is set so a lit surface
shows close to its filament hex: ivory stays ivory, black reads as charcoal with highlights.
Azimuths mean the same as in render_3mf.py (mesh rotated about Z, camera looking along +Y), so the
view you call "front" there is the same number here.

GPU via moderngl (pip install moderngl); AO via trimesh + embree, cached in ~/.cache/render_pretty.
"""
import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

PAINT_CODES = ["4", "8", "0C", "1C", "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C", "AC", "BC", "CC", "DC"]


# ---------------------------------------------------------------- loading

def _hex(h):
    h = h.lstrip("#")[:6]
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def load_model(path):
    """-> (vertices, faces, per-face RGB uint8). Accepts colorgroup or paint_color 3MFs."""
    z = zipfile.ZipFile(path)
    names = z.namelist()
    models = [n for n in names if n.endswith(".model")]
    V_all, F_all, C_all, off = [], [], [], 0
    palette = None
    if "Metadata/project_settings.config" in names:
        try:
            palette = json.loads(z.read("Metadata/project_settings.config"))["filament_colour"]
        except Exception:
            palette = None
    for n in models:
        x = z.read(n).decode()
        if "<triangle" not in x:
            continue
        V = np.array(re.findall(r'<vertex x="([^"]+)" y="([^"]+)" z="([^"]+)"', x), float)
        tris = re.findall(r'<triangle ([^>]*)/>', x)
        F = np.array([re.findall(r'v[123]="(\d+)"', t) for t in tris], int)
        cg = re.findall(r'<m:color color="([^"]+)"', x)
        if cg:                                                     # colorgroup encoding
            p1 = np.array([int((re.search(r'p1="(\d+)"', t) or [0, 0])[1]) for t in tris])
            C = np.array([_hex(c) for c in cg])[np.clip(p1, 0, len(cg) - 1)]
        else:                                                      # Bambu paint_color encoding
            pal = np.array([_hex(c) for c in (palette or ["#B0B0B0"])])
            codes = [(re.search(r'paint_color="([0-9A-F]+)"', t) or [None, "4"])[1] for t in tris]
            idx = np.array([PAINT_CODES.index(c) if c in PAINT_CODES else 0 for c in codes])
            C = pal[np.clip(idx, 0, len(pal) - 1)]
        V_all.append(V); F_all.append(F + off); C_all.append(C); off += len(V)
    if not V_all:
        sys.exit(f"{path}: no triangles found")
    return np.vstack(V_all), np.vstack(F_all), np.vstack(C_all).astype(np.float32)


# ---------------------------------------------------------------- ambient occlusion (baked, per vertex)

def bake_ao(V, F, N, n_rays=24, reach=None):
    key = hashlib.sha1(V.tobytes() + F.tobytes() + str(n_rays).encode()).hexdigest()[:16]
    cache = Path.home() / ".cache/render_pretty" / f"{key}.npy"
    if cache.exists():
        return np.load(cache)
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    m = trimesh.Trimesh(V, F, process=False)
    ray = RayMeshIntersector(m)
    h = V[:, 2].max() - V[:, 2].min()
    reach = reach or 0.07 * h
    rng = np.random.default_rng(0)
    # cosine-weighted hemisphere samples in local frame
    u1, u2 = rng.random(n_rays), rng.random(n_rays)
    r, th = np.sqrt(u1), 2 * np.pi * u2
    local = np.stack([r * np.cos(th), r * np.sin(th), np.sqrt(1 - u1)], 1)
    a = np.where(np.abs(N[:, :1]) < 0.9, [[1, 0, 0]], [[0, 1, 0]])
    t1 = np.cross(N, a); t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 = np.cross(N, t1)
    occ = np.zeros(len(V))
    zfloor = V[:, 2].min()
    for k in range(n_rays):
        d = local[k, 0] * t1 + local[k, 1] * t2 + local[k, 2] * N
        o = V + N * (1e-3 * h)
        hit = np.zeros(len(V), bool)
        for s in range(0, len(V), 400_000):
            sl = slice(s, s + 400_000)
            loc, ir, _ = ray.intersects_location(o[sl], d[sl], multiple_hits=False)
            dist = np.linalg.norm(loc - o[sl][ir], axis=1)
            hh = np.zeros(len(o[sl]), bool); hh[ir[dist < reach]] = True
            hit[sl] = hh
        # the floor occludes too: this is what darkens feet and hems where they meet the ground
        with np.errstate(divide="ignore", invalid="ignore"):
            tf = (zfloor - o[:, 2]) / d[:, 2]
        hit |= (d[:, 2] < 0) & (tf < reach)
        occ += hit
    ao = (1 - occ / n_rays).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, ao)
    return ao


# ---------------------------------------------------------------- shaders

VS_MODEL = """
#version 330
uniform mat4 mvp; uniform mat4 model;
in vec3 in_pos; in vec3 in_nrm; in vec3 in_col; in float in_ao;
out vec3 w_pos; out vec3 w_nrm; out vec3 v_col; out float v_ao;
void main() {
    vec4 wp = model * vec4(in_pos, 1.0);
    w_pos = wp.xyz; w_nrm = mat3(model) * in_nrm; v_col = in_col; v_ao = in_ao;
    gl_Position = mvp * vec4(in_pos, 1.0);
}"""

FS_MODEL = """
#version 330
uniform vec3 cam_pos;
uniform vec3 key_dir; uniform vec3 fill_dir; uniform vec3 rim_dir;
uniform vec3 key_col; uniform vec3 fill_col; uniform vec3 rim_col; uniform vec3 sky_col; uniform vec3 gnd_col;
uniform mat4 key_vp; uniform sampler2DShadow key_map; uniform float texel; uniform float soft;
uniform float exposure; uniform float spec_k; uniform float shininess;
in vec3 w_pos; in vec3 w_nrm; in vec3 v_col; in float v_ao;
out vec4 frag;
const vec2 P[16] = vec2[](vec2(-0.94,-0.40),vec2(0.95,-0.77),vec2(-0.09,-0.93),vec2(0.34,0.29),
  vec2(-0.92,0.46),vec2(-0.82,-0.88),vec2(-0.38,0.28),vec2(-0.20,-0.25),vec2(0.43,-0.65),
  vec2(0.96,0.27),vec2(0.65,0.75),vec2(0.56,-0.09),vec2(-0.69,-0.14),vec2(-0.07,0.65),
  vec2(0.15,-0.47),vec2(0.31,0.88));
float shadow(vec3 p, vec3 n) {
    vec4 s = key_vp * vec4(p + n * texel * 1.5, 1.0);
    vec3 q = s.xyz / s.w * 0.5 + 0.5;
    float acc = 0.0;
    for (int i = 0; i < 16; i++) acc += texture(key_map, vec3(q.xy + P[i] * texel * soft, q.z - 0.0008));
    return acc / 16.0;
}
vec3 toLinear(vec3 c) { return pow(c, vec3(2.2)); }
vec3 shoulder(vec3 x) { return mix(x, 0.70 + 0.30 * (1.0 - exp(-(x - 0.70) / 0.30)), step(0.70, x)); }
void main() {
    vec3 n = normalize(w_nrm);
    vec3 v = normalize(cam_pos - w_pos);
    if (dot(n, v) < 0.0) n = -n;
    vec3 alb = toLinear(v_col);
    float ao = mix(0.35, 1.0, pow(v_ao, 1.3));
    vec3 amb = mix(gnd_col, sky_col, 0.5 + 0.5 * n.z) * ao;
    float sh = shadow(w_pos, n);
    float nk = max(dot(n, key_dir), 0.0), nf = max(dot(n, fill_dir), 0.0), nr = max(dot(n, rim_dir), 0.0);
    vec3 diff = amb + key_col * nk * mix(0.25, 1.0, sh) + fill_col * nf * mix(0.6, 1.0, ao) + rim_col * nr * 0.6;
    vec3 hk = normalize(key_dir + v), hr = normalize(rim_dir + v);
    float spec = spec_k * (pow(max(dot(n, hk), 0.0), shininess) * nk * sh + 0.6 * pow(max(dot(n, hr), 0.0), shininess) * nr);
    float fres = pow(1.0 - max(dot(n, v), 0.0), 4.0) * 0.10 * ao;
    vec3 c = alb * diff * exposure + vec3(spec + fres) * mix(vec3(1.0), alb + 0.4, 0.3);
    frag = vec4(pow(shoulder(c), vec3(1.0 / 2.2)), 1.0);
}"""

VS_DEPTH = """
#version 330
uniform mat4 mvp; in vec3 in_pos;
void main() { gl_Position = mvp * vec4(in_pos, 1.0); }"""
FS_DEPTH = """
#version 330
void main() {}"""

VS_GROUND = """
#version 330
uniform mat4 vp; in vec2 in_xy; out vec3 w_pos;
uniform float zf;
void main() { w_pos = vec3(in_xy, zf); gl_Position = vp * vec4(w_pos, 1.0); }"""
FS_GROUND = """
#version 330
uniform mat4 key_vp; uniform sampler2DShadow key_map; uniform float texel; uniform float soft;
uniform mat4 top_vp; uniform sampler2DShadow top_map; uniform float top_texel;
uniform vec3 center; uniform float radius; uniform float cast_k; uniform float contact_k;
in vec3 w_pos; out vec4 frag;
const vec2 P[16] = vec2[](vec2(-0.94,-0.40),vec2(0.95,-0.77),vec2(-0.09,-0.93),vec2(0.34,0.29),
  vec2(-0.92,0.46),vec2(-0.82,-0.88),vec2(-0.38,0.28),vec2(-0.20,-0.25),vec2(0.43,-0.65),
  vec2(0.96,0.27),vec2(0.65,0.75),vec2(0.56,-0.09),vec2(-0.69,-0.14),vec2(-0.07,0.65),
  vec2(0.15,-0.47),vec2(0.31,0.88));
float pcf(sampler2DShadow m, mat4 vp, float tx, float r) {
    vec4 s = vp * vec4(w_pos, 1.0); vec3 q = s.xyz / s.w * 0.5 + 0.5;
    float acc = 0.0;
    for (int ring = 1; ring <= 3; ring++)
        for (int i = 0; i < 16; i++) acc += texture(m, vec3(q.xy + P[i] * tx * r * float(ring) / 3.0, q.z - 0.002));
    return acc / 48.0;
}
void main() {
    float lit_key = pcf(key_map, key_vp, texel, soft * 2.5);
    float lit_top = pcf(top_map, top_vp, top_texel, 18.0);
    float fade = 1.0 - smoothstep(radius * 0.6, radius * 1.6, length(w_pos.xy - center.xy));
    float dark = ((1.0 - lit_key) * cast_k + (1.0 - lit_top) * contact_k) * fade;
    frag = vec4(0.0, 0.0, 0.0, clamp(dark, 0.0, 0.85));
}"""

VS_BG = """
#version 330
in vec2 in_xy; out vec2 uv;
void main() { uv = in_xy * 0.5 + 0.5; gl_Position = vec4(in_xy, 0.9999, 1.0); }"""
FS_BG = """
#version 330
uniform vec3 top; uniform vec3 bot; in vec2 uv; out vec4 frag;
void main() {
    vec3 c = mix(bot, top, smoothstep(0.0, 1.0, uv.y));
    float vig = 1.0 - 0.06 * pow(length(uv - vec2(0.5, 0.55)) * 1.3, 2.0);
    frag = vec4(c * vig, 1.0);
}"""


# ---------------------------------------------------------------- matrices

def look_at(eye, target, up=(0, 0, 1)):
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = s, u, -f
    M[:3, 3] = -M[:3, :3] @ eye
    return M


def perspective(fovy, aspect, near, far):
    f = 1 / np.tan(np.radians(fovy) / 2)
    M = np.zeros((4, 4))
    M[0, 0], M[1, 1] = f / aspect, f
    M[2, 2], M[2, 3] = (far + near) / (near - far), 2 * far * near / (near - far)
    M[3, 2] = -1
    return M


def ortho(l, r, b, t, n, f):
    M = np.eye(4)
    M[0, 0], M[1, 1], M[2, 2] = 2 / (r - l), 2 / (t - b), -2 / (f - n)
    M[:3, 3] = [-(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n)]
    return M


def light_vp(direction, center, radius):
    eye = center + direction * radius * 3
    up = (0, 1, 0) if abs(direction[2]) > 0.95 else (0, 0, 1)
    return ortho(-radius, radius, -radius, radius, radius * 0.5, radius * 6) @ look_at(eye, center, up)


def _u(prog, name, val):
    if name in prog:
        prog[name].value = tuple(np.asarray(val, np.float32).T.flatten()) if np.ndim(val) == 2 else (
            tuple(map(float, val)) if np.ndim(val) == 1 else float(val))


def nrm(v):
    v = np.asarray(v, float); return v / np.linalg.norm(v)


# ---------------------------------------------------------------- render

DEFAULTS = dict(angles="270,315,225,90", width=900, height=1200, elevation=8.0, fov=22.0, exposure=0.92,
                bg="#FFFFFF,#E6E6E9", ss=2, ao_rays=24)


def render_views(input, **kw):
    """Render a coloured 3MF -> [(azimuth, PIL.Image)]. Keyword args as DEFAULTS. Used by render_3mf.py too."""
    args = argparse.Namespace(input=Path(input), **{**DEFAULTS, **kw})

    import moderngl
    V, F, C = load_model(args.input)
    mesh = trimesh.Trimesh(V, F, process=False)
    N = np.asarray(mesh.vertex_normals, np.float64)
    print(f"{args.input.name}: {len(V):,} verts, {len(F):,} tris; baking AO ...", flush=True)
    ao = bake_ao(V, F, N, args.ao_rays)

    zf = V[:, 2].min()
    center = (V.min(0) + V.max(0)) / 2
    radius = np.linalg.norm(V.max(0) - V.min(0)) / 2
    # unindexed buffers: smooth normals, flat per-face colour
    pos = V[F].reshape(-1, 3) - [center[0], center[1], 0]
    nr = N[F].reshape(-1, 3)
    col = np.repeat(C / 255.0, 3, axis=0)
    aov = ao[F].reshape(-1, 1)
    data = np.hstack([pos, nr, col, aov]).astype("f4")
    c0 = np.array([0, 0, center[2]])

    ctx = moderngl.create_standalone_context()
    ctx.enable(moderngl.DEPTH_TEST)
    W, H = args.width * args.ss, args.height * args.ss
    p_model = ctx.program(vertex_shader=VS_MODEL, fragment_shader=FS_MODEL)
    p_depth = ctx.program(vertex_shader=VS_DEPTH, fragment_shader=FS_DEPTH)
    p_ground = ctx.program(vertex_shader=VS_GROUND, fragment_shader=FS_GROUND)
    p_bg = ctx.program(vertex_shader=VS_BG, fragment_shader=FS_BG)
    vbo = ctx.buffer(data.tobytes())
    vao_m = ctx.vertex_array(p_model, [(vbo, "3f 3f 3f 1f", "in_pos", "in_nrm", "in_col", "in_ao")])
    vao_d = ctx.vertex_array(p_depth, [(vbo, "3f 12x 12x 4x", "in_pos")])
    g = radius * 4
    gquad = ctx.buffer(np.array([[-g, -g], [g, -g], [-g, g], [g, g]], "f4").tobytes())
    vao_g = ctx.vertex_array(p_ground, [(gquad, "2f", "in_xy")])
    bquad = ctx.buffer(np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], "f4").tobytes())
    vao_b = ctx.vertex_array(p_bg, [(bquad, "2f", "in_xy")])

    SM = 4096
    def shadow_fbo():
        t = ctx.depth_texture((SM, SM)); t.compare_func = "<="; t.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return t, ctx.framebuffer(depth_attachment=t)
    key_tex, key_fbo = shadow_fbo()
    top_tex, top_fbo = shadow_fbo()

    # studio rig, fixed relative to the camera (camera at -Y looking +Y, screen-right = +X)
    key_dir = nrm([-0.55, -0.75, 0.85])
    fill_dir = nrm([0.85, -0.55, 0.25])
    rim_dir = nrm([0.35, 0.9, 0.75])
    top_dir = nrm([0.0, 0.0, 1.0])
    bg_top, bg_bot = [np.array(_hex(h)) / 255 for h in args.bg.split(",")]

    fbo = ctx.framebuffer(color_attachments=[ctx.texture((W, H), 3)], depth_attachment=ctx.depth_texture((W, H)))
    views = []
    for az in [float(a) for a in args.angles.split(",") if a.strip()]:
        Rz = trimesh.transformations.rotation_matrix(np.radians(az), [0, 0, 1])
        model = Rz @ trimesh.transformations.translation_matrix(-c0) @ np.eye(4)
        model = trimesh.transformations.translation_matrix(c0) @ model    # rotate about the vertical axis through the centre
        # camera: fit the figurine (as rotated) into the frame, leaving room for the floor shadow
        Pr = (model[:3, :3] @ pos[::7].T).T + model[:3, 3]
        hh = (Pr[:, 2].max() - Pr[:, 2].min()) / 2
        hw = np.abs(Pr[:, 0]).max()
        depth = np.abs(Pr[:, 1]).max()
        t = np.tan(np.radians(args.fov) / 2)
        dist = max(hh * 1.13 / t, hw * 1.10 / (t * W / H)) + depth
        aim = np.array([0, 0, Pr[:, 2].min() + hh * 0.97])
        el = np.radians(args.elevation)
        eye = aim + np.array([0, -np.cos(el), np.sin(el)]) * dist
        view = look_at(eye, aim)
        proj = perspective(args.fov, W / H, dist * 0.2, dist * 4)
        key_vp = light_vp(key_dir, c0, radius * 2.2)   # wide enough to hold the whole cast shadow on the floor
        top_vp = light_vp(top_dir, c0, radius * 1.3)

        ctx.disable(moderngl.BLEND)
        for vp_, fb in ((key_vp, key_fbo), (top_vp, top_fbo)):
            fb.use(); fb.clear(depth=1.0)
            _u(p_depth, "mvp", vp_ @ model)
            vao_d.render(moderngl.TRIANGLES)

        fbo.use(); fbo.clear(1, 1, 1, 1, depth=1.0)
        ctx.disable(moderngl.DEPTH_TEST)
        _u(p_bg, "top", bg_top); _u(p_bg, "bot", bg_bot)
        vao_b.render(moderngl.TRIANGLE_STRIP)
        ctx.enable(moderngl.DEPTH_TEST)

        key_tex.use(0); top_tex.use(1)
        # ground: shadow catcher, multiplied onto the background
        ctx.enable(moderngl.BLEND); ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        for k, v in dict(vp=proj @ view, key_vp=key_vp, top_vp=top_vp, texel=1 / SM, top_texel=1 / SM, soft=3.0,
                         center=c0, radius=radius * 0.6, cast_k=0.30, contact_k=0.42, zf=zf).items():
            _u(p_ground, k, v)
        p_ground["key_map"].value = 0; p_ground["top_map"].value = 1
        vao_g.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)

        for k, v in dict(mvp=proj @ view @ model, model=model, cam_pos=eye, key_dir=key_dir, fill_dir=fill_dir,
                         rim_dir=rim_dir, key_col=[1.00, 0.95, 0.88], fill_col=[0.42, 0.46, 0.52],
                         rim_col=[0.95, 0.95, 1.0], sky_col=[0.62, 0.63, 0.66], gnd_col=[0.40, 0.38, 0.36],
                         key_vp=key_vp, texel=1 / SM, soft=1.6, exposure=args.exposure, spec_k=0.22,
                         shininess=36.0).items():
            _u(p_model, k, v)
        p_model["key_map"].value = 0
        vao_m.render(moderngl.TRIANGLES)

        img = Image.frombytes("RGB", (W, H), fbo.color_attachments[0].read()).transpose(Image.FLIP_TOP_BOTTOM)
        img = img.resize((args.width, args.height), Image.LANCZOS)
        views.append((az, img))
    return views


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--angles", default="270,315,225,90",
                    help="azimuths, same convention as render_3mf.py (default 270,315,225,90)")
    ap.add_argument("--width", type=int, default=900)
    ap.add_argument("--height", type=int, default=1200)
    ap.add_argument("--elevation", type=float, default=8, help="camera elevation in degrees (default 8)")
    ap.add_argument("--fov", type=float, default=22, help="vertical field of view (default 22, telephoto)")
    ap.add_argument("--exposure", type=float, default=0.92)
    ap.add_argument("--bg", default="#FFFFFF,#E6E6E9", help="background gradient top,bottom")
    ap.add_argument("--ss", type=int, default=2, help="supersampling factor (default 2)")
    ap.add_argument("--ao-rays", type=int, default=24)
    ap.add_argument("--separate", action="store_true", help="also save each view as <out>_<angle>.png")
    args = ap.parse_args()
    views = render_views(**{k: v for k, v in vars(args).items() if k not in ("output", "separate")})
    canvas = Image.new("RGB", (args.width * len(views), args.height), "white")
    for i, (az, im) in enumerate(views):
        canvas.paste(im, (i * args.width, 0))
        if args.separate:
            im.save(args.output.with_name(f"{args.output.stem}_{int(az)}{args.output.suffix}"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Wrote {args.output} ({len(views)} views: {', '.join(str(int(a)) for a, _ in views)})")


if __name__ == "__main__":
    main()
