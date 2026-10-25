---
name: figurine-colorize
description: Turn a figurine image into a multi-colour, slicer-ready 3MF with N filament colours (one per AMS / toolhead slot). Use whenever the user wants a "colored 3mf", "multi-color figurine", "4 colors / 5 colors", an MMU / AMS print, or a printable multi-material model from a character or figurine image.
---

# Figurine Colorize

One image of a figurine → a watertight, N-colour 3MF that Bambu Studio / Orca opens with every region
already assigned to a filament. Eyes, brows and lips get painted from the sculpted shape, not
guessed from a blurry texture.

```
image.png ──Tripo──▶ textured GLB (colour)  ┐
                     watertight 3MF (shape) ┴─▶ clean_colorize ─▶ paint_features ─▶ render_3mf
                                                 (N flat colours)   (eyes, brows, lips)  (check it)
```

## Setup (once)

The scripts sit next to this file, in `scripts/`. Below, `S` is that folder, e.g.
`S=~/.claude/skills/figurine-colorize/scripts` (use this skill's actual base directory).

```bash
python3 -m venv ~/.venvs/figurine && source ~/.venvs/figurine/bin/activate
pip install -r "$S/../requirements.txt"
export TRIPO_API_KEY=tsk_...          # from https://platform.tripo3d.ai
```

`embreex` and `moderngl` are optional. Without `embreex` the shadow fix is skipped; without `moderngl` the
check renders are flat-shaded. Both install fine on macOS and Linux with Python 3.10-3.13.

## Hard rules
- **Ask before running Tripo.** Phase 1 spends the user's Tripo credits. Confirm the image is the one they
  want, then run it once per image. Everything after Phase 1 is free and local, so iterate there.
- **Never judge colour from `colorize_3mf.py --preview`.** It fakes salt-and-pepper speckle that is not in
  the model. Use `render_3mf.py` (Phase 5).
- Keep one folder per job: `<job>/input.png`, `<job>/models/`. Write the palette and the exact commands
  you ran into `<job>/README.md`. It is the only record of the tuning.

## The input image
Best results: one character, full body, front view, plain white background, stylised (Pixar / Funko /
chibi look), clean flat colours, PNG. Glossy renders work but give light rims (Phase 4). Photos of real
people work much worse than a stylised render of them, so render the photo first with any image model.

## Phase 1 — Tripo, twice, off the same task (costs credits)

```bash
python3 $S/gen_textured_for_color.py <job>/input.png <job>/models
```

This makes two calls off one model task:
1. `image_to_model` with `texture=True, pbr=False` → `<n>_textured.glb`, the **colour source**. Its mesh is
   Tripo's raw one (many shells, open edges), useless for printing. `pbr=False` matters: PBR bakes
   lighting into the texture and makes flat colours much harder.
2. `convert_model` to `3mf` from the same task → `<n>_geometry.3mf`, the **shape**: repaired, watertight,
   one body, but flat grey.

Both must come from the same task so they share a coordinate frame. Task ids are saved to
`<n>_tasks.json` the moment they're submitted; if anything dies, run the same command again and it
resumes without paying twice. Takes 2-4 min per model (longer for groups).

## Phase 2 — choose the palette

N filament hex colours, N = the user's slot count. Pick them by eye from the image, or seed with k-means
on the image (white background dropped) and then clean up: raw cluster means come back shading-averaged
and muddy. Aim for colours that exist as real filaments.

Spend slots on regions, not detail. A 4-colour person is typically skin / hair-and-dark / top / bottom.
White eyes and red lips each cost a slot. If something in the image has no slot left, say so; don't
silently drop it.

## Phase 3 — colour it

```bash
python3 $S/clean_colorize.py <job>/models/<n>_textured.glb \
  --geometry <job>/models/<n>_geometry.3mf \
  -o <job>/models/<n>_clean.3mf \
  --palette "#RRGGBB,#RRGGBB,..." --height 120 --up-axis y --geometry-up-axis z --obj
```

`clean_colorize` labels the whole surface at once (a graph cut). Every face pays for its distance from the
texture colour; every colour boundary pays for its length, and boundaries are cheap along sculpted creases.
So colour edges land on the sculpted edges, there's no band of a third colour between two regions, and
specks get absorbed.

It also removes baked-in shadow: it computes its own ambient occlusion, measures how much darker each
palette colour gets in shadow, and divides that back out. Then any leftover dark patch sitting in deep
shadow inside a lighter colour (black under arms and collars, dark shoe soles) is re-solved without its
own colour. This is what stops "black wherever there's a shadow".

`--height` is the print height in mm on Z. The model is centred on X/Y and sat on Z=0.

Knobs:
- `--lam` (6) — boundary cost. Higher = flatter regions; above ~10 thin features (irises) start vanishing.
  Lower (3) keeps thin stripes.
- `--sigma` (25°) — how sharp a crease must be before a boundary on it gets cheap.
- `--min-island` (0.4 mm²) — smallest island kept.
- `--black-max-chroma 14` — only when your darkest slot really is black: black is then only allowed where
  the texture is genuinely neutral, so shaded coloured clothing can't go black.
- Boxes, in output mm (`x0:x1,y0:y1,z0:z1`, measure them on the written 3MF):
  - `--only "#HEX@box"` — a colour allowed only inside the box (e.g. red only on the mouth).
  - `--forbid "#HEX@box"` — a colour banned inside the box (e.g. no trouser colour above the waist).
  - `--prefer "#HEX@box=12"` — a nudge (e.g. make both earrings match).

`colorize_3mf.py` is the older per-vertex method (same inputs, same output format). It's the fallback if
PyMaxflow won't install, and its knobs (Phase 4) are useful for diagnosis.

## Phase 4 — when a region goes wrong

Start with no extra flags. Add one only when the render (Phase 5) or the printed colour percentages show
a problem.

| Symptom | Fix |
|---|---|
| Dark creases (chin, armpits, folds) going black | `clean_colorize` handles this by default. With `colorize_3mf.py`: `--black-chroma 8` |
| A light rim along every hairline / brow / eye | Glossy source art: baked highlights read as white. `colorize_3mf.py --open "#<lightest>=6"` removes bands thinner than 12 faces. Or raise `--lam` |
| A shaded light region bleeding into a warm colour (white → gold) | `colorize_3mf.py --color-space lab --luma-weight 0.35` |
| White body + black detail speckling into each other | `colorize_3mf.py --dark-below 20 --color-space lab` (always with lab, or mid-greys snap to random colours) |
| Shoes / base the wrong colour | `--forbid` box on clean_colorize, or `colorize_3mf.py --floor-color <hex> --floor-frac 0.05` |
| One region simply the wrong colour | a `--forbid` / `--only` box, not a new palette |

**Preflight for black bleed:** sort the palette by lightness. If black is much darker than everything else
(a gap of 30+ in Lab L), expect shadows to bleed into it. Saturated outfits and black-and-white subjects
are naturally immune.

## Phase 4b — the face pass (every model with a face)

Tripo sculpts the eyes, brows and lips, but the texture over them is blurry. This pass repaints them from
the geometry:

- **eye white** = exactly the sculpted eyeball (it's a sphere); **iris** ~1.1× the opening, clipped by the
  lids; a small white **catchlight** up-right.
- **women:** one solid **liner** band over the whole upper lid with a small wing, and **lips** as one smooth
  red lens over both lip bulges. **Men:** no liner, no lip colour.
- **brows / beard / moustache:** dark paint grown out to the crease at the foot of the raised shape, so no
  skin is left on top.
- **shadow patches** in dark colours repainted to the colour around them.

```bash
# 1. find the faces; prints them numbered left to right in the front view
python3 $S/paint_features.py <n>_clean.3mf -o <n>_final.3mf --auto-face --refine 2
# 2. re-run saying which faces are women (liner + lips) and the lip filament
python3 $S/paint_features.py <n>_clean.3mf -o <n>_final.3mf --auto-face --women 0 --lip "#CE0000" --refine 2
```

The eye white, iris, skin and dark colours must be in the palette. `--auto-face` writes the shapes it
used to `<n>_final.features.txt`. If one face comes out wrong, copy that file, fix the line, and re-run
with `--spec <file>` instead of `--auto-face`:

```
shadowfix:#BLACK,#SKIN                       # shadow islands -> the lighter colour around them
eye3d:#WHITE,#IRIS,#SKIN@u,v                 # a man's eye: seed = any point on the eyeball
eye3d:#WHITE,#IRIS,#SKIN@u,v,liner           # a woman's eye, with liner
grow:#BLACK>#SKIN@u0,u1,v0,v1                # brows/beard out to their sculpted foot (window optional)
lips3d:#RED,#SKIN@u1,v1;u2,v2|keep=#WHITE+#BLACK|teeth=#WHITE   # mouth line corner to corner
```

`u,v` are front-view mm. To read coordinates off a face, render a gridded preview:

```bash
python3 $S/paint_features.py <n>_clean.3mf --view 270 --preview face.png --window=-15,15,95,125
```

(`--window=` needs the `=` because a negative first number otherwise parses as a flag.)

Limits: `eye3d` reports `NO sculpted eyeball there` when an eye is only painted on (very realistic or very
small eyes, eyes behind glasses). For those, use the 2D shapes `eye:`, `stroke:`, `poly:`, `lips:` —
`python3 $S/paint_features.py --help` lists them. It can't invent a feature the texture doesn't have.

Mind the print scale: a 0.4 mm liner printed at 70% is thinner than a nozzle.

## Phase 5 — check it (don't skip)

```bash
python3 $S/render_3mf.py <n>_final.3mf -o <n>_views.png
```

Renders the actual file from 5 angles, the way the slicer will show it, in studio lighting (`--flat` for
plain shading when judging a colour edge). Compare it against the input image. Also read the colour
percentages the scripts print: a necklace has no business being 12% of the faces.

Tripo invents the back of the model, which it never saw. Say so if the back looks off.

## Phase 6 — final checks and handoff

```python
import trimesh
s = trimesh.load(path, process=False)        # process=False: load exactly as written
m = s.to_geometry() if isinstance(s, trimesh.Scene) else s
print(m.extents, m.is_watertight, m.body_count)
```

Always load with `process=False`; `process=True` re-welds vertices and reports false holes. Don't chase
sub-millimetre mesh defects: the slicer repairs them on import and they never matter at a 0.4 mm nozzle.
Real defects in this pipeline are always in the colour, not the geometry.

Hand over: file path, height, size W×D×H, watertight / body count, and the filament hex list in slot
order so the user can load the right spools. Send the `_views.png`. Call out any colour the slot budget
couldn't cover.

**In the slicer:** open the `.3mf` in Bambu Studio or Orca and map each colour to a filament slot. The
`.obj` + `.mtl` (from `--obj`) also imports with crisp colour borders if the 3MF ever doesn't show colours.
Expect to touch up a few spots with the slicer's paint tool; the goal here is clean regions, not a
finished paint job.

## Script reference

| Script | Phase | What it does |
|---|---|---|
| `gen_textured_for_color.py <image> <out_dir>` | 1 | Both Tripo calls → `<n>_textured.glb` + `<n>_geometry.3mf` |
| `clean_colorize.py <glb> --geometry <3mf> -o <out>` | 3 | Graph-cut colouring with shadow removal (default) |
| `colorize_3mf.py <glb> --geometry <3mf> -o <out>` | 3/4 | Older per-vertex colouring + diagnosis knobs |
| `paint_features.py <3mf> -o <out> --auto-face` | 4b | Eyes, brows, lips, shadow patches from the sculpt |
| `sculpt_features.py` | 4b | Library behind the face pass (finding eyes, lips, creases) |
| `render_3mf.py <3mf> -o <png>` | 5 | Check render of the actual file |
| `render_pretty.py <3mf> -o <png>` | 5 | Hero shot / ambient occlusion helper |

All of them take `--help`.
