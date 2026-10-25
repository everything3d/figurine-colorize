# figurine-colorize

A [Claude Code](https://claude.com/claude-code) skill that turns **one image of a figurine into a
multi-colour 3D print file**: every region already assigned to a filament, ready for an AMS / multi-toolhead
printer.

<!-- before/after: drop an image at docs/before-after.png and uncomment
![input image → coloured 3MF in the slicer](docs/before-after.png)
-->

You give Claude an image and say *"make this a 4-colour 3MF, 120 mm tall"*. It:

1. **Generates the 3D model** with [Tripo](https://www.tripo3d.ai): a textured model for colour and a
   watertight one for printing.
2. **Colours it with exactly N filaments.** The colour edges follow the sculpted edges, and shadows don't
   turn black.
3. **Paints the face from the sculpt:** eye whites on the actual eyeballs, irises, catchlights, eyeliner
   and lips, and brows and beards that cover the whole raised shape.
4. **Renders it from 5 angles** so you can check it before you print.

You get a `.3mf` that opens in Bambu Studio or Orca Slicer with the colours already on.

## Install

You need [Claude Code](https://claude.com/claude-code), Python 3.10+, and a
[Tripo API key](https://platform.tripo3d.ai) (generating a model uses Tripo credits; everything else runs
on your computer for free).

```bash
git clone https://github.com/apoorvumang/figurine-colorize ~/.claude/skills/figurine-colorize
python3 -m venv ~/.venvs/figurine
~/.venvs/figurine/bin/pip install -r ~/.claude/skills/figurine-colorize/requirements.txt
export TRIPO_API_KEY=tsk_your_key_here
```

Then open Claude Code anywhere (with `source ~/.venvs/figurine/bin/activate` run first) and ask:

> Turn `cat.png` into a 4-colour 3MF, 100 mm tall

Claude picks up the skill on its own. It will ask before spending Tripo credits.

## Tips for the input image

- **Stylised beats realistic.** A Pixar, Funko or chibi-style render of a person prints and colours far
  better than a photo. Make the render first with any image model, then feed it in.
- One character, full body, front view, plain white background, PNG.
- Bold, flat colours. Every distinct colour costs a filament slot, and white eyes and red lips need slots
  too.

## What's in here

| | |
|---|---|
| `SKILL.md` | The instructions Claude follows: the steps, settings, and fixes for common problems |
| `scripts/gen_textured_for_color.py` | Generates the model with Tripo |
| `scripts/clean_colorize.py` | Colours the model with N filaments |
| `scripts/paint_features.py`, `sculpt_features.py` | Paints the eyes, lips and brows |
| `scripts/render_3mf.py`, `render_pretty.py` | Check renders |
| `scripts/colorize_3mf.py` | Older colouring method and shared helpers |

You can also run the scripts by hand without Claude. Every script takes `--help`.

## License

MIT
