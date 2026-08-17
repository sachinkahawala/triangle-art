"""Build all deliverables from a run directory (self-contained: needs only
state.json, log.jsonl, and the per-generation PNG/py files).

    uv run python report/make_report.py runs/<name>

Outputs into runs/<name>/report/:
  filmstrip.png    best hero canvas per accepted generation, noise -> image
  fitness.png      best-so-far mean-RMSE curve (lower = better), candidate
                   scatter, holdout overlay, breakthrough annotations
  progression.png  champion canvas at the n/8, n/4, n/2, n triangle snapshots
  evolution.gif    animated hero canvas across accepted generations
  diffs.md         unified code diffs between successive accepted programs
  rationales.md    the model's stated rationale per accepted generation
  report.html      self-contained single page bundling everything
"""
from __future__ import annotations

import argparse
import base64
import difflib
import html as html_mod
import json
import textwrap
from pathlib import Path

import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Validated reference palette (dataviz skill), light mode.
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE_C = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES_BEST = "#2a78d6"   # slot 1 blue
SERIES_HOLD = "#eb6834"   # slot 2 orange

BREAKTHROUGH_REL = 0.015  # accepted drop > 1.5% of previous best RMSE


def load_run(run_dir: Path) -> dict:
    state = json.loads((run_dir / "state.json").read_text())
    rows = [json.loads(l) for l in (run_dir / "log.jsonl").read_text().splitlines()]
    return {"state": state, "rows": rows}


def accepted_gens(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("accepted")]


def accepted_rationale(row: dict) -> str:
    if row.get("rationale"):
        return row["rationale"]
    cands = [c for c in (row.get("candidates") or [])
             if c.get("score") is not None]
    if not cands:
        return ""
    best = min(cands, key=lambda c: c["score"])
    return best.get("rationale", "")


def find_breakthroughs(rows: list[dict]) -> list[dict]:
    out = []
    prev = None
    for r in rows:
        cur = r["best_score_after"]
        if prev is not None and r.get("accepted"):
            if prev - cur > abs(prev) * BREAKTHROUGH_REL:
                out.append({"gen": r["gen"], "score": cur, "delta": prev - cur,
                            "idea": accepted_rationale(r)})
        prev = cur
    return out


# ---------------------------------------------------------------- filmstrip

def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def make_filmstrip(run_dir: Path, rows: list[dict], out: Path, tile: int = 168) -> None:
    frames = []
    for r in accepted_gens(rows):
        png = run_dir / f"gen_{r['gen']:03d}" / "best_hero.png"
        if png.exists():
            frames.append((r["gen"], r["best_score_after"], png))
    if not frames:
        return
    cols = min(8, len(frames))
    rows_n = (len(frames) + cols - 1) // cols
    cap_h = 26
    W, H = cols * tile + (cols + 1) * 8, rows_n * (tile + cap_h + 8) + 8
    sheet = Image.new("RGB", (W, H), SURFACE)
    draw = ImageDraw.Draw(sheet)
    font = _font(13)
    for k, (gen, score, png) in enumerate(frames):
        r_, c_ = divmod(k, cols)
        x = 8 + c_ * (tile + 8)
        y = 8 + r_ * (tile + cap_h + 8)
        img = Image.open(png).convert("L").resize((tile, tile), Image.LANCZOS)
        sheet.paste(img, (x, y))
        draw.text((x + tile // 2, y + tile + 6), f"gen {gen} · {score:.4f}",
                  fill=INK_2, font=font, anchor="ma")
    sheet.save(out)


# ---------------------------------------------------------------- progression

def make_progression(run_dir: Path, state: dict, out: Path, tile: int = 200) -> None:
    """Champion canvas at the snapshot slots (hero image)."""
    gen = state["best"]["gen_found"]
    gen_dir = run_dir / f"gen_{gen:03d}"
    snaps = sorted(gen_dir.glob("hero_s*.png"))
    if not snaps:
        return
    cap_h = 24
    W = len(snaps) * tile + (len(snaps) + 1) * 8
    H = tile + cap_h + 16
    sheet = Image.new("RGB", (W, H), SURFACE)
    draw = ImageDraw.Draw(sheet)
    font = _font(13)
    for k, png in enumerate(snaps):
        slot = int(png.stem.split("_s")[1])
        x = 8 + k * (tile + 8)
        img = Image.open(png).convert("L").resize((tile, tile), Image.LANCZOS)
        sheet.paste(img, (x, 8))
        draw.text((x + tile // 2, tile + 12), f"{slot} triangles",
                  fill=INK_2, font=font, anchor="ma")
    sheet.save(out)


# ---------------------------------------------------------------- fitness curve

def make_fitness_curve(rows: list[dict], out: Path) -> None:
    gens = [r["gen"] for r in rows]
    best = [r["best_score_after"] for r in rows]
    hold = [(r["gen"], float(np.mean(list(r["holdout"].values()))))
            for r in rows if r.get("holdout")]
    cand_pts = [(r["gen"], c["score"]) for r in rows
                for c in (r.get("candidates") or []) if c.get("score") is not None]

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    if cand_pts:
        xs, ys = zip(*cand_pts)
        ax.scatter(xs, ys, s=14, color=MUTED, alpha=0.55, zorder=2,
                   label="candidate attempts", edgecolors="none")
    ax.step(gens, best, where="post", color=SERIES_BEST, lw=2, zorder=3,
            label="best so far (train mean)")
    if hold:
        hx, hy = zip(*hold)
        ax.plot(hx, hy, color=SERIES_HOLD, lw=2, zorder=3, marker="o", ms=4,
                label="held-out images (never shown to the model)")

    span = (max(best) - min(best)) or 1.0
    for b in find_breakthroughs(rows):
        idea = textwrap.shorten(b["idea"] or "", 46, placeholder="…")
        ax.annotate(f"gen {b['gen']}: -{b['delta']:.4f}\n{idea}",
                    xy=(b["gen"], b["score"]),
                    xytext=(b["gen"], b["score"] - span * 0.12),
                    fontsize=7.5, color=INK_2, ha="center", va="top",
                    arrowprops=dict(arrowstyle="-", color=BASELINE_C, lw=0.8))

    ax.set_xlabel("generation", color=INK_2)
    ax.set_ylabel("mean RMSE (lower is better)", color=INK_2)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE_C)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncols=3,
                    fontsize=8, frameon=False, borderaxespad=0)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- gif

def make_gif(run_dir: Path, rows: list[dict], out: Path, px: int = 320) -> None:
    frames = []
    for r in accepted_gens(rows):
        png = run_dir / f"gen_{r['gen']:03d}" / "best_hero.png"
        if not png.exists():
            continue
        img = Image.open(png).convert("L").resize((px, px), Image.LANCZOS)
        frame = Image.new("L", (px, px + 24), 255)
        frame.paste(img, (0, 0))
        ImageDraw.Draw(frame).text(
            (px // 2, px + 4), f"gen {r['gen']}  RMSE {r['best_score_after']:.4f}",
            fill=60, font=_font(13), anchor="ma")
        frames.append(np.asarray(frame))
    if not frames:
        return
    frames += [frames[-1]] * 4  # hold the final canvas
    iio.imwrite(out, frames, duration=500, loop=0)


# ---------------------------------------------------------------- diffs

def make_diffs(run_dir: Path, rows: list[dict], out: Path) -> str:
    acc = accepted_gens(rows)
    chunks = []
    bts = {b["gen"] for b in find_breakthroughs(rows)}
    for prev, cur in zip(acc[:-1], acc[1:]):
        a_file = _accepted_file(run_dir, prev)
        b = _accepted_file(run_dir, cur)
        if a_file is None or b is None:
            continue
        diff = "".join(difflib.unified_diff(
            a_file.read_text().splitlines(keepends=True),
            b.read_text().splitlines(keepends=True),
            fromfile=f"gen_{prev['gen']}", tofile=f"gen_{cur['gen']}"))
        tag = " **breakthrough**" if cur["gen"] in bts else ""
        idea = accepted_rationale(cur)
        chunks.append(
            f"## gen {prev['gen']} → gen {cur['gen']}"
            f" ({prev['best_score_after']:.4f} → {cur['best_score_after']:.4f})"
            f"{tag}\n\nRationale: {idea}\n\n```diff\n{diff}```\n")
    text = "# Code evolution — accepted diffs\n\n" + "\n".join(chunks)
    out.write_text(text)
    return text


def _accepted_file(run_dir: Path, row: dict) -> Path | None:
    gen = row["gen"]
    if gen == 0:
        p = run_dir / "gen_000" / "cand_0.py"
        return p if p.exists() else None
    cands = [c for c in (row.get("candidates") or []) if c.get("score") is not None]
    if not cands:
        return None
    idx = min(cands, key=lambda c: c["score"])["idx"]
    p = run_dir / f"gen_{gen:03d}" / f"cand_{idx}.py"
    return p if p.exists() else None


# ---------------------------------------------------------------- rationales

def make_rationales(rows: list[dict], out: Path) -> str:
    lines = ["# Rationale log — the model's stated idea per accepted generation\n"]
    for r in accepted_gens(rows):
        idea = accepted_rationale(r)
        if r["gen"] == 0:
            idea = idea or "naive random-search baseline"
        lines.append(f"## gen {r['gen']} — mean RMSE {r['best_score_after']:.4f}\n")
        lines.append((idea or "(no rationale given)") + "\n")
    text = "\n".join(lines)
    out.write_text(text)
    return text


# ---------------------------------------------------------------- html

def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def make_html(run_dir: Path, state: dict, rows: list[dict], rpt: Path,
              diffs_md: str, rationales_md: str) -> None:
    best = state["best"]
    cfgc = state["config"]
    n_gens = rows[-1]["gen"] if rows else 0
    base_score = rows[0]["best_score_after"] if rows else 0
    holdout = best.get("holdout") or {}
    hold_txt = ", ".join(f"{k}: {v:.4f}" for k, v in holdout.items()) or "n/a"
    placed = best.get("placed") or {}
    placed_txt = ", ".join(f"{k}: {v}/{cfgc['task']['n_tri']}"
                           for k, v in placed.items()) or "n/a"

    imgs = {}
    for name in ("fitness.png", "filmstrip.png", "progression.png",
                 "evolution.gif"):
        p = rpt / name
        if p.exists():
            mime = "image/gif" if name.endswith("gif") else "image/png"
            imgs[name] = f"data:{mime};base64,{_b64(p)}"

    diff_html = html_mod.escape(diffs_md)
    rat_html = html_mod.escape(rationales_md)
    code_html = html_mod.escape(best["code"])
    rows_html = "\n".join(
        f"<tr><td>{r['gen']}</td><td>{r.get('mode', '')}</td>"
        f"<td>{r['best_score_after']:.4f}</td>"
        f"<td>{'yes' if r.get('accepted') else ''}</td>"
        f"<td>{html_mod.escape(textwrap.shorten(accepted_rationale(r), 110) if r.get('accepted') else '')}</td></tr>"
        for r in rows)

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Triangle Placement Evolution — {run_dir.name}</title>
<style>
  body {{ background:#f9f9f7; color:#0b0b0b; font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
         max-width:980px; margin:2rem auto; padding:0 1rem; }}
  h1,h2 {{ font-weight:650; }} h1 {{ font-size:1.6rem; }} h2 {{ font-size:1.15rem; margin-top:2.2rem; }}
  .stats {{ display:flex; gap:1rem; flex-wrap:wrap; margin:1.2rem 0; }}
  .tile {{ background:#fcfcfb; border:1px solid rgba(11,11,11,.10); border-radius:10px;
           padding:.8rem 1.1rem; min-width:140px; }}
  .tile .v {{ font-size:1.5rem; font-weight:650; }} .tile .l {{ color:#52514e; font-size:.8rem; }}
  img {{ max-width:100%; border:1px solid rgba(11,11,11,.10); border-radius:8px; background:#fcfcfb; }}
  pre {{ background:#fcfcfb; border:1px solid rgba(11,11,11,.10); border-radius:8px;
         padding:1rem; overflow-x:auto; font-size:12.5px; }}
  table {{ border-collapse:collapse; font-size:13px; width:100%; }}
  td,th {{ border-bottom:1px solid #e1e0d9; padding:.35rem .6rem; text-align:left; }}
  th {{ color:#52514e; }} td:nth-child(3) {{ font-variant-numeric: tabular-nums; }}
</style></head><body>
<h1>Triangle Placement Evolution — local LLM auto-research</h1>
<p>Model <b>{cfgc['model']['name']}</b> evolved <code>propose_triangle()</code> over
{n_gens} generations · {cfgc['task']['n_tri']} triangles · K={cfgc['task']['k_evals']}
evaluations/slot · A={cfgc['task']['alpha']} ·
train images: {', '.join(cfgc['train_images'])} · held-out: {', '.join(cfgc['holdout_images'])}</p>
<div class="stats">
 <div class="tile"><div class="v">{base_score:.4f} → {best['score']:.4f}</div><div class="l">mean RMSE, gen 0 → best (lower is better)</div></div>
 <div class="tile"><div class="v">{best['gen_found']}</div><div class="l">generation of best program</div></div>
 <div class="tile"><div class="v">{placed_txt}</div><div class="l">triangles placed / budget</div></div>
 <div class="tile"><div class="v">{hold_txt}</div><div class="l">held-out RMSE (unseen images)</div></div>
</div>
<h2>Fitness over generations</h2><img src="{imgs.get('fitness.png', '')}" alt="fitness curve">
<h2>Filmstrip — best canvas per accepted generation</h2><img src="{imgs.get('filmstrip.png', '')}" alt="filmstrip">
<h2>Champion progression — canvas as triangles accumulate</h2><img src="{imgs.get('progression.png', '')}" alt="progression">
<h2>Evolution</h2><img src="{imgs.get('evolution.gif', '')}" alt="evolution gif">
<h2>Winning program</h2><pre>{code_html}</pre>
<h2>Generation table</h2>
<table><tr><th>gen</th><th>mode</th><th>best after</th><th>accepted</th><th>rationale</th></tr>{rows_html}</table>
<h2>Rationale log</h2><pre>{rat_html}</pre>
<h2>Accepted code diffs</h2><pre>{diff_html}</pre>
</body></html>"""
    (rpt / "report.html").write_text(html)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()

    run_dir = args.run_dir.resolve()
    data = load_run(run_dir)
    rpt = run_dir / "report"
    rpt.mkdir(exist_ok=True)

    make_filmstrip(run_dir, data["rows"], rpt / "filmstrip.png")
    make_fitness_curve(data["rows"], rpt / "fitness.png")
    make_progression(run_dir, data["state"], rpt / "progression.png")
    make_gif(run_dir, data["rows"], rpt / "evolution.gif")
    diffs_md = make_diffs(run_dir, data["rows"], rpt / "diffs.md")
    rationales_md = make_rationales(data["rows"], rpt / "rationales.md")
    make_html(run_dir, data["state"], data["rows"], rpt, diffs_md, rationales_md)
    print(f"report written to {rpt}")
    for f in sorted(rpt.iterdir()):
        print(" ", f.name)


if __name__ == "__main__":
    main()
