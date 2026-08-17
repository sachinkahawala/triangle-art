"""Prompt construction for the evolution loop.

Kept deliberately small (< ~4k tokens): the contract, 2-3 best programs with
fitness, one weaker program for contrast, short score history, error feedback
from failed siblings, one focused ask. Local models degrade with long context,
so nothing else goes in.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from .config import TaskCfg

SYSTEM_TEMPLATE = """\
You are an expert algorithm designer in an automated code-evolution experiment.

Task: approximate a grayscale target image by placing at most {n_tri} translucent
triangles on an initially white canvas. A fixed harness calls your function once
per triangle slot. A painted triangle fills its area with one gray level
g in [0,1] (0=black, 1=white) at fixed opacity A={alpha}:
    canvas[mask] = canvas[mask] * (1 - A) + g * A
Fitness = RMSE(target, canvas) over the full image, averaged over SEVERAL
different target images. LOWER is better.

Contract:
def propose_triangle(target, canvas, t, n_total, evaluate, rng)
- target: (H, W) float64 in [0,1], the goal image (1=white, 0=black), read-only
- canvas: (H, W) float64 in [0,1], current canvas state, read-only copy
- t: index of this slot (0..n_total-1); n_total = {n_tri}
- evaluate: fn(pts, g=None) -> (delta_sse, pts, g, mask) or None
    pts: (3, 2) array of (x, y) vertex coordinates (floats fine; parts outside
    the canvas are clipped). g=None -> the harness computes the least-squares
    optimal gray for that triangle; pass an explicit g in [0,1] to force one.
    delta_sse: the change in total squared error if painted now (negative = improvement).
    mask: boolean (H, W) array of the triangle's pixels.
    Returns None for degenerate/off-canvas triangles or invalid g (still costs a call).
- rng: seeded numpy Generator — the ONLY allowed randomness source
HARD BUDGET: at most K={k} evaluate() calls per slot; call K+1 raises
BudgetExceeded and if it escapes your function the whole run scores worst-case.
Return the best (delta_sse, pts, g, mask) tuple exactly as evaluate() returned
it, or None to skip this slot. The harness re-verifies your triangle from pts
and g and paints it only if the true delta_sse < 0 — claimed numbers are ignored.

Globals available (no imports needed; ONLY numpy may be imported):
- np (numpy), A = {alpha}, K = {k}, BudgetExceeded

Rules:
- Module-level helpers/constants are allowed; state persists across the {n_tri}
  slots of one image, then is reset before the next image.
- All {n_tri} slots must finish within {timeout:.0f}s per image; never crash.
- Use only `rng` for randomness.
- BANNED: hard-coded absolute coordinates, image-specific constants, or calling
  evaluate on precomputed answers. Fitness is averaged over several images —
  image-specific tricks will not transfer.
- Reply with ONE short paragraph stating your idea and why it should lower RMSE
  (it is logged as your rationale), then exactly ONE ```python block containing
  the COMPLETE replacement program (helpers + propose_triangle).
"""


def system_prompt(task: TaskCfg, timeout_s: float) -> str:
    return SYSTEM_TEMPLATE.format(
        n_tri=task.n_tri, alpha=task.alpha, k=task.k_evals, timeout=timeout_s,
    )


def _score_table(history_rows: list[dict], limit: int = 10) -> str:
    rows = history_rows[-limit:]
    lines = ["gen | best mean RMSE | accepted"]
    for r in rows:
        mark = "yes" if r["accepted"] else "no (kept best)"
        lines.append(f"{r['gen']:3d} | {r['best_score']:.4f} | {mark}")
    return "\n".join(lines)


def render_to_png_b64(render: np.ndarray, target: np.ndarray, px: int = 160) -> str:
    """Side-by-side (render | target) grayscale PNG as base64."""
    def to_img(a: np.ndarray) -> Image.Image:
        img = Image.fromarray((a * 255).clip(0, 255).astype(np.uint8))
        return img.resize((px, px), Image.LANCZOS)

    combo = Image.new("L", (px * 2 + 8, px), 255)
    combo.paste(to_img(render), (0, 0))
    combo.paste(to_img(target), (px + 8, 0))
    buf = io.BytesIO()
    combo.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _program_block(label: str, entry: dict) -> str:
    per_img = ", ".join(f"{k}: {v:.4f}" for k, v in entry["per_image"].items())
    placed = entry.get("placed")
    placed_txt = (f"; triangles placed: "
                  + ", ".join(f"{k}: {v}" for k, v in placed.items())
                  if placed else "")
    return (f"{label} (mean RMSE {entry['score']:.4f}; per image: {per_img}"
            f"{placed_txt}):\n```python\n{entry['code']}```")


def generation_prompt(
    parents: list[dict],          # best-first: [{code, score, per_image, placed?}]
    weaker: dict | None,          # one lower-ranked program for contrast
    history_rows: list[dict],
    sibling_errors: list[str],
    mode: str,                    # "normal" | "bold" | "restart"
    vision_b64: str | None,
) -> list[dict]:
    parts = []
    labels = ["Best program", "Second-best program", "Third-best program"]
    for label, entry in zip(labels, parents):
        parts.append(_program_block(label, entry))
    if weaker is not None:
        parts.append(_program_block(
            "A WEAKER program for contrast (avoid what makes it lose)", weaker))
    parts.append("Recent score history (lower RMSE is better):")
    parts.append(_score_table(history_rows))
    if sibling_errors:
        parts.append(
            "Failed attempts last generation (do not repeat these mistakes):\n"
            + "\n".join(f"- {e}" for e in sibling_errors[:3]))
    if mode == "restart":
        parts.append(
            "RESTART: ignore the incumbent designs above. Start from the simple "
            "baseline structure and take a structurally different approach — a "
            "fresh idea beats a micro-tuned copy. Complete working program only.")
    elif mode == "bold":
        parts.append(
            "The score has PLATEAUED. Do not micro-tune: propose a structurally "
            "different strategy (where triangles go, how their shape/size is "
            "chosen, how the K evaluation budget is split between exploring new "
            "triangles and refining the best one, how gray is chosen...). "
            "Still a complete working program.")
    else:
        parts.append(
            "Propose ONE focused improvement you predict will LOWER mean RMSE. "
            "State the idea in a one-paragraph rationale, then give the complete "
            "replacement program.")
    text = "\n\n".join(parts)

    msg: dict = {"role": "user", "content": text}
    if vision_b64 is not None:
        msg["content"] = (
            text + "\n\nAttached: current best canvas (left) vs target (right) "
                   "for one training image. Look at where they differ.")
        msg["images"] = [vision_b64]
    return [msg]


RETRY_NUDGE = (
    "Your reply did not contain a usable ```python block defining "
    "propose_triangle. Reply again with one short rationale paragraph and "
    "exactly one ```python block containing the complete program."
)
