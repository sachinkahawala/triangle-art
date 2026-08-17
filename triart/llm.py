"""Local LLM client over Ollama's native API. (Lifted from the string-art
harness; only the code-extraction marker changed.)

The native /api/chat endpoint is used (rather than the OpenAI-compat layer)
because it supports `think: false` properly — reasoning tokens are the
difference between a 90-second and a 4-minute candidate on a 27B local model.

Message format: {"role": ..., "content": str, "images": [base64-png, ...]?}
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass

from .config import ModelCfg

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


@dataclass
class GenResult:
    text: str
    code: str | None
    wall_s: float
    completion_tokens: int | None = None
    tok_s: float | None = None


def extract_code(text: str) -> str | None:
    """Last fenced block that defines propose_triangle (falls back to last block)."""
    blocks = _FENCE_RE.findall(text)
    if not blocks:
        return None
    with_fn = [b for b in blocks if "def propose_triangle" in b]
    return (with_fn[-1] if with_fn else blocks[-1]).strip() + "\n"


def extract_rationale(text: str, limit: int = 700) -> str:
    """First paragraph outside any code fence — the model's stated idea."""
    prose = re.split(r"```", text, maxsplit=1)[0].strip()
    for para in re.split(r"\n\s*\n", prose):
        para = " ".join(para.split())
        if para:
            return para[:limit]
    return ""


class LLM:
    def __init__(self, cfg: ModelCfg):
        self.cfg = cfg
        # config carries the OpenAI-style endpoint; native root is its parent
        self.root = cfg.endpoint.removesuffix("/v1").rstrip("/")
        self.model = self._resolve_model()

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.root + path, timeout=30) as r:
            return json.loads(r.read())

    def _post(self, path: str, body: dict, timeout: float = 1200.0) -> dict:
        req = urllib.request.Request(
            self.root + path, json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())

    def _resolve_model(self) -> str:
        try:
            tags = self._get("/api/tags")
        except Exception as e:
            raise RuntimeError(
                f"cannot reach Ollama at {self.root} — is it running? ({e})"
            ) from e
        available = {m["name"] for m in tags.get("models", [])}
        for name in [self.cfg.name, *self.cfg.fallbacks]:
            if name in available:
                return name
        for name in [self.cfg.name, *self.cfg.fallbacks]:
            base = name.split(":")[0]
            for avail in sorted(available):
                if avail.split(":")[0] == base:
                    return avail
        raise RuntimeError(
            f"none of {[self.cfg.name, *self.cfg.fallbacks]} installed; "
            f"server has: {sorted(available)}"
        )

    def generate(self, messages: list[dict], temperature: float) -> GenResult:
        body = {
            "model": self.model,
            "messages": messages,
            "think": bool(self.cfg.thinking),
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.95 if self.cfg.thinking else 0.8,
                "top_k": 20,
                "num_predict": self.cfg.max_tokens,
            },
        }
        t0 = time.monotonic()
        resp = self._post("/api/chat", body)
        wall = time.monotonic() - t0
        text = _THINK_RE.sub("", resp["message"].get("content") or "").strip()
        ec = resp.get("eval_count")
        ed = resp.get("eval_duration")
        return GenResult(
            text=text,
            code=extract_code(text),
            wall_s=wall,
            completion_tokens=ec,
            tok_s=(ec / (ed / 1e9)) if ec and ed else None,
        )
