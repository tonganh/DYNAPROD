from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

from ..utils import CACHE, ROOT

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass


def available() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def default_model(second: bool = False) -> str:
    if second:
        return os.environ.get("OPENROUTER_MODEL_2", "google/gemini-2.0-flash-001")
    return os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")


class LLMError(RuntimeError):
    pass


class OpenRouter:
    def __init__(self, model: Optional[str] = None, temperature: float = 0.0, timeout: float = 120.0,
                 cache_dir: Optional[str] = None, max_retries: int = 4):
        self.key = os.environ.get("OPENROUTER_API_KEY")
        if not self.key:
            raise LLMError("OPENROUTER_API_KEY is not set (copy .env.example to .env and fill it in)")
        self.base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.model = model or default_model()
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = cache_dir or os.path.join(CACHE, "llm")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.usage = dict(calls=0, cached=0, prompt_tokens=0, completion_tokens=0, cost_usd=0.0, seconds=0.0)

    def _key(self, payload) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def chat(self, messages: List[Dict[str, Any]], tools: Optional[list] = None, json_mode: bool = False,
             max_tokens: int = 2000) -> Dict[str, Any]:
        payload = dict(model=self.model, messages=messages, temperature=self.temperature, max_tokens=max_tokens)
        if tools:
            payload["tools"] = tools
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        key = self._key(payload)
        path = os.path.join(self.cache_dir, key + ".json")
        if os.path.exists(path):
            with open(path) as f:
                self.usage["cached"] += 1
                return json.load(f)
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                   "X-Title": "DynaProd research code"}
        err = None
        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                r = requests.post(f"{self.base}/chat/completions", headers=headers, json=payload, timeout=self.timeout)
                if r.status_code == 429 or r.status_code >= 500:
                    err = LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                    time.sleep(2 ** attempt)
                    continue
                if r.status_code != 200:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:500]}")
                data = r.json()
                if "choices" not in data:
                    err = LLMError(f"bad response: {str(data)[:300]}")
                    time.sleep(2 ** attempt)
                    continue
                msg = data["choices"][0]["message"]
                u = data.get("usage", {}) or {}
                self.usage["calls"] += 1
                self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
                self.usage["completion_tokens"] += u.get("completion_tokens", 0)
                self.usage["cost_usd"] += float(u.get("cost", 0.0) or 0.0)
                self.usage["seconds"] += time.time() - t0
                out = {"content": msg.get("content"), "tool_calls": msg.get("tool_calls") or []}
                with open(path, "w") as f:
                    json.dump(out, f)
                return out
            except requests.RequestException as e:
                err = LLMError(str(e))
                time.sleep(2 ** attempt)
        raise err or LLMError("request failed")

    def json(self, messages, max_tokens=2000) -> Any:
        out = self.chat(messages, json_mode=True, max_tokens=max_tokens)
        return parse_json(out.get("content") or "")


def parse_json(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            return json.loads(t[i:j + 1])
        raise
