"""Provider-agnostic chat client: OpenAI or OpenRouter.

Configuration comes from environment variables (fail closed when absent):

    EVOQUANT_LLM_PROVIDER   "openai" | "openrouter"
    OPENAI_API_KEY          when provider=openai
    OPENROUTER_API_KEY      when provider=openrouter
    EVOQUANT_LLM_MODEL      model id (e.g. "gpt-4o-mini",
                            "meta-llama/llama-3.3-70b-instruct:free")

The HTTP transport is injectable so unit tests exercise the full
request/response path without any network. Keys are never logged and
never persisted by this module; ``redacted()`` is the only key
representation allowed in artifacts.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from evoquant.errors import EvoquantError, MissingMetadataError

PROVIDERS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}

#: transport(url, headers, body_bytes, timeout_s) -> (status_code, body_text)
Transport = Callable[[str, dict[str, str], bytes, float], tuple[int, str]]


class LLMClientError(EvoquantError):
    code = "LLM_CLIENT"


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    model: str
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if self.provider not in PROVIDERS:
            raise MissingMetadataError(
                f"Unknown LLM provider {self.provider!r}", allowed=sorted(PROVIDERS)
            )
        if not self.api_key.strip():
            raise MissingMetadataError(
                f"Empty API key for provider {self.provider!r} (fail closed)"
            )
        if not self.model.strip():
            raise MissingMetadataError("LLM model id must be set")

    @property
    def endpoint(self) -> str:
        return PROVIDERS[self.provider]

    def redacted(self) -> dict[str, Any]:
        key = self.api_key
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key": f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***",
        }

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> LLMConfig:
        e = env if env is not None else dict(os.environ)
        provider = e.get("EVOQUANT_LLM_PROVIDER", "").strip().lower()
        if not provider:
            raise MissingMetadataError(
                "EVOQUANT_LLM_PROVIDER not set ('openai' or 'openrouter')"
            )
        key_var = "OPENAI_API_KEY" if provider == "openai" else "OPENROUTER_API_KEY"
        api_key = e.get(key_var, "")
        if not api_key:
            raise MissingMetadataError(
                f"{key_var} not set for provider {provider!r} (fail closed)",
                required_env=key_var,
            )
        model = e.get("EVOQUANT_LLM_MODEL", "")
        if not model:
            raise MissingMetadataError("EVOQUANT_LLM_MODEL not set")
        return cls(provider=provider, api_key=api_key, model=model)


def _urllib_transport(
    url: str, headers: dict[str, str], body: bytes, timeout_s: float
) -> tuple[int, str]:  # pragma: no cover - exercised only against live APIs
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read().decode("utf-8", errors="replace")


class ChatClient:
    def __init__(self, config: LLMConfig, transport: Transport | None = None) -> None:
        self._config = config
        self._transport: Transport = transport or _urllib_transport

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }
        if self._config.provider == "openrouter":
            headers["HTTP-Referer"] = "https://github.com/evoquant"
            headers["X-Title"] = "evoquant research"
        return headers

    def complete(self, messages: list[dict[str, str]]) -> str:
        """One chat completion; returns the assistant message content."""
        body = json.dumps(
            {"model": self._config.model, "messages": messages, "temperature": 0.2}
        ).encode()
        status, text = self._transport(
            self._config.endpoint, self._headers(), body, self._config.timeout_s
        )
        if status != 200:
            raise LLMClientError(
                f"LLM request failed with HTTP {status}",
                provider=self._config.provider,
                status=status,
                body=text[:500],
            )
        try:
            doc = json.loads(text)
            content = doc["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMClientError(
                "LLM response body has unexpected shape",
                provider=self._config.provider,
                body=text[:500],
            ) from exc
        if not isinstance(content, str):
            raise LLMClientError("LLM message content is not a string")
        return content

    def test_connection(self) -> dict[str, Any]:
        """Cheap round-trip check. Returns a redacted, artifact-safe report."""
        started = time.monotonic()
        content = self.complete(
            [{"role": "user", "content": "Reply with exactly: ok"}]
        )
        return {
            "ok": True,
            "provider": self._config.provider,
            "model": self._config.model,
            "latency_s": round(time.monotonic() - started, 3),
            "reply_preview": content[:40],
            "config": self._config.redacted(),
        }
