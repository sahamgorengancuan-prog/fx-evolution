from __future__ import annotations

import json

import pytest

from evoquant.errors import MissingMetadataError
from evoquant.llm.client import ChatClient, LLMClientError, LLMConfig


def _ok_transport(reply="ok"):
    calls: list[dict] = []

    def transport(url: str, headers: dict, body: bytes, timeout: float):
        calls.append({"url": url, "headers": headers, "body": json.loads(body)})
        return 200, json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": reply}}]}
        )

    return transport, calls


class TestConfig:
    def test_from_env_openai(self):
        cfg = LLMConfig.from_env(
            {
                "EVOQUANT_LLM_PROVIDER": "openai",
                "OPENAI_API_KEY": "sk-test-1234567890",
                "EVOQUANT_LLM_MODEL": "gpt-4o-mini",
            }
        )
        assert cfg.provider == "openai"
        assert cfg.endpoint.startswith("https://api.openai.com")

    def test_from_env_openrouter(self):
        cfg = LLMConfig.from_env(
            {
                "EVOQUANT_LLM_PROVIDER": "openrouter",
                "OPENROUTER_API_KEY": "or-test-1234567890",
                "EVOQUANT_LLM_MODEL": "meta-llama/llama-3.3-70b-instruct:free",
            }
        )
        assert cfg.endpoint.startswith("https://openrouter.ai")

    @pytest.mark.parametrize(
        "env",
        [
            {},
            {"EVOQUANT_LLM_PROVIDER": "openai"},  # no key
            {"EVOQUANT_LLM_PROVIDER": "openai", "OPENAI_API_KEY": "k"},  # no model
            {
                "EVOQUANT_LLM_PROVIDER": "skynet",
                "OPENAI_API_KEY": "k",
                "EVOQUANT_LLM_MODEL": "m",
            },
        ],
    )
    def test_incomplete_env_fails_closed(self, env):
        with pytest.raises(MissingMetadataError):
            LLMConfig.from_env(env)

    def test_redacted_never_contains_full_key(self):
        cfg = LLMConfig(provider="openai", api_key="sk-verysecretkey123", model="m")
        red = json.dumps(cfg.redacted())
        assert "sk-verysecretkey123" not in red
        assert red.count("...") == 1


class TestClient:
    def _cfg(self, provider="openai"):
        return LLMConfig(provider=provider, api_key="sk-test-1234567890", model="test-model")

    def test_complete_sends_auth_and_model(self):
        transport, calls = _ok_transport("hello")
        client = ChatClient(self._cfg(), transport=transport)
        out = client.complete([{"role": "user", "content": "hi"}])
        assert out == "hello"
        call = calls[0]
        assert call["headers"]["Authorization"] == "Bearer sk-test-1234567890"
        assert call["body"]["model"] == "test-model"

    def test_openrouter_headers_added(self):
        transport, calls = _ok_transport()
        ChatClient(self._cfg("openrouter"), transport=transport).complete(
            [{"role": "user", "content": "hi"}]
        )
        assert "HTTP-Referer" in calls[0]["headers"]

    def test_http_error_raises_structured(self):
        def transport(url, headers, body, timeout):
            return 401, json.dumps({"error": {"message": "bad key"}})

        client = ChatClient(self._cfg(), transport=transport)
        with pytest.raises(LLMClientError) as exc:
            client.complete([{"role": "user", "content": "hi"}])
        assert exc.value.details["status"] == 401

    def test_malformed_body_raises(self):
        def transport(url, headers, body, timeout):
            return 200, json.dumps({"unexpected": True})

        client = ChatClient(self._cfg(), transport=transport)
        with pytest.raises(LLMClientError):
            client.complete([{"role": "user", "content": "hi"}])

    def test_connection_check_reports_redacted_config(self):
        transport, _ = _ok_transport("ok")
        report = ChatClient(self._cfg(), transport=transport).test_connection()
        assert report["ok"] is True
        assert report["provider"] == "openai"
        assert "sk-test-1234567890" not in json.dumps(report)
