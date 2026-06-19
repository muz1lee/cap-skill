from capx.llm.client import ModelQueryArgs, query_model


class _FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "ok"}}]}


def test_query_model_uses_openai_api_key_env_for_openai_compatible_models(monkeypatch):
    captured = {}

    def fake_post(url, headers, data, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["data"] = data
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setattr("capx.llm.client.requests.post", fake_post)

    args = ModelQueryArgs(
        model="gemini-3.5-flash",
        server_url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    )

    result = query_model(args, [{"role": "user", "content": "hello"}])

    assert result["content"] == "ok"
    assert captured["headers"]["Authorization"] == "Bearer env-key"
