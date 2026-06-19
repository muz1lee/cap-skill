"""Unified multi-route LLM proxy server.

A single FastAPI process that exposes an OpenAI-compatible
``/chat/completions`` endpoint and dispatches each request to one of several
upstream providers based on the ``model`` field of the request.

Each upstream is described by a ``Route`` entry loaded from a YAML config
(see ``capx/serving/llm_routes.yaml``). A route specifies:

- ``base_url`` / ``key_file``: the upstream OpenAI-compatible endpoint.
- ``prefix_match`` / ``model_match``: how incoming ``model`` strings are
  matched to this route.
- ``strip_prefix``: optional prefix removed from the model name before it is
  forwarded upstream (used for ``openrouter/`` style namespaces).
- ``default_headers``: extra HTTP headers attached to all upstream requests.
- ``default_extra_body``: default non-OpenAI fields merged into every
  upstream call (e.g. ``backend=dashscope`` for the Qwen proxy). The client's
  own extra fields win on conflict.

This replaces the single-upstream ``openrouter_server.py`` but remains
wire-compatible with it (same port, same ``/chat/completions`` URL shape),
so existing callers do not need to change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tyro
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)


# Known OpenAI chat-completions fields. Anything else in the request body is
# forwarded through the OpenAI SDK's ``extra_body`` channel so upstreams that
# accept custom fields (``backend``, ``enable_thinking``,
# ``chat_template_kwargs`` ...) see them verbatim.
_KNOWN_OPENAI_FIELDS = {
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "top_p",
    "stream",
    "reasoning_effort",
    "max_completion_tokens",
    "n",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "seed",
    "response_format",
    "tools",
    "tool_choice",
}


class ChatRequest(BaseModel):
    """Permissive chat-completions request schema.

    ``extra='allow'`` is critical: unknown fields (``backend``,
    ``enable_thinking``, ...) are preserved on the model and forwarded
    upstream as ``extra_body``, instead of being silently dropped.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[dict[str, Any]]
    stream: bool = False


@dataclass
class Route:
    name: str
    base_url: str
    api_key: str
    prefix_match: str | None = None
    model_match: list[str] = field(default_factory=list)
    strip_prefix: str | None = None
    default_headers: dict[str, str] = field(default_factory=dict)
    default_extra_body: dict[str, Any] = field(default_factory=dict)
    default: bool = False
    models: list[dict[str, str]] = field(default_factory=list)
    client: AsyncOpenAI | None = None

    def matches(self, model: str) -> bool:
        if self.prefix_match and model.startswith(self.prefix_match):
            return True
        return model in self.model_match


def _read_first_nonempty_line(path: Path) -> str:
    """Return the first non-empty, non-comment line of ``path``.

    Matches the original ``openrouter_server._load_api_keys`` behaviour
    (which effectively used only the first key) but raises clearly if the
    file is empty.
    """

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    raise ValueError(f"No API key found in {path}")


def load_routes(config_path: str | Path) -> list[Route]:
    """Load route definitions from a YAML file.

    A route whose ``key_file`` is missing is skipped with a visible warning
    (so a machine without every provider key can still run the proxy for
    the providers it does have). The process fails loudly only when *zero*
    routes end up enabled.
    """

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Route config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text()) or {}
    raw_routes = raw.get("routes")
    if not raw_routes:
        raise ValueError(f"No 'routes' section in {config_path}")

    routes: list[Route] = []
    for entry in raw_routes:
        name = entry.get("name") or entry.get("base_url")
        key_file = entry.get("key_file")
        api_key = entry.get("api_key")

        if api_key is None:
            if not key_file:
                logger.warning(
                    "Route '%s' has neither api_key nor key_file; skipping.",
                    name,
                )
                continue
            key_path = Path(key_file)
            if not key_path.exists():
                logger.warning(
                    "Route '%s' key file %s not found; skipping this route.",
                    name,
                    key_path,
                )
                continue
            try:
                api_key = _read_first_nonempty_line(key_path)
            except ValueError as exc:
                logger.warning(
                    "Route '%s' key file %s unusable (%s); skipping.",
                    name,
                    key_path,
                    exc,
                )
                continue

        routes.append(
            Route(
                name=name,
                base_url=entry["base_url"],
                api_key=api_key,
                prefix_match=entry.get("prefix_match"),
                model_match=list(entry.get("model_match") or []),
                strip_prefix=entry.get("strip_prefix"),
                default_headers=dict(entry.get("default_headers") or {}),
                default_extra_body=dict(entry.get("default_extra_body") or {}),
                default=bool(entry.get("default", False)),
                models=list(entry.get("models") or []),
            )
        )

    if not routes:
        raise RuntimeError(
            f"No usable routes in {config_path} (all skipped due to missing keys)."
        )

    logger.info("Loaded %d route(s): %s", len(routes), [r.name for r in routes])
    return routes


def _pick_route(routes: list[Route], model: str) -> Route:
    default_route: Route | None = None
    for route in routes:
        if route.matches(model):
            return route
        if route.default:
            default_route = route
    if default_route is not None:
        return default_route
    raise HTTPException(
        status_code=404,
        detail=(
            f"No upstream route matches model '{model}'. "
            f"Known routes: {[r.name for r in routes]}."
        ),
    )


def _split_payload(
    payload: dict[str, Any], route: Route
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a request payload into (known OpenAI kwargs, extra_body)."""

    known: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _KNOWN_OPENAI_FIELDS:
            known[key] = value
        else:
            extras[key] = value

    # Merge route-level defaults under caller-provided extras (caller wins).
    merged_extras = {**route.default_extra_body, **extras}
    return known, merged_extras


def create_app(routes: list[Route]) -> FastAPI:
    for route in routes:
        route.client = AsyncOpenAI(
            api_key=route.api_key,
            base_url=route.base_url,
            default_headers=route.default_headers or None,
        )

    app = FastAPI(title="CaP-X LLM Proxy", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/chat/completions")
    async def chat_completions(request: ChatRequest):
        payload = request.model_dump(exclude_none=True)
        route = _pick_route(routes, payload["model"])

        # Rewrite model name for upstream (e.g. drop "openrouter/" prefix).
        if route.strip_prefix and payload["model"].startswith(route.strip_prefix):
            payload["model"] = payload["model"][len(route.strip_prefix):]

        known, extra_body = _split_payload(payload, route)

        assert route.client is not None
        stream = bool(known.pop("stream", False))
        try:
            if stream:
                upstream = await route.client.chat.completions.create(
                    stream=True, extra_body=extra_body, **known
                )

                async def event_stream():
                    async for chunk in upstream:
                        yield f"data: {chunk.model_dump_json()}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    event_stream(), media_type="text/event-stream"
                )

            response = await route.client.chat.completions.create(
                extra_body=extra_body, **known
            )
            return response.model_dump()
        except Exception as exc:
            # Surface the upstream error with the route name so the caller can
            # tell which upstream failed. Re-raise as HTTP 500 to match the
            # previous openrouter_server.py behaviour.
            logger.exception("Upstream error on route '%s'", route.name)
            raise HTTPException(
                status_code=500, detail=f"[{route.name}] {exc}"
            ) from exc

    @app.get("/models")
    async def list_models():
        return {
            "routes": [
                {
                    "route": r.name,
                    "default": r.default,
                    "models": r.models,
                }
                for r in routes
            ]
        }

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "routes": [r.name for r in routes],
        }

    return app


def main(
    config: str = "capx/serving/llm_routes.yaml",
    host: str = "0.0.0.0",
    port: int = 8110,
) -> None:
    """Start the unified LLM proxy server.

    Args:
        config: Path to a YAML file describing the upstream routes.
        host: Bind address.
        port: Bind port. Default 8110 matches
            ``capx/llm/client.py::OPENROUTER_SERVER_URL`` so callers that go
            through the proxy do not need to change.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    routes = load_routes(config)
    app = create_app(routes)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
