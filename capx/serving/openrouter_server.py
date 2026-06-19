"""Backwards-compatible entrypoint for the OpenRouter proxy.

This file used to implement a single-upstream OpenAI-compatible proxy for
OpenRouter. It has been replaced by the unified multi-route proxy in
``capx/serving/llm_proxy_server.py``; this module is now a thin shim that
preserves the historical CLI (``--key-file`` / ``--api-key`` /
``--base-url`` / ``--port``) so existing scripts and docs (e.g.
``scripts/start_servers_and_eval.sh``, ``README.md``) continue to work
without edits.

For new setups prefer:

    python -m capx.serving.llm_proxy_server --config capx/serving/llm_routes.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

import tyro
import uvicorn

from capx.serving.llm_proxy_server import (
    Route,
    _read_first_nonempty_line,
    create_app,
    load_routes,
)

logger = logging.getLogger(__name__)


def _openrouter_only_routes(
    key_file: str, api_key: str | None, base_url: str
) -> list[Route]:
    """Build a single-route list that replicates the legacy behaviour."""

    if api_key is None:
        path = Path(key_file)
        if not path.exists():
            raise FileNotFoundError(f"Key file not found: {key_file}")
        api_key = _read_first_nonempty_line(path)
        logger.info("Loaded API key from %s", key_file)

    return [
        Route(
            name="openrouter",
            base_url=base_url,
            api_key=api_key,
            prefix_match="openrouter/",
            strip_prefix="openrouter/",
            default_headers={
                "HTTP-Referer": "https://github.com/nvidia-gear/CaP-X",
                "X-Title": "CaP-X",
            },
        )
    ]


def main(
    key_file: str = ".openrouterkey",
    api_key: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8110,
    base_url: str = "https://openrouter.ai/api/v1/",
    config: str | None = None,
    # Accepted for backwards compatibility; the unified proxy always uses the
    # async OpenAI client.
    async_client: bool = True,
) -> None:
    """Start the LLM proxy.

    If ``--config`` is provided the unified multi-route proxy is launched
    from that YAML file. Otherwise the legacy single-route (OpenRouter only)
    mode is used, driven by ``--key-file`` / ``--api-key`` / ``--base-url``.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    del async_client  # legacy flag, retained to avoid breaking CLI callers

    if config is not None:
        routes = load_routes(config)
    else:
        routes = _openrouter_only_routes(
            key_file=key_file, api_key=api_key, base_url=base_url
        )

    app = create_app(routes)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    tyro.cli(main)
