"""Resolve the LLM endpoint base URL across explicit input and env vars.

A single place so every entry point (Web, CLI, direct `TradingAgentsGraph`
usage in main.py) shares the same precedence. Native dual-format gateways
expose Claude and GPT under different URLs, so each provider has its own
env var; a generic BACKEND_URL stays as a catch-all fallback.
"""

import os
from typing import Optional

# Provider-specific base URL env vars (checked before the generic fallback).
_PROVIDER_BASE_URL_ENV = {
    "openai": "OPENAI_BASE_URL",
    "anthropic": "ANTHROPIC_BASE_URL",
}


def resolve_base_url(provider: str, explicit: Optional[str] = None) -> Optional[str]:
    """Return the base URL to use, or None to fall back to the official endpoint.

    Precedence: explicit (UI/CLI input) > provider-specific env
    (OPENAI_BASE_URL / ANTHROPIC_BASE_URL) > generic BACKEND_URL > None.
    """
    if explicit and explicit.strip():
        return explicit.strip()

    env_name = _PROVIDER_BASE_URL_ENV.get((provider or "").lower())
    if env_name:
        val = os.getenv(env_name)
        if val and val.strip():
            return val.strip()

    backend = os.getenv("BACKEND_URL")
    if backend and backend.strip():
        return backend.strip()

    return None
