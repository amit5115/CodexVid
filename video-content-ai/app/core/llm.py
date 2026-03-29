"""Pluggable LLM backend abstraction.

Supports Ollama (default, local), OpenAI, Anthropic, and Company GPT
(internal proxy) via a unified interface.  The active provider is
selected via the VCAI_LLM_PROVIDER env var (defaults to "ollama").
"""

# =============================================================================
# WHAT THIS FILE IS FOR (plain English)
# =============================================================================
# Imagine a universal remote that can control several TV brands. Each brand
# (Ollama, OpenAI, Anthropic, Company GPT) speaks a different "language," but
# you want one set of buttons: "answer this chat," "stream the answer,"
# "turn text into search numbers (embeddings)," "list channels (models)."
# This file defines those buttons and the adapters for each brand.
#
# WHO CALLS WHAT IN THE REAL APP
# - app/services/generation.py → get_provider(), then .chat(), .chat_stream(),
#   or .vision_chat() to generate content from transcripts and slides.
# - app/core/vectorstore.py → get_provider(), then .embed() for semantic search.
# - app/api/misc.py → get_provider(), then .list_models() for a models dropdown.
# - tests/test_core_modules.py → reset_provider(), register_provider(), and
#   get_provider() to test without hitting real AI services.
# =============================================================================

from __future__ import (
    annotations,  # Allows `str | None` style types everywhere in this file; must stay near the top (special import).
)

import logging  # Built-in "diary" for servers: log lines show what happened and where.
import os  # Read environment variables (keys, provider choice) and set proxy-related env defaults.
from abc import ABC, abstractmethod  # ABC = blueprint class; abstractmethod = "subclasses must implement this button."
from collections.abc import (
    Generator,  # Type hint: a stream that yields many strings one after another (like a ticker tape).
)

import requests  # Library to send HTTP (web) requests — used by the company internal AI gateway provider.

logger = logging.getLogger(__name__)  # Logger named after this module; who uses it: _get_or_create() below. Who configures logging: the app at startup.


class ContentFilterError(RuntimeError):
    """Raised when Azure's content management policy rejects a prompt."""


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def chat(self, model: str, messages: list[dict], **kwargs) -> str:
        """Synchronous chat completion.  Returns the assistant message content."""

    @abstractmethod
    def chat_stream(self, model: str, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        """Streaming chat completion.  Yields tokens."""

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support embeddings")

    def list_models(self) -> list[str]:
        return []

    def vision_chat(self, model: str, prompt: str, image_paths: list[str], **kwargs) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} does not support vision")


# ── Ollama ────────────────────────────────────────────────────────────
# Section divider: everything below is the "Ollama brand" adapter (local AI on your machine).

class OllamaProvider(LLMProvider):
    # Provider for Ollama: runs models on your computer — like a TV in the house, not a satellite feed.

    def __init__(self):
        import ollama  # Import here so the app can start without Ollama until you actually pick this provider.
        self._client = ollama  # Store the library module; we call functions on it like buttons on a remote.

        # Who constructs this: _get_or_create("ollama") when get_provider() picks Ollama.

    def chat(self, model: str, messages: list[dict], **kwargs) -> str:
        resp = self._client.chat(model=model, messages=messages)  # One blocking request; resp is a dict from the library.
        return resp["message"]["content"]  # Extract assistant text from Ollama's JSON-shaped result.

        # Who calls this: generation.py (via get_provider()) when using Ollama as the backend.

    def chat_stream(self, model: str, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        stream = self._client.chat(model=model, messages=messages, stream=True)  # Ask for incremental chunks, not one big blob.
        for chunk in stream:  # Walk each partial update as it arrives from Ollama.
            token = chunk["message"]["content"]  # This chunk's new text piece (may be empty between events).
            if token:  # Skip empty strings so the UI doesn't flicker with blank updates.
                yield token  # Hand this piece to whoever is iterating (generation.py streams to the client).

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        resp = self._client.embed(model=model, input=texts)  # Batch-encode texts into vectors for search.
        return resp["embeddings"]  # List of float lists — semantic "coordinates" for each string.

        # Who calls it: vectorstore.py after get_provider().

    def list_models(self) -> list[str]:
        try:  # If Ollama isn't running, don't crash the whole HTTP endpoint.
            models = self._client.list()  # Ask Ollama which models are installed locally.
            return [m["name"] for m in models.get("models", [])]  # Build simple names; .get avoids KeyError if shape differs.
        except Exception:  # Broad catch: any failure → behave as "no models" instead of error page.
            return []  # Empty list tells misc.py / UI "nothing to show."

        # Who calls it: app/api/misc.py through the shared LLMProvider interface.

    def vision_chat(self, model: str, prompt: str, image_paths: list[str], **kwargs) -> str:
        resp = self._client.chat(  # Same chat API, but the message carries image paths Ollama can load from disk.
            model=model,  # Which vision-capable model name is running in Ollama.
            messages=[{"role": "user", "content": prompt, "images": image_paths}],  # One user turn: words + pictures.
        )
        return resp["message"]["content"]  # Model's text answer about the images.

        # Who calls it: generation.py when analyzing slide screenshots.


# ── OpenAI ────────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    def __init__(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY env var is required for OpenAI provider")
        self._client = OpenAI(api_key=api_key)

    def chat(self, model: str, messages: list[dict], **kwargs) -> str:
        cleaned = [{"role": m["role"], "content": m["content"]} for m in messages]
        resp = self._client.chat.completions.create(model=model, messages=cleaned)
        return resp.choices[0].message.content or ""

    def chat_stream(self, model: str, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        cleaned = [{"role": m["role"], "content": m["content"]} for m in messages]
        stream = self._client.chat.completions.create(model=model, messages=cleaned, stream=True)
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    def list_models(self) -> list[str]:
        try:
            models = self._client.models.list()
            return [m.id for m in models.data]
        except Exception:
            return []


# ── Anthropic ─────────────────────────────────────────────────────────
# Section divider: Anthropic (Claude) cloud API — another brand, different wire protocol than OpenAI.

class AnthropicProvider(LLMProvider):
    def __init__(self):
        try:  # Same pattern as OpenAI: fail clearly if SDK missing.
            import anthropic  # Third-party `anthropic` package.
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")
        api_key = os.getenv("ANTHROPIC_API_KEY", "")  # Claude API key from the environment.
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY env var is required for Anthropic provider")
        self._client = anthropic.Anthropic(api_key=api_key)  # Official Anthropic client object.

        # Who constructs this: _get_or_create("anthropic") when env selects Anthropic.

    def chat(self, model: str, messages: list[dict], **kwargs) -> str:
        cleaned = [{"role": m["role"], "content": m["content"]} for m in messages]  # Normalize message list.
        resp = self._client.messages.create(  # Anthropic uses "messages" API, not OpenAI's "chat.completions".
            model=model, messages=cleaned, max_tokens=4096,  # Cap how long the reply may be (token ≈ word piece).
        )
        return resp.content[0].text  # First block of the response is the assistant's text.

        # Who calls it: generation.py through get_provider().

    def chat_stream(self, model: str, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        cleaned = [{"role": m["role"], "content": m["content"]} for m in messages]  # Same cleaning as chat().
        with self._client.messages.stream(  # Context manager ensures the HTTP stream closes cleanly.
            model=model, messages=cleaned, max_tokens=4096,
        ) as stream:  # `stream` is the live streaming session object from the SDK.
            for text in stream.text_stream:  # SDK helper that yields human-readable fragments.
                yield text  # Pass each fragment to generation.py's streaming loop.

        # Who calls it: generation.py for token-by-token UI updates when Anthropic is selected.


# ── Company GPT (internal proxy) ─────────────────────────────────────

class CompanyGPTProvider(LLMProvider):
    """LLM provider for Amdocs AI Framework proxy (ai-framework1)."""

    def __init__(self):
        from app.config import COMPANY_GPT_API_KEY, COMPANY_GPT_CALLER, COMPANY_GPT_ENDPOINT
        self._endpoint = COMPANY_GPT_ENDPOINT
        self._api_key = COMPANY_GPT_API_KEY
        self._caller = COMPANY_GPT_CALLER
        if not self._api_key:
            from app.config import BASE_DIR

            raise ValueError(
                "COMPANY_GPT_API_KEY is required for company_gpt provider. "
                f"Add COMPANY_GPT_API_KEY to the environment or to {BASE_DIR / '.env'} "
                "(copy from .env.example)."
            )
        if not self._caller:
            raise ValueError("COMPANY_GPT_CALLER is required for company_gpt provider")
        host = self._endpoint.split("//")[-1].split(":")[0]
        os.environ.setdefault("NO_PROXY", host)
        os.environ.setdefault("no_proxy", host)
        self._verify_ssl = os.getenv("COMPANY_GPT_VERIFY_SSL", "false").lower() in ("1", "true", "yes")
        self._session = requests.Session()
        self._session.verify = self._verify_ssl
        self._session.headers.update({
            "Content-Type": "application/json",
            "API-Key": self._api_key,
            "X-Effective-Caller": self._caller,
        })

    def _call(self, model: str, messages: list[dict]) -> str:
        _company_system = (
            "You are a precise, professional assistant for enterprise users. "
            "Follow instructions closely; stay factual and concise."
        )
        cleaned = [{"role": m["role"], "content": m["content"]} for m in messages]
        has_system = any(m["role"] == "system" for m in cleaned)
        if not has_system:
            cleaned.insert(0, {"role": "system", "content": _company_system})

        payload = {"llm_model": model, "messages": cleaned}
        resp = self._session.post(
            f"{self._endpoint}/api/v1/call_llm",
            json=payload,
            timeout=300,
        )
        if resp.status_code != 200:
            if resp.status_code == 400 and "content_filter" in resp.text:
                raise ContentFilterError(
                    "Azure content filter triggered — the video transcript "
                    "contains language flagged by the content policy. "
                    "Try a different video or switch to Ollama (local model)."
                )
            raise RuntimeError(f"Company GPT error {resp.status_code}: {resp.text[:500]}")
        return resp.json()["message"]

    def chat(self, model: str, messages: list[dict], **kwargs) -> str:
        return self._call(model, messages)

    def chat_stream(self, model: str, messages: list[dict], **kwargs) -> Generator[str, None, None]:
        yield self._call(model, messages)

    def list_models(self) -> list[str]:
        return ["gpt-4.1", "gpt-4o", "gpt-4o-mini"]


# ── Provider registry ─────────────────────────────────────────────────
# Section divider: lookup tables and small functions that pick which "remote" to use.

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,  # String name from env → class to instantiate (not an instance yet).
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "company_gpt": CompanyGPTProvider,
}

# Who reads _PROVIDERS: _get_or_create() and register_provider(). Tests may register extra names.

_COMPANY_GPT_MODELS = {"gpt-4.1", "gpt-4o", "gpt-4o-mini"}  # If user picks one of these model IDs, force company GPT path.

# UI / legacy labels that must map to the proxy's ``llm_model`` id (see CompanyGPTProvider._call).
_COMPANY_GPT_UI_ALIASES: dict[str, str] = {
    "company-gpt4o": "gpt-4o",
    "company-gpt-4o": "gpt-4o",
    "company_gpt4o": "gpt-4o",
}

_instances: dict[str, LLMProvider] = {}  # Cache of built providers: one shared object per provider name (singleton-style).


def normalize_llm_model_id(model: str) -> str:
    """Map dropdown / legacy names to the id sent to the backend (e.g. company-gpt4o → gpt-4o)."""
    if not model:
        return model
    return _COMPANY_GPT_UI_ALIASES.get(model.strip().lower(), model)


def _get_or_create(name: str) -> LLMProvider:
    if name not in _instances:  # First time we need this provider name — build it.
        cls = _PROVIDERS.get(name)  # Fetch the class object from the registry dict.
        if cls is None:  # Typo in env var or unregistered custom name.
            raise ValueError(
                f"Unknown LLM provider '{name}'. Available: {list(_PROVIDERS.keys())}"
            )
        logger.info("Initializing LLM provider: %s", name)  # Operator-visible log line (two-arg style avoids string concat).
        _instances[name] = cls()  # Call the class → runs __init__, store the instance in the cache dict.
    return _instances[name]  # On every later call, return the same instance (reuse connections and config).

    # Who calls _get_or_create: only get_provider() in this module (leading _ means "internal use" by convention).


def get_provider(model: str | None = None) -> LLMProvider:
    """Return the LLM provider for *model*.

    Auto-routes Company GPT models (gpt-4.1, gpt-4o, gpt-4o-mini) to the
    CompanyGPTProvider regardless of the default.  Everything else goes to
    the provider selected via VCAI_LLM_PROVIDER (defaults to "ollama").
    """
    if model and model.lower() in _COMPANY_GPT_MODELS:  # model can be None — then skip this branch.
        return _get_or_create("company_gpt")  # These model names always go through the internal gateway.
    default = os.getenv("VCAI_LLM_PROVIDER", "ollama").lower()  # Read which backend to use; default local Ollama.
    return _get_or_create(default)  # Build or return cached provider for that name.

    # Who calls get_provider: generation.py, vectorstore.py, misc.py, and tests — main public entry point.


def reset_provider() -> None:
    """Reset all cached provider instances."""
    _instances.clear()  # Empty the cache dict so the next get_provider() constructs fresh provider objects.

    # Who calls reset_provider: tests (test_core_modules.py) to isolate cases; occasionally useful if config changes live.


def register_provider(name: str, cls: type[LLMProvider]) -> None:
    """Register a custom LLM provider class."""
    _PROVIDERS[name] = cls  # Add or overwrite a mapping — lets tests inject a fake "dummy" provider.

    # Who calls register_provider: tests; a future plugin could call this at startup to add a new backend.
