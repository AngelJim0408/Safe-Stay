"""Provider client with fallback. Only sends a prompt and returns text.

Groq and OpenRouter are OpenAI-compatible; Gemini uses its REST endpoint.
Model names are placeholders - set them to models your keys can access.
"""
import requests

import config

MODELS = config.LLM_MODELS
_OPENAI_STYLE = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


def _check(r) -> None:
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")


def _call(provider: str, system: str, user: str) -> str:
    key = config.LLM_KEYS.get(provider, "")
    if not key:
        raise RuntimeError(f"no key for {provider}")
    if provider in _OPENAI_STYLE:
        r = requests.post(
            _OPENAI_STYLE[provider],
            headers={"Authorization": f"Bearer {key}"},
            json={"model": MODELS[provider], "temperature": 0,
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}]},
            timeout=180)
        _check(r)
        return r.json()["choices"][0]["message"]["content"]
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{MODELS['gemini']}:generateContent")
    r = requests.post(url, headers={"x-goog-api-key": key}, timeout=180, json={
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0}})
    _check(r)
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def generate(system: str, user: str) -> str:
    """Try the main provider, then fallbacks. Raises if all fail."""
    errors = []
    for provider in [config.MAIN_LLM, *config.LLM_FALLBACK_ORDER]:
        try:
            return _call(provider, system, user)
        except Exception as exc:  # noqa: BLE001 - report which provider failed, not the key
            errors.append(f"{provider}: {exc}")
    raise RuntimeError("all LLM providers failed: " + "; ".join(errors))
