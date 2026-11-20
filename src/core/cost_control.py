"""Small, explicit limits; no automatic retries or optional model calls."""
import os
import json

# Server-reported usage only, never an estimate. NeMo self-checks have separate SDK calls.
GEMINI_USAGE = []


def token_limit(provider: str) -> int:
    name = "BLUE_MAX_OUTPUT_TOKENS" if provider == "openrouter" else "RED_MAX_OUTPUT_TOKENS"
    value = int(os.environ.get(name, "768"))
    if not 64 <= value <= 4096:
        raise ValueError(f"{name} must be between 64 and 4096")
    return value


def safe_error(exc: Exception) -> str:
    """Never write a real credential from a provider exception into artifacts."""
    message = str(exc)
    body = getattr(exc, "body", None)
    if body:
        message += " | provider_details=" + json.dumps(body, ensure_ascii=False, default=str)
    for key in ("OPENROUTER_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(key)
        if value:
            message = message.replace(value, "[REDACTED]")
    return f"{type(exc).__name__}: {message[:1200]}"


def gemini_generation_config():
    from google.genai import types
    return types.GenerateContentConfig(max_output_tokens=token_limit("gemini"))


def gemini_model(model: str):
    import httpx
    from google.adk.models.google_llm import Gemini
    from google.genai import types
    return Gemini(model=model, retry_options=types.HttpRetryOptions(attempts=1),
                  # Explicit HTTPX transport: aiohttp connections reset on this network.
                  client_kwargs={"http_options": types.HttpOptions(
                      timeout=45000, httpx_async_client=httpx.AsyncClient(timeout=45.0))})


def blue_endpoint_health() -> dict:
    """Free advisory metadata; negative status does not prove inference is impossible."""
    from urllib.request import urlopen
    from core.config import get_blue_model
    url = f"https://openrouter.ai/api/v1/models/{get_blue_model()}/endpoints"
    try:
        with urlopen(url, timeout=10) as response:
            data = json.load(response).get("data", {})
        endpoints = [{key: ep.get(key) for key in ("provider_name", "status", "uptime_last_30m")}
                     for ep in data.get("endpoints", [])]
        unavailable = not endpoints or all((ep.get("status") or 0) < 0 for ep in endpoints)
        return {"checked": True, "unavailable": unavailable, "endpoints": endpoints}
    except Exception as exc:
        # An unavailable health API is not evidence that inference itself is down.
        return {"checked": False, "unavailable": False, "error": safe_error(exc)}
