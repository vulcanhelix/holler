"""Transcript -> {"url", "goals"} via Cerebras. The only prose step upstream of the agent."""

import json
import os
import re
import tomllib
from pathlib import Path
from urllib.parse import urlparse

import httpx

ALIASES_PATH = Path(__file__).resolve().parent.parent / "aliases.toml"

ROUTE_SCHEMA = {
    "name": "route",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "goals": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 4,
            },
        },
        "required": ["url", "goals"],
        "additionalProperties": False,
    },
}

SYSTEM = """You convert a spoken request into a browser task for an agent that only
understands narrow, page-local goals.

Sites the user may name (spoken name -> URL):
{sites}

Return JSON {{"url": "...", "goals": ["...", ...]}}:
- url: the matching site URL above, or a well-known site's root URL if none matches.
- goals: ordered, 1-4 items. Each is ONE thing verifiable on a single page
  ("open the inboxes page", "read the warmup status"), under 12 words.
- The transcript may contain meta-instructions ("just google it", "actually").
  Follow the intent; never turn instruction words into a search query or URL.
"""


def load_aliases(path=None):
    path = Path(path or os.environ.get("HOLLER_ALIASES") or ALIASES_PATH)
    with path.open("rb") as f:
        data = tomllib.load(f)
    return data.get("sites", {}), data.get("mishearings", {})


def normalize(transcript, mishearings):
    text = transcript.lower()
    for heard, name in mishearings.items():
        text = re.sub(rf"\b{re.escape(heard.lower())}\b", name, text)
    return text


def chat(system, user, *, schema=None):
    """One Cerebras chat completion; returns message content."""
    base = os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1").rstrip("/")
    body = {
        "model": os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b"),
        "temperature": 0,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": schema}
    resp = httpx.post(
        f"{base}/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['CEREBRAS_API_KEY']}"},
        json=body,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _route_llm(system, user):
    return json.loads(chat(system, user, schema=ROUTE_SCHEMA))


def route(transcript, *, call_llm=None, aliases_path=None):
    """Return {"url": str, "goals": [str, ...]} or None if routing fails."""
    sites, mishearings = load_aliases(aliases_path)
    heard = normalize(transcript, mishearings)
    call_llm = call_llm or _route_llm
    site_lines = "\n".join(f"{name} = {url}" for name, url in sites.items())
    user = json.dumps({"heard": heard, "raw_transcript": transcript})
    try:
        out = call_llm(SYSTEM.format(sites=site_lines), user)
        url = out["url"]
        goals = [g.strip() for g in out["goals"]]
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    # If a named site's transcript mention produced a non-alias URL, force the alias —
    # unless the model deliberately chose another known site (e.g. "search X on google").
    named = [n for n in sites if re.search(rf"\b{re.escape(n)}\b", heard)]
    if named and url not in sites.values():
        url = sites[named[0]]
    parsed = urlparse(url) if isinstance(url, str) else None
    if not parsed or parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    if not goals or len(goals) > 4 or any(not g or len(g) >= 120 for g in goals):
        return None
    return {"url": url, "goals": goals}
