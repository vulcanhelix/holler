"""Transcript -> {"url", "goals"} via Cerebras. The only prose step upstream of the agent."""

import json
import os
import re
import time
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
                "maxItems": 4,
            },
            "clarify": {"type": "string"},
        },
        "required": ["url", "goals", "clarify"],
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
- When the user asks to "check"/"tell me about" X, prefer a goal that READS
  what's already on the target page ("read the campaign reply counts") over
  navigating deeper — only open into things the user explicitly names.
- When the user names a specific thing inside a site ("the X campaign"), make
  the first goal search/filter the site's list for that name.
- If the request is ambiguous — e.g. several things could match — return
  {{"url": "", "goals": [], "clarify": "<one short spoken question>"}} instead
  of guessing. Keep "clarify" empty when the request is clear.
- The transcript may contain meta-instructions ("just google it", "actually").
  Follow the intent; never turn instruction words into a search query or URL.
- "context" may hold the browser's current page, the previous request, and a
  clarifying question you asked. When "pending" is set, the new transcript is
  the user's answer — merge it into the previous request. Follow-ups ("now
  check inbox two", "go back") refer to the current page.
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
    model = os.environ.get("CEREBRAS_MODEL", "gpt-oss-120b")
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    # qwen models spend reasoning tokens before content; "none" is unsupported on gpt-oss.
    effort = os.environ.get("CEREBRAS_REASONING_EFFORT") or ("none" if "qwen" in model else None)
    if effort:
        body["reasoning_effort"] = effort
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": schema}
    resp = None
    for attempt in range(2):
        try:
            resp = httpx.post(
                f"{base}/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['CEREBRAS_API_KEY']}"},
                json=body,
                timeout=15,
            )
        except httpx.TransportError:
            if attempt:
                raise
            time.sleep(0.6)
            continue
        if (resp.status_code == 429 or resp.status_code >= 500) and not attempt:
            time.sleep(0.6)
            continue
        break
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


DECIDE_SCHEMA = {
    "name": "decide",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string"},
            "answer": {"type": "string"},
            "url": {"type": "string"},
            "goals": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
            "clarify": {"type": "string"},
        },
        "required": ["status", "answer", "url", "goals", "clarify"],
        "additionalProperties": False,
    },
}

DECIDE_SYSTEM = """You drive a browser agent one narrow step at a time and answer the user by voice.
Given the user's spoken request, the CURRENT page text, the agent status, and
actions already taken, return JSON
{"status": "done"|"continue"|"clarify", "answer": "...", "url": "...", "goals": [...], "clarify": "..."}:
- "done": the request is answered by what is on the page or already done. Put
  the spoken reply in "answer": ONE plain sentence under 25 words, no markdown,
  no URLs, using only the page text. If the page does not contain the answer,
  say what you found and that the rest is not visible.
- "continue": give 1-3 narrow next goals verifiable on this page. If the next
  step needs a different site entirely, put its URL in "url". Leave "answer" empty.
- "clarify": the user must choose first (e.g. several matching items) — put one
  short spoken question in "clarify". Leave "answer" empty.
Never repeat an action already taken. Never guess past what the page text shows.
If the agent blocked or made no progress, either propose a materially different
approach or return done (with what was found) / clarify — do not repeat the plan.
"""


def _route_llm(system, user):
    return json.loads(chat(system, user, schema=ROUTE_SCHEMA))


def _decide_llm(system, user):
    return json.loads(chat(system, user, schema=DECIDE_SCHEMA))


def route(transcript, *, context=None, call_llm=None, aliases_path=None):
    """Return {"url": str, "goals": [str, ...]} or None if routing fails."""
    sites, mishearings = load_aliases(aliases_path)
    heard = normalize(transcript, mishearings)
    call_llm = call_llm or _route_llm
    site_lines = "\n".join(f"{name} = {url}" for name, url in sites.items())
    user = {"heard": heard, "raw_transcript": transcript}
    if context:
        user["context"] = context
    try:
        out = call_llm(SYSTEM.format(sites=site_lines), json.dumps(user))
        clarify = out.get("clarify") or ""
        named = [n for n in sites if re.search(rf"\b{re.escape(n)}\b", heard)]
        if clarify.strip():
            # keep the named site's URL so the clarified follow-up keeps its destination
            return {"clarify": clarify.strip(), "url": sites.get(named[0], "") if named else ""}
        url = out["url"]
        goals = [g.strip() for g in out["goals"]]
    except (AttributeError, KeyError, TypeError, ValueError, httpx.HTTPError):
        return None
    # If a named site's transcript mention produced a non-alias URL, force the alias —
    # unless the model deliberately chose another known site (e.g. "search X on google").
    if named and url not in sites.values():
        url = sites[named[0]]
    parsed = urlparse(url) if isinstance(url, str) else None
    if not parsed or parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    if not goals or len(goals) > 4 or any(not g or len(g) >= 120 for g in goals):
        return None
    return {"url": url, "goals": goals}


def decide(request, page, history, status="", *, call_llm=None):
    """Given the request and the real current page, pick the next step.

    Returns {"status": "done", "answer": str} | {"status": "continue", "goals": [...], "url": str}
    | {"clarify": str} | None (on failure).
    """
    call_llm = call_llm or _decide_llm
    user = json.dumps({
        "request": request,
        "page": {
            "url": page.get("url", ""),
            "title": page.get("title", ""),
            "text": (page.get("text") or "")[:4000],
        },
        "actions_taken": [h.get("action") for h in (history or [])][-10:],
        "agent_status": status,
    })
    try:
        out = call_llm(DECIDE_SYSTEM, user)
        clarify = (out.get("clarify") or "").strip()
        if clarify or out.get("status") == "clarify":
            return {"clarify": clarify or "can you say that another way?"}
        answer = (out.get("answer") or "").strip()
        if out.get("status") == "done":
            return {"status": "done", "answer": answer}
        goals = [g.strip() for g in (out.get("goals") or []) if isinstance(g, str)]
        goals = [g for g in goals if g and len(g) < 120][:3]
        if not goals:
            return {"status": "done", "answer": answer}
        return {"status": "continue", "goals": goals, "url": out.get("url") or ""}
    except (AttributeError, KeyError, TypeError, ValueError, httpx.HTTPError):
        return None
