"""One spoken line back: the blocked reason, or a Cerebras summary of the final page."""

import json
import subprocess

from .route import chat

SYSTEM = (
    "You answer the user's spoken request in ONE short spoken sentence, using "
    "only the page text provided. Plain words, no markdown, no URLs, under 25 "
    "words. If the answer is not on the page, say so in one sentence."
)


def say(text):
    print(f"say: {text}", flush=True)
    subprocess.run(["say", text], check=False)


def speak_result(state, transcript, *, call_llm=None):
    """Speak one line about the final agent state. Returns the spoken text."""
    call_llm = call_llm or chat
    if state.get("status") == "blocked":
        history = state.get("history") or []
        last = history[-1]["action"] if history else "no progress"
        line = f"blocked, {last}"
    else:
        page = state.get("page") or {}
        user = json.dumps({"request": transcript, "url": page.get("url", ""), "page_text": page.get("text", "")[:4000]})
        try:
            line = call_llm(SYSTEM, user).strip() or "done"
        except Exception:
            line = "done"
    say(line)
    return line
