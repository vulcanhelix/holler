"""One spoken line back: the blocked reason, or a Cerebras summary of the final page."""

import json
import os
import subprocess
import tempfile

import httpx

from .route import chat

SYSTEM = (
    "You answer the user's spoken request in ONE short spoken sentence, using "
    "only the page text provided. Plain words, no markdown, no URLs, under 25 "
    "words. If the answer is not on the page, say so in one sentence. The agent "
    "status and last actions are provided — if it got blocked, say what it "
    "found before failing, not just that it failed."
)


def say(text):
    print(f"say: {text}", flush=True)
    if os.environ.get("DEEPGRAM_API_KEY"):
        try:
            resp = httpx.post(
                "https://api.deepgram.com/v1/speak",
                params={"model": os.environ.get("DEEPGRAM_MODEL", "aura-2-thalia-en")},
                headers={
                    "Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}",
                    "Content-Type": "application/json",
                },
                json={"text": text},
                timeout=15,
            )
            resp.raise_for_status()
            fd, path = tempfile.mkstemp(suffix=".mp3")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(resp.content)
                subprocess.run(["afplay", path], check=False)
            finally:
                os.unlink(path)
            return
        except Exception as e:
            print(f"deepgram tts failed ({e}); falling back to say", flush=True)
    voice = os.environ.get("HOLLER_VOICE")
    subprocess.run(["say", *(["-v", voice] if voice else []), text], check=False)


def speak_result(state, transcript, *, call_llm=None):
    """Speak one line about the final agent state. Returns the spoken text."""
    call_llm = call_llm or chat
    page = state.get("page") or {}
    actions = [h.get("action", "") for h in (state.get("history") or [])][-5:]
    user = json.dumps({
        "request": transcript,
        "status": state.get("status", ""),
        "url": page.get("url", ""),
        "last_actions": actions,
        "page_text": page.get("text", "")[:4000],
    })
    try:
        line = call_llm(SYSTEM, user).strip()
    except Exception:
        line = ""
    if not line:
        line = f"blocked, {actions[-1]}" if state.get("status") == "blocked" and actions else "done"
    say(line)
    return line
