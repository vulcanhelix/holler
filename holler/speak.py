"""One spoken line back: streaming Deepgram TTS, a canned-phrase PCM cache, or macOS say."""

import hashlib
import json
import os
import subprocess
import threading
from pathlib import Path

import httpx
import sounddevice as sd

from .route import chat

CANNED = {"stopped", "failed", "didn't get that", "the page didn't load"}
CACHE_DIR = Path.home() / ".cache" / "holler" / "tts"

SYSTEM = (
    "You answer the user's spoken request in ONE short spoken sentence, using "
    "only the page text provided. Plain words, no markdown, no URLs, under 25 "
    "words. If the answer is not on the page, say so in one sentence. The agent "
    "status and last actions are provided — if it got blocked, say what it "
    "found before failing, not just that it failed."
)


def _model():
    return os.environ.get("DEEPGRAM_MODEL", "aura-2-thalia-en")


def _canned_path(text):
    digest = hashlib.sha1((_model() + text).encode()).hexdigest()
    return CACHE_DIR / f"{digest}.pcm"


def _play(pcm):
    with sd.RawOutputStream(samplerate=24000, channels=1, dtype="int16") as out:
        out.write(pcm)


def _deepgram_pcm(text):
    """Stream linear16 PCM from Deepgram, playing chunks as they land; returns the bytes."""
    buf = bytearray()
    tail = b""
    with httpx.stream(
        "POST",
        "https://api.deepgram.com/v1/speak",
        params={
            "model": _model(),
            "encoding": "linear16",
            "sample_rate": "24000",
            "container": "none",
        },
        headers={
            "Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={"text": text},
        timeout=15,
    ) as resp:
        resp.raise_for_status()
        with sd.RawOutputStream(samplerate=24000, channels=1, dtype="int16") as out:
            for chunk in resp.iter_bytes(chunk_size=4800):
                data = tail + chunk
                if len(data) % 2:
                    data, tail = data[:-1], data[-1:]
                else:
                    tail = b""
                if data:
                    out.write(data)
                    buf += data
    return bytes(buf) + tail


def say(text):
    print(f"say: {text}", flush=True)
    if os.environ.get("DEEPGRAM_API_KEY"):
        try:
            path = _canned_path(text) if text in CANNED else None
            if path is not None and path.exists():
                _play(path.read_bytes())
                return
            pcm = _deepgram_pcm(text)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pcm)
            return
        except Exception as e:
            print(f"deepgram tts failed ({e}); falling back to say", flush=True)
    voice = os.environ.get("HOLLER_VOICE")
    subprocess.run(["say", *(["-v", voice] if voice else []), text], check=False)


def say_async(text):
    """Speak on a daemon thread; returns the thread."""
    t = threading.Thread(target=say, args=(text,), daemon=True)
    t.start()
    return t


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
