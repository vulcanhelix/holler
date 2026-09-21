"""Push-to-talk: hold a key, speak, release. Returns the transcript. No wake word."""

import asyncio
import io
import json
import os
import queue
import threading
import wave
from urllib.parse import quote

import httpx
import numpy as np
import sounddevice as sd
from pynput import keyboard

SAMPLE_RATE = 16000
_fw_model = None


def ptt_key():
    name = os.environ.get("HOLLER_PTT_KEY", "alt_r").strip()
    try:
        return keyboard.Key[name]
    except KeyError:
        return keyboard.KeyCode.from_char(name)


def abort_on_ptt_release():
    """Start a listener that sets the event when the PTT key is released (a tap)."""
    key = ptt_key()
    abort = threading.Event()

    def on_release(k):
        if k == key:
            abort.set()

    listener = keyboard.Listener(on_release=on_release)
    listener.start()
    return listener, abort


def _wav_bytes(audio):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((audio * 32767).clip(-32768, 32767).astype("<i2").tobytes())
    return buf.getvalue()


def _keyterms():
    """Site alias names + mishearing targets — Deepgram Nova-3 keyterm hints."""
    try:
        from .route import load_aliases

        sites, mishearings = load_aliases()
        return sorted(set(sites) | set(mishearings.values()))
    except Exception:
        return []


class _DGStream:
    """Deepgram live transcription on a background thread with its own asyncio loop."""

    def __init__(self, keyterms):
        self._q = queue.Queue()
        self._keyterms = keyterms
        self.finals = []
        self.error = None
        self._ws = None
        self._loop = None
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def feed(self, chunk):
        """Push one int16 PCM chunk (or None to end the stream)."""
        self._q.put_nowait(chunk)

    def finish(self, timeout=3.0):
        """Send CloseStream and wait for the socket to drain; returns the transcript."""
        self._q.put_nowait(None)
        if not self._closed.wait(timeout):
            if self._loop and self._ws:
                asyncio.run_coroutine_threadsafe(self._ws.close(), self._loop)
            if not self._closed.wait(1.0):
                raise TimeoutError("deepgram stream did not close")
        if self.error:
            raise self.error
        return " ".join(self.finals).strip()

    def _run(self):
        try:
            asyncio.run(self._main())
        except Exception as e:
            self.error = e
        finally:
            self._closed.set()

    async def _main(self):
        import websockets  # lazy: only needed on the Deepgram path

        url = (
            "wss://api.deepgram.com/v1/listen?model=nova-3&encoding=linear16"
            "&sample_rate=16000&channels=1&smart_format=true"
            "&interim_results=true&endpointing=300"
        )
        url += "".join("&keyterm=" + quote(k, safe="") for k in self._keyterms)
        self._loop = asyncio.get_running_loop()
        async with websockets.connect(
            url,
            additional_headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"},
        ) as ws:
            self._ws = ws

            async def sender():
                while True:
                    try:
                        chunk = self._q.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.02)
                        continue
                    if chunk is None:
                        await ws.send('{"type": "CloseStream"}')
                        return
                    await ws.send(chunk)

            send_task = asyncio.ensure_future(sender())
            try:
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except ValueError:
                        continue
                    if msg.get("is_final"):
                        alts = (msg.get("channel") or {}).get("alternatives") or [{}]
                        t = (alts[0].get("transcript") or "").strip()
                        if t:
                            self.finals.append(t)
            finally:
                send_task.cancel()


def _stream_transcribe(chunk_iter, keyterms):
    """Drive the Deepgram live socket with an iterable of int16 PCM chunks."""
    stream = _DGStream(keyterms)
    stream.start()
    for chunk in chunk_iter:
        stream.feed(chunk)
    return stream.finish()


def transcribe(audio):
    if key := os.environ.get("DEEPGRAM_API_KEY"):
        resp = httpx.post(
            "https://api.deepgram.com/v1/listen",
            params={
                "model": os.environ.get("DEEPGRAM_STT_MODEL", "nova-3"),
                "smart_format": "true",
            },
            headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"},
            content=_wav_bytes(audio),
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
    model = os.environ.get("HOLLER_WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
    try:
        import mlx_whisper
    except ImportError:
        # mlx is Apple Silicon only; Intel/CPU fallback keeps the same env knob.
        global _fw_model
        from faster_whisper import WhisperModel

        fw_name = model.rsplit("/", 1)[-1].removeprefix("whisper-") or "large-v3-turbo"
        if _fw_model is None:
            _fw_model = WhisperModel(fw_name, device="cpu", compute_type="int8")
        segments, _ = _fw_model.transcribe(audio)
        return "".join(s.text for s in segments).strip()
    return mlx_whisper.transcribe(audio, path_or_hf_repo=model)["text"].strip()


def listen():
    """Block until the push-to-talk key is held and released; return the transcript."""
    key = ptt_key()
    pressed = threading.Event()
    released = threading.Event()

    def on_press(k):
        if k == key:
            pressed.set()

    def on_release(k):
        if k == key and pressed.is_set():
            released.set()

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    pcm = []
    try:
        print(f"hold {os.environ.get('HOLLER_PTT_KEY', 'alt_r')} to talk", flush=True)
        pressed.wait()
        print("listening...", flush=True)
        stream = _DGStream(_keyterms()) if os.environ.get("DEEPGRAM_API_KEY") else None
        if stream:
            stream.start()
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16") as mic:
            while not released.is_set():
                data, _ = mic.read(SAMPLE_RATE // 10)
                pcm.append(data.copy())
                if stream:
                    stream.feed(data.tobytes())
    finally:
        listener.stop()
    if not pcm:
        if stream:
            stream.feed(None)
        return ""
    if stream:
        try:
            return stream.finish()
        except Exception:
            pass
    print("transcribing...", flush=True)
    audio = np.concatenate(pcm).reshape(-1).astype(np.float32) / 32768
    return transcribe(audio)
