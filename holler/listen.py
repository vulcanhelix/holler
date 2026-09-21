"""Push-to-talk: hold a key, speak, release. Returns the transcript. No wake word."""

import os
import threading

import numpy as np
import sounddevice as sd
from pynput import keyboard

SAMPLE_RATE = 16000
_fw_model = None


def _ptt_key():
    name = os.environ.get("HOLLER_PTT_KEY", "alt_r").strip()
    try:
        return keyboard.Key[name]
    except KeyError:
        return keyboard.KeyCode.from_char(name)


def transcribe(audio):
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
    key = _ptt_key()
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
    chunks = []
    try:
        print(f"hold {os.environ.get('HOLLER_PTT_KEY', 'alt_r')} to talk", flush=True)
        pressed.wait()
        print("listening...", flush=True)
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            while not released.is_set():
                data, _ = stream.read(SAMPLE_RATE // 10)
                chunks.append(data.copy())
    finally:
        listener.stop()
    if not chunks:
        return ""
    print("transcribing...", flush=True)
    return transcribe(np.concatenate(chunks).reshape(-1))
