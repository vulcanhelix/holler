"""hold hotkey -> whisper -> route -> jev-ultrafast -> say. Loops until Ctrl-C."""

import os
import sys
import time
from pathlib import Path

from jev_ultrafast import Agent

from .listen import listen
from .route import route
from .speak import say, speak_result

# Cerebras rejects jev's OpenRouter-style `reasoning` param with HTTP 400;
# qwen models also need reasoning_effort=none or they burn max_tokens thinking.
import jev_ultrafast.model as _jev_model

_jev_post_json = _jev_model.post_json


def _post_json_no_reasoning(url, key, body):
    if "cerebras" in url:
        body = {k: v for k, v in body.items() if k != "reasoning"}
        if "qwen" in body.get("model", ""):
            body["reasoning_effort"] = "none"
    return _jev_post_json(url, key, body)


_jev_model.post_json = _post_json_no_reasoning


def _load_env(path=".env"):
    for line in Path(path).read_text().splitlines() if Path(path).exists() else []:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def run_once(transcript):
    print(f'heard: "{transcript}"', flush=True)
    task = route(transcript)
    if task is None:
        say("didn't get that")
        return
    print(f"-> {task['url']}")
    for i, g in enumerate(task["goals"], 1):
        print(f"  {i}. {g}")
    record_dir = None
    if os.environ.get("HOLLER_RECORD"):
        record_dir = Path("runs") / str(int(time.time()))
        print(f"recording -> {record_dir}")

    for attempt in range(2):
        state, agent = None, None
        try:
            agent = Agent(task["url"], task["goals"], record_dir=record_dir)
            if os.environ.get("HOLLER_FOREGROUND"):
                from browser_harness.helpers import cdp

                cdp("Target.activateTarget", targetId=agent.browser.target)
            for state in agent.run():
                print(state["elapsed_ms"], len(state["history"]), state["status"], flush=True)
        except Exception as e:
            if agent is not None:
                agent.close()
            print(f"agent failed: {e}", flush=True)
            say("failed")
            return
        # Blocked with zero actions = the SPA hadn't rendered yet. Retry once.
        dead = state is not None and state["status"] == "blocked" and not state.get("history")
        if dead or not os.environ.get("HOLLER_KEEP_TAB"):
            agent.close()
        if dead and attempt == 0:
            print("page was empty on load; retrying once", flush=True)
            continue
        if os.environ.get("HOLLER_KEEP_TAB") and agent is not None:
            try:  # jev pins 1120x780; unpin so a kept tab fits the real window
                from browser_harness.helpers import cdp

                cdp("Emulation.clearDeviceMetricsOverride", session_id=agent.browser.session)
            except Exception:
                pass
        break
    if state:
        speak_result(state, transcript)


def main():
    _load_env()
    if len(sys.argv) > 1:  # uv run holler "open github" -- one shot, no mic
        run_once(" ".join(sys.argv[1:]))
        return
    while True:
        transcript = listen()
        if transcript:
            run_once(transcript)


if __name__ == "__main__":
    main()
