"""hold hotkey -> whisper -> route -> jev-ultrafast -> say. Loops until Ctrl-C."""

import os
import sys
from pathlib import Path

from jev_ultrafast import Agent

from .listen import listen
from .route import route
from .speak import say, speak_result


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
    state = None
    agent = None
    try:
        agent = Agent(task["url"], task["goals"])
        if os.environ.get("HOLLER_FOREGROUND"):
            from browser_harness.helpers import cdp

            cdp("Target.activateTarget", targetId=agent.browser.target)
        for state in agent.run():
            print(state["elapsed_ms"], len(state["history"]), state["status"], flush=True)
    except Exception as e:
        print(f"agent failed: {e}", flush=True)
        say("failed")
        return
    finally:
        if agent is not None and not os.environ.get("HOLLER_KEEP_TAB"):
            agent.close()
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
