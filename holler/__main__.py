"""hold hotkey -> whisper -> route -> jev-ultrafast -> say. Loops until Ctrl-C."""

import os
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


def main():
    _load_env()
    while True:
        transcript = listen()
        if not transcript:
            continue
        print(f'heard: "{transcript}"', flush=True)
        task = route(transcript)
        if task is None:
            say("didn't get that")
            continue
        print(f"-> {task['url']}")
        for i, g in enumerate(task["goals"], 1):
            print(f"  {i}. {g}")
        state = None
        try:
            with Agent(task["url"], task["goals"]) as agent:
                for state in agent.run():
                    print(state["elapsed_ms"], len(state["history"]), state["status"], flush=True)
        except Exception as e:
            print(f"agent failed: {e}", flush=True)
            say("failed")
            continue
        if state:
            speak_result(state, transcript)


if __name__ == "__main__":
    main()
