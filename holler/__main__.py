"""hold hotkey -> whisper -> route -> jev-ultrafast -> say. Loops until Ctrl-C."""

import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from .listen import abort_on_ptt_release, listen
from .route import decide, route
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


def _agent_cls():
    """HOLLER_AGENT=jev keeps the thin browser-harness driver; default is Jev on browser-use."""
    if os.environ.get("HOLLER_AGENT", "bu") == "jev":
        from jev_ultrafast import Agent

        return Agent
    from .jevbu import BUAgent

    return BUAgent


def _load_env(path=".env"):
    for line in Path(path).read_text().splitlines() if Path(path).exists() else []:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _run_agent(agent):
    """Run one agent phase; returns (last_state, aborted)."""
    listener, abort = abort_on_ptt_release()  # tap PTT key to abort
    state = None
    try:
        if os.environ.get("HOLLER_FOREGROUND"):
            from browser_harness.helpers import cdp

            cdp("Target.activateTarget", targetId=agent.browser.target)
        for state in agent.run():
            print(state["elapsed_ms"], len(state["history"]), state["status"], flush=True)
            if abort.is_set():
                break
    except KeyboardInterrupt:
        abort.set()
    finally:
        listener.stop()
    return state, abort.is_set()


def _fail(e):
    print(f"agent failed: {e}", flush=True)
    say(f"failed, {e}" if isinstance(e, (RuntimeError, ValueError)) and len(str(e)) < 80 else "failed")


def run_once(transcript, last):
    print(f'heard: "{transcript}"', flush=True)
    task = route(transcript, context=last or None)
    if task is None:
        say("didn't get that")
        return
    if "clarify" in task:
        say(task["clarify"])
        last["request"] = transcript
        last["pending"] = task["clarify"]
        if task.get("url"):
            last["url"] = task["url"]
        return
    print(f"-> {task['url']}")
    for i, g in enumerate(task["goals"], 1):
        print(f"  {i}. {g}")
    record_dir = None
    if os.environ.get("HOLLER_RECORD"):
        record_dir = Path("runs") / str(int(time.time()))
        print(f"recording -> {record_dir}")

    state, agent = None, None
    try:
        agent = _agent_cls()(task["url"], task["goals"], record_dir=record_dir)
        state, aborted = _run_agent(agent)
    except Exception as e:
        if agent is not None:
            agent.close()
        _fail(e)
        return
    if aborted:
        agent.close()
        print("aborted", flush=True)
        say("stopped")
        return
    # Blocked with zero actions on an empty page = the site never loaded. Don't
    # ask decide() to plan against a blank page.
    if state["status"] == "blocked" and not state["history"] and not (state["page"].get("text") or "").strip():
        agent.close()
        print("page did not load", flush=True)
        say("the page didn't load")
        return

    # observe -> decide phases: the model sees the real page and picks the next
    # step (or done / clarify). Same tab unless the model wants another site.
    # A phase that adds no actions means the page can't advance — stop looping.
    answer = None
    for _ in range(3):
        prev_len = len(state["history"])
        nxt = decide(transcript, state["page"], state["history"], state["status"])
        if nxt is None:
            break
        if nxt.get("status") == "done":
            answer = nxt.get("answer")
            break
        if "clarify" in nxt:
            agent.close()
            say(nxt["clarify"])
            last["url"] = (state.get("page") or {}).get("url") or task["url"]
            last["request"] = transcript
            last["pending"] = nxt["clarify"]
            return
        goals = nxt["goals"]
        print("next:")
        for g in goals:
            print(f"  - {g}")
        if nxt.get("url") and urlparse(nxt["url"]).netloc != urlparse(state["page"]["url"]).netloc:
            agent.close()
            try:
                agent = _agent_cls()(nxt["url"], goals, record_dir=record_dir)
            except Exception as e:
                _fail(e)
                return
        else:
            agent.state["goal"] = "\n".join(goals)
            agent.state["status"] = "ready"
        try:
            state, aborted = _run_agent(agent)
        except Exception as e:
            agent.close()
            _fail(e)
            return
        if aborted:
            agent.close()
            print("aborted", flush=True)
            say("stopped")
            return
        if len(state["history"]) == prev_len:
            print("no progress this phase; stopping", flush=True)
            break

    if agent is not None and os.environ.get("HOLLER_KEEP_TAB"):
        try:  # jev pins 1120x780; unpin so a kept tab fits the real window
            from browser_harness.helpers import cdp

            cdp("Emulation.clearDeviceMetricsOverride", session_id=agent.browser.session)
        except Exception:
            pass
    elif agent is not None:
        agent.close()
    if state:
        last.clear()
        last["url"] = (state.get("page") or {}).get("url") or task["url"]
        last["request"] = transcript
        if answer:
            say(answer)
        else:
            speak_result(state, transcript)


def main():
    _load_env()
    last = {}  # follow-up context: {"url", "request"} from the previous run
    try:
        if len(sys.argv) > 1:  # uv run holler "open github" -- one shot, no mic
            run_once(" ".join(sys.argv[1:]), last)
            return
        while True:
            transcript = listen()
            if transcript:
                run_once(transcript, last)
    finally:
        if os.environ.get("HOLLER_AGENT") != "jev":
            from .jevbu import shutdown

            shutdown()


if __name__ == "__main__":
    main()
