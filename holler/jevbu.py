"""Jev judgment on browser-use hands.

jev_ultrafast.model.choose() decides the next action; browser-use's
BrowserSession supplies the DOM and executes it. Exposes the same surface as
jev_ultrafast.Agent: Agent(url, goals) -> .run() yields state dicts, .close(),
.browser.target/.session for tab control.
"""

import asyncio
import base64
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

os.environ.setdefault("BROWSER_USE_LOGGING_LEVEL", "warning")

from browser_use import BrowserSession
from browser_use.browser.events import (
    ClickElementEvent,
    NavigateToUrlEvent,
    ScrollEvent,
    SelectDropdownOptionEvent,
    SwitchTabEvent,
    TypeTextEvent,
)
from browser_use.browser.profile import BrowserProfile
from jev_ultrafast.model import choose, field_context, field_text
from jev_ultrafast.questions import MAX_STEPS

OBSERVE_TIMEOUT = float(os.environ.get("HOLLER_OBSERVE_TIMEOUT", "12"))

FILL_TAGS = {"input", "textarea"}
CLICK_TYPES = {"button", "submit", "checkbox", "radio", "file", "image", "reset", "range", "color", "date"}
FILL_ROLES = {"textbox", "searchbox", "combobox", "spinbutton"}

CONTROLS = [
    {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
    {"id": "scroll_up", "kind": "scroll", "label": "Scroll up", "delta": -560},
    {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
]


_shared = None  # (loop, session)


def _get_session():
    global _shared
    if _shared is None:
        loop = asyncio.new_event_loop()
        session = BrowserSession(
            cdp_url=os.environ.get("BU_CDP_URL", "http://127.0.0.1:9222"),
            browser_profile=BrowserProfile(
                cross_origin_iframes=False,
                max_iframes=8,
                max_iframe_depth=1,
                paint_order_filtering=False,
                highlight_elements=False,
                wait_for_network_idle_page_load_time=0.25,
            ),
        )
        try:
            loop.run_until_complete(session.start())
        except Exception:
            _shared = None
            try:
                loop.close()
            except Exception:
                pass
            raise
        _shared = (loop, session)
    return _shared


def shutdown():
    """Stop the shared browser-use session (call once on holler exit)."""
    global _shared
    if _shared is None:
        return
    loop, session = _shared
    _shared = None
    try:
        loop.run_until_complete(session.stop())
    except Exception:
        pass
    try:
        loop.close()
    except Exception:
        pass


class BUAgent:
    def __init__(self, url, goals, *, record_dir=None, screenshots=None):
        task = goals.strip() if isinstance(goals, str) else "\n".join(goals).strip()
        if not task:
            raise ValueError("Supply a task")
        self._loop, self.session = _get_session()
        self._current_map = {}
        self._masked = set()
        self._last_fp = None
        self.record_dir = Path(record_dir) if record_dir else None
        self.screenshots = bool(screenshots or record_dir)
        try:
            self._arun(self._dispatch(NavigateToUrlEvent(url=url, new_tab=True)))
            self._sync_tab()
            # browser-use no-ops the nav (warning only) when agent focus is mid-recovery,
            # and recovery can also park focus on another live tab — confirm our tab
            # really loaded the requested site before touching the page.
            for _ in range(4):
                try:
                    landed = self._arun(self._eval("location.href")) or ""
                except Exception:
                    landed = ""
                if self.browser.target and urlparse(landed).netloc == urlparse(url).netloc:
                    break
                self._arun(asyncio.sleep(0.4))
                self._arun(self._dispatch(NavigateToUrlEvent(url=url, new_tab=True)))
                self._sync_tab()
            else:
                raise RuntimeError(f"navigation to {url} did not land")
            self._settle()
            page = self._observe()
        except Exception:
            self.close()
            raise
        self.state = dict(
            browser=self.browser,
            goal=task,
            page=page,
            decision=None,
            history=[],
            status="ready",
            elapsed_ms=0,
            started_at=None,
        )
        if self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            self._record(page)

    # -- browser-use plumbing ------------------------------------------------

    def _arun(self, coro):
        return self._loop.run_until_complete(coro)

    def _sync_tab(self):
        """Expose the agent tab's CDP target/session for foreground + unpin."""
        target_id = getattr(self.session, "agent_focus_target_id", None)
        session_id = None
        try:
            cdp_session = self._arun(self.session.get_or_create_cdp_session())
            target_id = target_id or getattr(cdp_session, "target_id", None)
            session_id = getattr(cdp_session, "session_id", None)
        except Exception:
            pass
        self.browser = type("Tab", (), {"target": target_id, "session": session_id})()

    async def _eval(self, expression):
        # Bind to our tab, not agent focus — focus can be stolen by recovery or the user.
        target_id = getattr(getattr(self, "browser", None), "target", None)
        cdp_session = await self.session.get_or_create_cdp_session(target_id, focus=False)
        result = await cdp_session.cdp_client.send.Runtime.evaluate(
            params={"expression": expression, "returnByValue": True},
            session_id=cdp_session.session_id,
        )
        return result.get("result", {}).get("value")

    async def _dispatch(self, event):
        return await self.session.event_bus.dispatch(event)

    def _settle(self, budget=float(os.environ.get("HOLLER_SETTLE_S", "4"))):
        """Wait until body text is non-empty and stable across two polls, or the budget runs out."""
        deadline, prev = time.perf_counter() + budget, None
        while time.perf_counter() < deadline:
            try:
                got = self._arun(
                    self._eval(
                        "document.body ? [document.body.innerText.length,"
                        " document.querySelectorAll('a,button,input,select,textarea,[role]').length]"
                        " : [0, 0]"
                    )
                )
                n, inter = got if isinstance(got, (list, tuple)) and len(got) == 2 else (0, 0)
            except Exception:
                n, inter = 0, 0
            # SPA shells hold a tiny stable DOM before hydrating; only a substantive
            # page counts as settled.
            if (n, inter) == prev and (n >= 200 or inter >= 10):
                return
            prev = (n, inter)
            self._arun(asyncio.sleep(0.25))

    def _observe(self):
        # Keep focus pinned to our tab — browser-use auto-recovers to ANY open tab on detach.
        if self.browser.target:
            try:
                if getattr(self.session, "agent_focus_target_id", None) != self.browser.target:
                    self._arun(self._dispatch(SwitchTabEvent(target_id=self.browser.target)))
            except Exception:
                pass
        # browser-use's clean-screenshot path stalls on this CDP attach; grab it raw instead.
        summary, text, shot = self._arun(self._observe_wait())
        # jev's snapshot only offered onscreen elements; cap near-viewport nodes so the
        # TypeSafe payload stays small enough for the API to accept.
        self._current_map = {
            i: n for i, n in (summary.dom_state.selector_map or {}).items() if self._near_viewport(n)
        }
        actions = []
        for index, node in sorted(self._current_map.items()):
            actions.extend(self._node_actions(index, node))
        page = {
            "url": summary.url,
            "title": summary.title,
            "text": text[:8000],
            "actions": actions,
            "screenshot": shot,
        }
        page["fp"] = self._fingerprint(page)
        page["actions"] = actions[:200] + [dict(c) for c in CONTROLS]
        return page

    async def _observe_wait(self):
        # Retrying must wait on the SAME task — cancelling a mid-flight DOM build
        # wedges the CDP session, so a restarted observe would time out too.
        task = asyncio.ensure_future(self._observe_parts())
        for _ in range(2):
            done, _ = await asyncio.wait({task}, timeout=OBSERVE_TIMEOUT)
            if done:
                return task.result()
        task.cancel()
        raise RuntimeError("page observation timed out")

    async def _observe_parts(self):
        shot_coro = self._screenshot() if self.screenshots else asyncio.sleep(0)
        summary, text, shot = await asyncio.gather(
            self.session.get_browser_state_summary(include_screenshot=False),
            self._eval("document.body ? document.body.innerText : ''"),
            shot_coro,
            return_exceptions=True,
        )
        if isinstance(summary, BaseException):
            raise summary
        return (
            summary,
            "" if isinstance(text, BaseException) else (text or ""),
            None if isinstance(shot, BaseException) else shot,
        )

    async def _screenshot(self):
        cdp_session = await self.session.get_or_create_cdp_session(self.browser.target, focus=False)
        r = await cdp_session.cdp_client.send.Page.captureScreenshot(
            params={"format": "jpeg", "quality": 72},
            session_id=cdp_session.session_id,
        )
        return r.get("data")

    @staticmethod
    def _fingerprint(page):
        return hash(
            (
                page["url"],
                page["text"],
                tuple((a["id"], a.get("label")) for a in page["actions"] if a.get("node") is not None),
            )
        )

    @staticmethod
    def _near_viewport(node):
        box = getattr(node, "bounding_box", None)
        if box is None:
            return True
        top = getattr(box, "y", 0)
        return -400 <= top <= 1400

    @staticmethod
    def _label(node):
        ax = getattr(node, "ax_node", None)
        for cand in (getattr(ax, "name", None),):
            if cand:
                return str(cand)[:120]
        attrs = node.attributes or {}
        for key in ("aria-label", "alt", "placeholder", "name", "title", "value"):
            if attrs.get(key):
                return attrs[key][:120]
        text = getattr(node, "node_value", None) or getattr(node, "get_all_children_text", lambda: "")()
        return (str(text).strip() or node.tag_name or "element")[:120]

    @staticmethod
    def _node_kind(node):
        tag = (node.tag_name or "").lower()
        role = (node.attributes or {}).get("role", "").lower()
        if tag == "select" or role in {"listbox"}:
            return "select"
        if tag in FILL_TAGS and (node.attributes or {}).get("type", "text").lower() not in CLICK_TYPES:
            return "fill"
        if role in FILL_ROLES or (node.attributes or {}).get("contenteditable") == "true":
            return "fill"
        return "click"

    def _node_actions(self, index, node):
        kind = self._node_kind(node)
        label = self._label(node)
        attrs = node.attributes or {}
        extra = {
            k: v
            for k, v in {
                "role": attrs.get("role") or node.tag_name,
                "value": attrs.get("value"),
                "checked": attrs.get("checked") or attrs.get("aria-checked"),
                "selected": attrs.get("selected") or attrs.get("aria-selected"),
                "expanded": attrs.get("expanded") or attrs.get("aria-expanded"),
            }.items()
            if v not in (None, "")
        }
        if kind == "select":
            actions = []
            for i, opt in enumerate(node.children or []):
                if (opt.tag_name or "").lower() != "option":
                    continue
                actions.append(
                    {
                        "id": f"N{index}O{i}",
                        "kind": "select",
                        "node": index,
                        "label": self._label(opt),
                        "value": (opt.attributes or {}).get("value", self._label(opt)),
                        "current_value": attrs.get("value", ""),
                        **extra,
                    }
                )
            if actions:
                return actions
            return [
                {
                    "id": f"N{index}",
                    "kind": kind,
                    "node": index,
                    "label": label,
                    "value": attrs.get("value", label),
                    **extra,
                }
            ]
        return [{"id": f"N{index}", "kind": kind, "node": index, "label": label, **extra}]

    def _act(self, action, text=None):
        kind = action["kind"]
        if kind == "wait":
            self._arun(asyncio.sleep(0.5))
            return
        if kind == "scroll":
            self._arun(self._dispatch(ScrollEvent(direction="down" if action["delta"] > 0 else "up", amount=abs(action["delta"]))))
            return
        node = (self._current_map or {}).get(action["node"])
        if node is None:
            raise ValueError(f"element {action['node']} no longer observed")
        if kind == "click":
            self._arun(self._dispatch(ClickElementEvent(node=node)))
        elif kind == "fill":
            self._arun(self._dispatch(TypeTextEvent(node=node, text=text or "", clear=True)))
        elif kind == "select":
            self._arun(self._dispatch(SelectDropdownOptionEvent(node=node, text=action["label"])))

    def _record(self, page):
        if not self.record_dir or not page.get("screenshot"):
            return
        data = page["screenshot"]
        if isinstance(data, str):
            data = base64.b64decode(data)
        (self.record_dir / f"{self.state['elapsed_ms']:06d}.jpg").write_bytes(data)

    # -- jev Agent interface ---------------------------------------------------

    def run(self):
        state = self.state
        while state["status"] not in {"done", "blocked"}:
            if state["started_at"] is None:
                state["started_at"] = time.perf_counter()
            state["elapsed_ms"] = round((time.perf_counter() - state["started_at"]) * 1000)
            page = self._observe()
            state["page"] = page
            self._record(page)
            if state["history"]:
                last = state["history"][-1]
                if page["fp"] == self._last_fp:
                    last["page_changed"] = False
                    key = (last.get("id"), last.get("action"))
                    if key not in self._masked:
                        self._masked.add(key)
                        print(f"masked: {last.get('action')}", flush=True)
                else:
                    last["page_changed"] = True
                    self._masked.clear()
            self._last_fp = page["fp"]
            # An action that left the page identical gets hidden until the page changes — kills repeat loops.
            if self._masked:
                page["actions"] = [
                    a for a in page["actions"] if (a["id"], a.get("label")) not in self._masked
                ]
            decision = choose(page, state["goal"], state["history"])
            state["decision"] = decision
            choice = decision["choice"]
            if choice in {"DONE", "BLOCKED"}:
                state["status"] = "done" if choice == "DONE" else "blocked"
                yield state
                return
            action = next(a for a in page["actions"] if a["id"] == choice)
            text = None
            if action["kind"] == "fill":
                text, helper = field_text(field_context(state["goal"], action, page, state["history"]))
                state.setdefault("text_calls", []).append({"model": helper["model"], "field": action["label"]})
            self._act(action, text)
            if action["kind"] in {"click", "fill", "select"}:
                self._settle(budget=1.5)
            old_url = page["url"]
            state["history"].append(
                {
                    "step": len(state["history"]) + 1,
                    "id": action["id"],
                    "action": action["label"],
                    "kind": action["kind"],
                    "text": text,
                    "operation": decision["operation"],
                    "confidence": decision["confidence"],
                    "latency_ms": decision["latency_ms"],
                    "page_changed": None,
                    "url": old_url,
                }
            )
            if len(state["history"]) >= MAX_STEPS:
                state["status"] = "blocked"
                yield state
                return
            yield state
        return state

    def close(self):
        try:
            if getattr(self, "browser", None) and self.browser.target:
                cdp_session = self._arun(
                    self.session.get_or_create_cdp_session(self.browser.target, focus=False)
                )
                self._arun(
                    cdp_session.cdp_client.send.Target.closeTarget(params={"targetId": self.browser.target})
                )
        except Exception:
            pass
