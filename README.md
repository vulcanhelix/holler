# holler

Say what you want done in Chrome. The browser does it in a background tab, then reads the result back. No keyboard.

Voice -> [mlx-whisper](https://github.com/ml-explore/mlx-examples) -> Cerebras routes to url+goals -> [jev-ultrafast](https://github.com/browser-use/jev-ultrafast) drives your real Chrome over CDP -> Cerebras summarizes -> `say`.

## Setup

1. `brew install uv`
2. `uv sync`
3. `cp .env.example .env` — fill `TYPESAFE_API_KEY`, `CEREBRAS_API_KEY`, `TEXT_MODEL_API_KEY` (same Cerebras key)
4. `uv run holler` — grant Input Monitoring and allow Chrome remote debugging when prompted
5. Hold Right-Option, speak, release. `uv run browser-harness --doctor` if the browser won't attach

While an agent is working, **tap the PTT key** (or Ctrl-C) to abort — the tab closes and it says "stopped". Ctrl-C at the prompt quits holler.

`uv run holler "open github"` runs one cycle with a typed command — no mic needed.

Non-Chrome Chromium (Thorium, Arc, Edge, Brave): relaunch with `--remote-debugging-port=9222` and set `BU_CDP_URL=http://127.0.0.1:9222` in `.env`. Plain Chrome just needs the toggle at `chrome://inspect/#remote-debugging`.

`uv run pytest` runs the routing tests. Edit `aliases.toml` to teach it your site names.

On Apple Silicon, speech-to-text is mlx-whisper; elsewhere it falls back to faster-whisper on CPU (same `HOLLER_WHISPER_MODEL` knob).
