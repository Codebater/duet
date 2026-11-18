"""The orb: a page on this laptop that shows the call. Luz as a glowing orb that pulses with her voice, her words
under it (German explanations in their own colour), what he said, and the lesson's question with its German and an
answer he can borrow.

    python -m duet live --persona novia          # opens http://127.0.0.1:4330 (--no-orb to skip)

Standard library only: a threaded HTTP server serves `web/orb.html` and an event stream (`/events`, server-sent
events). A ticker sends the state 30 times a second (listening / he talks / thinking / she speaks, her output level,
his voice level, the sentence she is saying); the conductor's log lines (what he said, what she said) and the
brain's lesson view go out as they happen. Nothing leaves the laptop: it binds 127.0.0.1 only.
"""
import json
import queue
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = Path(__file__).resolve().parent / "web" / "orb.html"
PORT = 4330
TICK_S = 1 / 30


class Orb:
    def __init__(self, conductor=None, brain=None, port=PORT, log=print):
        self.c, self.brain, self.port, self.print = conductor, brain, port, log
        self.clients = []
        self.lock = threading.Lock()
        self.last_view = None
        self.last_heard = ""
        self.server = ThreadingHTTPServer(("127.0.0.1", port), self._handler())
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True, name="orb-http").start()
        threading.Thread(target=self._tick, daemon=True, name="orb-tick").start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/"

    # ---------------------------------------------------------------- what the page hears
    def send(self, event):
        data = f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")
        with self.lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(data)
                except queue.Full:  # a page that stopped reading: dropped
                    self.clients.remove(q)

    LINE = re.compile(r'^\[\s*[\d.]+\]\s+(\S+)\s+"(.*)"\s{2}')

    def log(self, line):
        """The conductor's log line ("[  8.96] caller    "..."  act=..."): printed as before, and his words and
        hers sent to the page."""
        self.print(line)
        m = self.LINE.match(line)
        if m and m.group(1) in ("caller", "said"):
            self.send({"t": "heard" if m.group(1) == "caller" else "said", "text": m.group(2)})

    def tip(self, text):
        """The coach's tips after the call, on the page too."""
        self.print(text)
        if text.strip().startswith("Tipp:"):
            self.send({"t": "tip", "text": text.strip()[5:].strip()})

    def _state(self):
        c = self.c
        if c is None:
            return {"t": "tick", "state": "idle", "out": 0.0, "mic": 0.0}
        speaking = c.player.busy or c.speaking_reply is not None
        if speaking:
            state = "speak"
        elif c.state in ("caller", "overlap"):
            state = "caller"
        elif c.state == "think":
            state = "think"
        else:
            state = "listen"
        meta = c.player.current
        sentence, lang = "", "es"
        if speaking and isinstance(meta, tuple) and len(meta) == 2 and meta[1] >= 0:
            reply, i = meta
            if i < len(reply.sentences):
                sentence = reply.sentences[i]
                if hasattr(self.brain, "sentence_lang"):
                    lang = self.brain.sentence_lang(sentence)
        mic = getattr(c, "vad_p", 0.0) if getattr(c, "caller_talking", False) or c.state == "caller" else 0.0
        return {"t": "tick", "state": state, "out": round(min(1.0, 6 * c.player.level), 3), "mic": round(mic, 3),
                "sentence": sentence, "lang": lang}

    def _tick(self):
        while True:
            time.sleep(TICK_S)
            if not self.clients:
                continue
            try:
                self.send(self._state())
                view = dict(getattr(self.brain, "view", {}) or {})
                if view != self.last_view:
                    self.last_view = view
                    self.send({"t": "lesson", **view})
            except Exception:  # noqa: BLE001 - the page is never worth stopping the call
                pass

    # ---------------------------------------------------------------- HTTP
    def _handler(self):
        orb = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet: the call log stays readable
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = PAGE.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path.startswith("/events"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    q = queue.Queue(maxsize=600)
                    with orb.lock:
                        orb.clients.append(q)
                    view = getattr(orb.brain, "view", None)
                    if view:
                        q.put(f"data: {json.dumps({'t': 'lesson', **view}, ensure_ascii=False)}\n\n".encode())
                    try:
                        while True:
                            try:
                                data = q.get(timeout=5)
                            except queue.Empty:
                                data = b": keep-alive\n\n"
                            self.wfile.write(data)
                            self.wfile.flush()
                    except (OSError, ConnectionError):
                        pass
                    finally:
                        with orb.lock:
                            if q in orb.clients:
                                orb.clients.remove(q)
                else:
                    self.send_error(404)

        return Handler

    def close(self):
        self.server.shutdown()
