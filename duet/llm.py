"""Mouth 2, the words: a small local language model that writes what the reflexes decided, nothing more.

Qwen3 4B Instruct 2507 (Apache-2.0, Q4_K_M, 2.5 GB) served by Ollama on this laptop's GPU (Radeon 8060S): first token
62-77 ms with the conversation prefix cached, ~57 tokens/s, so a one-sentence reply is ready in ~0.3 s, while the
opener the reflexes chose is already playing. Replies stream sentence by sentence into the voice; a cancelled reply
closes its request at once. Standard library only (urllib).
"""
import json
import re
import urllib.error
import urllib.request

from .voice import sentences as split

URL = "http://127.0.0.1:11434"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"

_EMOJI = re.compile("[\U00010000-\U0010ffff☀-➿️‍]")
_STAGE = re.compile(r"\*[^*]{0,40}\*|\([^)]{0,40}\)|\[[^\]]{0,80}\]")  # *laughs*, (smiles), [note]
# words that start a sentence: after an opener with a comma ("Hmm, maybe...") they go to lower case, names don't
STARTERS = {"that", "that's", "this", "it", "it's", "maybe", "yeah", "well", "so", "the", "a", "an", "you", "you're",
            "we", "they", "there", "there's", "here", "here's", "what", "what's", "how", "why", "who", "when",
            "where", "if", "just", "honestly", "probably", "no", "yes", "sure", "definitely", "totally", "let's",
            "not", "like", "one", "my", "your", "some", "all", "oh", "ah", "okay", "and", "but", "or", "because",
            "actually", "really", "anyway", "good", "great", "nice", "cool", "wow", "sounds", "seems", "looks",
            "glad", "happy", "love", "same", "true", "fair", "exactly", "right", "absolutely", "of", "for", "to",
            "in", "on", "at", "with", "as", "every", "then", "now", "still", "even", "also", "only", "see", "big",
            "birthday", "thanks", "thank", "hope", "have", "do", "does", "did", "is", "are", "was", "can", "could",
            "sorry", "ooh", "ha", "wait", "hey", "aw", "alright", "please", "guess", "honestly"}
SENTENCE_END = re.compile(r"(?:[!?]+|(?<!\.)\.(?!\.))[\"')]*\s+|\n+")  # "..." trails off; it doesn't end one
# what a friend shouldn't call the caller (the model says "man" even when told not to)
_ADDRESS = re.compile(r"^(?:man|dude|bro|buddy|pal),\s*|,\s*(?:man|dude|bro|buddy|pal)(?=[.!?]|$)", re.I)
_ASCII = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "—": ", ", "–": ", ",
                        "…": "..."})


def clean(text):
    """What the voice can say: no emojis, stage directions, markdown or fancy punctuation."""
    text = _STAGE.sub(" ", _EMOJI.sub("", text.translate(_ASCII)))
    text = re.sub(r"[#*_`~>|]", "", text)
    text = re.sub(r"\s+([,.!?])", r"\1", re.sub(r"\s+", " ", text)).strip()
    text = re.sub(r",(\s*,)+", ",", text)  # "like,, the movie" after a dropped "..."
    return _ADDRESS.sub("", re.sub(r"^[,.;:\s]+", "", text))


class Llm:
    def __init__(self, model=MODEL, url=URL, keep_alive="60m", load_timeout=60):
        """`load_timeout`: seconds the first request may wait while Ollama loads the model (a 12B on this laptop's
        ROCm can take minutes the first time)."""
        self.model, self.url, self.keep_alive = model, url, keep_alive
        try:
            with urllib.request.urlopen(f"{url}/api/version", timeout=3) as r:
                self.version = json.load(r)["version"]
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"Ollama is not running at {url} (start the Ollama app): {e}") from e
        with self._post([{"role": "user", "content": "Say hi."}], False, 2, 0.2, timeout=load_timeout) as r:
            json.load(r)  # load the model onto the GPU

    def _post(self, messages, stream, max_tokens, temperature, fmt=None, timeout=60):
        body = {"model": self.model, "messages": messages, "stream": stream, "keep_alive": self.keep_alive,
                "options": {"temperature": temperature, "num_predict": max_tokens, "top_p": 0.9,
                            "repeat_penalty": 1.1, "repeat_last_n": 256}}
        if fmt:
            body["format"] = fmt
        req = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout)

    def stream(self, messages, cancelled=lambda: False, max_tokens=70, temperature=0.65, max_sentences=2):
        """Yield the reply one spoken sentence at a time (cleaned), as soon as each is complete; at most
        `max_sentences` (the request is closed there: spoken replies stay short). A last message with the
        assistant's role is continued, not repeated (the opener DUET already said)."""
        buf, n, spoke, comma = "", 0, False, False

        def go_on(piece):  # the rest of a sentence whose first clause went ahead stays in lower case
            nonlocal comma
            if comma and piece.split(" ", 1)[0].lower().strip(",.!?") in STARTERS:
                piece = piece[0].lower() + piece[1:]
            comma = piece.endswith(",")
            return piece

        with self._post(messages, True, max_tokens, temperature) as r:
            for line in r:
                if cancelled():
                    return
                if not line.strip():
                    continue
                d = json.loads(line)
                buf += d.get("message", {}).get("content", "")
                while True:
                    m = SENTENCE_END.search(buf)
                    if not m and n == 0 and not spoke:
                        # nothing said yet: a first clause of 5+ words goes ahead at its comma, so the voice can start
                        # while the rest is written (the friend call had 0.3-0.8 s of silence after the opener)
                        c = re.search(r",\s", buf)
                        if c and len(buf[:c.start()].split()) >= 5:
                            head, buf = buf[:c.end()], buf[c.end():]
                            clause = clean(head)
                            if re.search(r"[A-Za-z0-9]", clause):
                                spoke = True
                                yield go_on(clause)
                            continue
                    if not m:
                        break
                    head, buf = buf[:m.end()], buf[m.end():]
                    for s in split(clean(head)):
                        if re.search(r"[A-Za-z0-9]", s):
                            yield go_on(s)
                            n += 1
                            if n >= max_sentences:
                                return
                if d.get("done"):
                    break
        for s in split(clean(buf)):  # what is left when the model stopped
            if re.search(r"[A-Za-z0-9]", s) and not cancelled() and n < max_sentences:
                yield go_on(s)
                n += 1

    def complete(self, messages, max_tokens=120, temperature=0.2, fmt=None, cancelled=None):
        """The whole answer as text, or None if `cancelled()` said so while it was being written."""
        if cancelled is None:
            with self._post(messages, False, max_tokens, temperature, fmt) as r:
                return json.load(r).get("message", {}).get("content", "")
        out = ""
        with self._post(messages, True, max_tokens, temperature, fmt) as r:
            for line in r:
                if cancelled():
                    return None
                if line.strip():
                    d = json.loads(line)
                    out += d.get("message", {}).get("content", "")
                    if d.get("done"):
                        break
        return out
