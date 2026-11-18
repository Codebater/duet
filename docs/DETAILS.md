# DUET: the full manual

The long version of [the README](../README.md): every decision, threshold and result.


A full-duplex speech-to-speech system for Windows, built from small models that each make one decision in
milliseconds. It listens while it speaks, tells a greeting from a request, reads mood from the tone of your voice and
your words, and answers in the same mood and energy. Everything runs on this laptop; nothing goes to a server.

Since 2026-11-03 its default persona is **a friend who listens** (`duet/friend.py`); the first one, Sona at
GRAPHIQ's front desk, is still there (`--persona frontdesk`).

## The friend

Small models decide, a small language model only writes. For every turn, in a few milliseconds:

| Decision | How | What it does |
|---|---|---|
| **Feeling** | the feeling words (and, with text reflex v6, its feeling head: five levels from very negative to very positive) + the prosody head's arousal measured against the caller's own usual level, smoothed over the call (a mood holds through a hard topic; it moves both ways alike) | one of 7 voices: excited, warm, lively, relaxed, calm, serious, gentle (`mood.match`): the caller's own energy and feeling, except anger (answered slower, softer, lower); something nice is met warmly, only big news gets "That's amazing!" |
| **Mode**, what they want now | text reflex act + patterns (`friend.mode_of`) | `listen` (good news, bad news, a story), `chat` (about you, banter), `help` (an answer, ideas, only when asked), a ritual (hi, how are you, thanks, bye), or `safety` |
| **Move**, the listening technique | rules from `docs/research-good-listener.md` | name and validate a bad feeling (never "at least"); be happy with good news and ask about the details, and stay on it; pick up a specific detail of a story; sum up every 5 listening turns; after 3 hard turns with no request, ask "ideas, or just get it off your chest?"; never a third question in a row |
| **Opener**, the first sound | pre-rendered per voice, for every kind of reply ("Oh no.", "No way!", "Mm.", "Hmm,", "Yeah,", "Aw,") | the reply starts the moment the turn ends; the language model continues it (handed over as the start of its own reply), and its first clause is spoken as soon as it reaches a comma |
| **Backchannels** while you talk | a pause of 0.4-0.6 s after 1 s of speech, when the pitch just moved by 30 Hz or dipped below your 26th percentile; 4-10 s apart | "mm-hm", "yeah", or "oh no" / "oh nice!" by the feeling so far, 5 dB under a reply; never in a question |
| **Endpoint** | text reflex "finished?" + the recogniser's punctuation | a question or a closing act gets an answer after 0.3 s, a finished statement 0.5-0.6 s, one the recogniser ended with a full stop at most 0.8 s, an unfinished one 1.1 s |

The words come from **Qwen3 4B Instruct 2507** (Apache-2.0, Q4_K_M, 2.5 GB) through Ollama on the laptop's GPU:
first token 62-77 ms, ~57 tokens/s, so the first sentence is ready while the opener plays. It sees the conversation
(last 8 exchanges), a short note with the decisions above, and what DUET remembers about you. Replies are capped at
two sentences (three for ideas or a summary) and cleaned of emojis, stage directions and "man"/"dude". It is told
it's an AI friend: it has tastes and opinions but no life of its own to invent, never pretends to know a film or a
person it doesn't, and reacts to what's new instead of recapping. A goodbye needs a goodbye word.

**Every call starts fresh** (a universal friend, so calls compare). **Memory** is off unless asked for with
`--remember`: then, after each turn, the model notes facts worth remembering (people with names, plans, worries), and
at the end of a call it writes a one-line summary and the question to open the next call with ("Did you get to
finish that movie?"). Both are kept in `data/friend/memory.json` (private: not in git, synced to the vault).

**Natural variation.** Glad to be talking: lively by default, excited for anything good, gentle or serious when the
caller is down. Rituals and short turns are whole pre-rendered lines, rotated so none comes back soon: greetings
("Hey! So good to hear from you!"), its own "how are you" answers and ways of asking back, "tell me more" nudges for
a bare "mm", "Yes, sure, gladly! What's up?" for "can you help me?", thank-yous, "go ahead"s when cut off, and the
murmurs while the caller talks ("mm-hm", "uh-huh", "ooh!", "aw"). Jokes when asked. The model's first sentence is
written again if it repeats something said recently.

**Safety.** Talk of self-harm or danger stops friend mode: a fixed, calm reply points to the local emergency number
or a crisis line. The model is not used for it.

**Run it**

```
ollama pull qwen3:4b-instruct-2507-q4_K_M          # once
.venv/Scripts/python -m duet live                   # talk; every call is recorded to runs/live-<date-time>/
.venv/Scripts/python -m duet live --remember        # the friend remembers you across calls
.venv/Scripts/python -m duet text                   # type instead of talking: shows every decision
.venv/Scripts/python tools/review_call.py           # the moments you marked in the last call
```

**Keys during a live call** (click the terminal first, press `n` once: a "mark note" line shows the keys arrive;
every other key is echoed too): `SPACE` stops DUET while it talks, or makes it answer now
while you talk; `g` good, `b` bad, `t` wrong tone or energy, `l` should have listened, `c` should have talked back,
`x` missed the context, `m` misheard me, `i` interrupted me, `s` too slow, `n` note; `q` ends the call. Each mark is
saved with what DUET had decided at that moment, so we can fix exactly those.

## Spanish practice partners (`--persona novia` / `cita`)

Two characters for practising conversational Spanish (the phrases from `../spanish-cards/cards.txt`). Both are from
Asunción and speak simple Paraguayan Spanish with vos:

- `novia`: **Luz**, a girlfriend on an evening phone call. She knows you from your "Über mich" cards. Voice `clara`.
- `cita`: **Sofía**, a woman you have just met at a café. Getting-to-know-you and flirting practice. Voice `aurora`.

```
.venv/Scripts/python -m duet live --persona novia      # talk (laptop mic + speakers; a headset is cleaner)
.venv/Scripts/python -m duet live --persona cita --voice vera
.venv/Scripts/python -m duet text --persona novia      # type instead: a German tip after every sentence
.venv/Scripts/python -m duet sim novia-charla          # a scripted Spanish call, no microphone
```

- **What they know:** each call puts 36 of your card phrases (from her decks) into the prompt, so you hear them in
  real talk. Ask in German ("Wie sagt man ...?") and she gives you the Spanish ("Se dice: te extraño.").
- **Level** (`--level 1|2`, default 1 since the first live call: "she is too advanced for my Spanish"):
  - Level 1 (A1): sentences of at most six words, present tense, slower voice (speed 0.85), 0.95 s endpoint.
  - Level 2 (A2): what the first call had.
- **Explaining:** "What means this?", "¿Qué significa ...?", "Was heißt das?" or "no entiendo" make her translate
  her last sentence. A request of its own does the translating: inside the conversation Gemma "translated" into
  Spanish or English.
  - She says it in German first, in the German voice (`Voice(lang_for=...)`): "Das heißt: Ich vermisse dich auch."
  - Then she says it again in very simple Spanish, slower.
  - A word you ask about comes as "Extrañar." / "Das heißt: vermissen.".
  - "Más despacio" / "otra vez" make her say it again, simpler and slower.
- **Teacher** (from the author's first two calls; analysis and design in `docs/amiga-teacher.md`):
  - "How do I say …?" gets "Se dice: <whole sentence>. Decilo vos." When you repeat it she praises you
    (≥ 0.72 by sound); a near miss gets the phrase once more, slowly.
  - Your profile `data/amiga/learner.json` (private) feeds the prompt: your usual mistakes (ser/estar,
    muy/mucho, …) and three phrases to practise today. It grows after every call from the coach's tips and
    what was taught or repeated.
  - "Starten wir nochmal von neu" starts the talk again. Her goodbye begins with "Hoy aprendiste: …".
  - The coach skips mixed German/English turns.
  - The ears hear odd "English" again with a German-only Whisper ("Buzz? Buzz? No more of touch." was
    "Was? Was? Noch mal auf Deutsch?").
- **Conversation lessons** (`duet/lessons.py`, after call 3: "I ask when I don't understand, then I understand and
  would like to keep going"):
  - Each lesson is a run of questions on one topic, each with its German and an answer to borrow, partly from his
    cards. Luz: Tu día, Lo que te gusta, Planes, Tu trabajo. Sofía: Conocerse, Gustos, Una cita.
  - The app asks the questions, Gemma only writes her one-sentence reaction, and she always ends on the question
    he can answer next:
    - his answer → reaction + next question
    - "Please translate" / "What means …" → German + the same question again
    - "No sé qué decir" → "Podés decir: … / Das heißt: … / Decilo vos."; repeating it → "¡Perfecto!" + next question
    - "Ask me something" → next question
    - a question about the Spanish ("is it common to say …?") → a short German teacher answer, then the question
  - The next call goes on with the lesson after the last one he finished (`next_lesson` in his profile).
- **Corrections** (after call 4: "if I say something wrong she should turn red and correct me, then ask again the
  question until I answer right"):
  - Every answer to a lesson question is checked before she replies.
  - The check is a teacher request of its own (`CHECK_SYSTEM`). Inside her persona, Gemma let every mistake of
    call 4 pass.
  - It is checked twice: `checked_tip`, then the second look.
  - **Right:** her praise (gold on the orb) and the next question.
  - **Wrong:** the orb turns red, and she says "Casi." / "Se dice: <his answer, corrected>" / "Tipp: <rule in
    German>" / "Otra vez: <the question>". The card shows ~~Come~~ → Comí.
  - After 3 wrong tries she gives the answer and goes on.
  - If the check takes longer than 0.8 s, she says "A ver..." first. A correction takes ~3 s, a right answer ~1.7 s.
  - An answer in German gets "Se dice: <his answer in Spanish>. Decilo vos.", and his repetition is checked too.
- **The orb** (`duet/orb.py`, `duet/web/orb.html`): `live` opens http://127.0.0.1:4330 (`--no-orb` to skip;
  `sim --orb` serves it without opening).
  - Luz is an orb that pulses with her voice (pink), listens (green), ripples while you talk and swirls while she
    thinks.
  - Her sentence shows under it, with German explanations in blue, and below that your last sentence.
  - The lesson card shows the question in Spanish and German. An answer to borrow fades in after 6 s of quiet.
  - Server-sent events from a local-only HTTP server (127.0.0.1), standard library only.
- **Mistakes:** she never points them out, she just says the right form back. A coach checks your sentences after
  the call and prints German tips (`yo voy -> yo fui (Vergangenheit ...)`), saved to `data/amiga/` (gitignored).
  Only checkable tips are kept. The wrong words must be in what you said, and foods, places and names are never
  "mistakes". A second look has to pick the corrected sentence over yours. The tips are hints: about 4 in 5 are
  right, and explanations are short or left out.
- **Hearing:** Whisper large-v3-turbo (`~/.cache/whisper/ggml-large-v3-turbo-q5_0.bin`, override with
  `DUET_WHISPER_MODEL`) in ONE long-running ffmpeg 8.1 (whisper filter, on the GPU), fed raw audio on stdin, one
  24 s chunk per turn: ~0.7 s a turn. Language is auto, so German questions stay German. It starts at your first
  0.3 s pause. TurboCTC still times the turn.
  - A new ffmpeg per turn took 1.4–1.9 s. pywhispercpp's wheel is CPU-only and took 6–12 s, so it isn't used.
- **Voice:** Supertonic-3 with the `es` tag (`Voice(lang="es")`). All Paperwire voices pass Whisper in Spanish.
- **Words:** Gemma 3 12B when Ollama has it (`amiga.BETTER_MODELS`, or set `DUET_AMIGA_MODEL`), else Qwen3 4B.
  - The friend's 4B is unloaded first: with both loaded, Gemma took minutes to load. Alone it loads in ~12 s.
  - Her system prompt is read while the greeting plays.
  - The 4B makes its own slips ("al Costanera", "¡Sos muy divertida!") and parrots prompt examples.
- **Timing (sim, 2026-11-13, Gemma):** 1.9–2.1 s from your last word to her first sound: 0.75 s endpoint, the
  rest of Whisper, the model's first sentence, the voice. The English partners answer in ~0.3 s.

**First live call** (2026-11-03, `runs/live-20261103-212305`, 2:48, a movie chat): no talking over, no
self-hearing, 4 backchannels, 9.5 dB echo cancellation, memory kept the name and the movie. It led to: energy against
the caller's own level (the prosody head read every turn as aroused), mild vs big news, openers for every reply, the
first clause spoken early (0.3-0.8 s silences after the opener), the full-stop endpoint (median reply 784 ms), the
honest-AI persona (it invented "watching the sky outside my window"), no recaps ("as I already told you"), and the
goodbye-word rule ("Thank you. Let's speak." read as a goodbye).

## The front desk (first persona)

It is Reflexx's rule applied to a whole conversation: **decide, don't generate.** Every model reads a fixed-size
input in one pass and picks; the only generator is the voice. The domain is Sona, GRAPHIQ's front desk, because the
Reflexx intent data is.

```
 mic 48k ─► decimate ─► ECHO CANCEL ─► VAD ─┬─► ASR (partials every 320 ms) ─► TEXT REFLEX ─┐
              ▲          (400 ms NLMS,     │      TurboCTC 470M, DirectML       one bge-small   │
              │           two-path)        │                                    pass, 5 heads   │
              │                            └─► TONE (pitch, level, rate) ─► MOOD HEAD ─────────┤
              │                                                                                ▼
 speaker ◄── PLAYER ◄── VOICE (Supertonic-3, our Ivy voices, ◄── BRAIN (call context, ◄── TURN-TAKER
  (duck,      (fade,      mood = register: pace, pitch lift)       greeting vs intent,      listen / caller /
   stop)       cut log)   openers pre-rendered                     replies, mood policy)    speak / overlap
```

## The small models

| Part | Model | Size | Time | Measured |
|---|---|---|---|---|
| Self-hearing | own two-path partitioned-block NLMS (`aec.py`) | 0 params | <1 ms / 16 ms block | 12-17 dB cancellation in the simulated room; **0 false barge-ins on echo alone, the caller caught 260 ms in** (`tests/test_aec.py`) |
| Voice activity | Silero VAD v5 (MIT) | 2 MB | ~0.1 ms / 32 ms | |
| Words | Granite Speech 5.0 TurboCTC (Apache-2.0), fp16 DirectML, from frontdesk-bench | 0.9 GB | 37-45 ms | word-perfect on our TTS; 3.3 % WER on phone audio in the Reflexx paper |
| Text reflex | bge-small-en-v1.5 (MIT) + 5 heads, trained here (`tools/train_text_reflex.py`) | 133 MB | ~5-10 ms CPU | **intent 90.6 % / 89.4 % (phone ASR) on the 310 Reflexx test calls** (paper's bge-base: 88.4 %); act 91 % and greeting 100 % on 58 unseen phrasings, 21 / 23 on held-out feelings and statements; sentiment 89 %; "finished?" 79-93 % (`models/text-reflex-v4/REPORT.md`, which compares v1-v4) |
| Tone | YIN pitch with an octave-jump fix, level, pauses, rate, tilt vs the caller's own baseline (`prosody.py`) + 12-16-6 MLP trained on CREMA-D (`tools/train_prosody_head.py`) | 2 KB | ~10-15 ms / turn | **arousal 79 % (AUC 0.88) on unseen actors**, 6 emotions 46 % (chance 17 %) (`models/prosody-head/REPORT.md`, head v3) |
| Voice | Supertonic-3 (OpenRAIL-M) + our 7 Ivy-builder voices (default Orion) | 380 MB | 3.7 s of speech in 0.75 s on 4 CPU threads | pitch-lift direction verified on Orion and Vera; the "liveliness" direction failed its check and is off (`models/mood-voice/REPORT.md`) |

The text reflex's five heads: **intent** (26 front-desk intents), **act** (request, greeting, small talk, backchannel,
stop, repeat, continue, thanks, goodbye, affirm, deny), **flags** (contains a greeting / contains a request, both
can be true), **sentiment** and **complete** (does the sentence sound finished).

## How it behaves

- **Listening while speaking.** While DUET talks, the echo canceller removes its own voice and scores what's left
  against the echo it expected. When the caller's voice holds for most of 224 ms, DUET ducks 12 dB at once and reads
  the overlap: "mm-hm" or "yeah" (a backchannel) and it carries on at full level; "wait", "sorry, what?" or a new
  question and it stops within 80 ms. It remembers the sentence it was cut in, so "go on" resumes there.
- **Greeting vs intent.** "Hi" gets a greeting back and an invitation; "Hi, how much is a store?" gets a short
  greeting back and the answer; "How much is a store?" gets the answer. Hello is said once per call (a later "hi"
  gets "Hi again"), and "how are you" gets an answer. Low intent confidence gets "Just to check, is this about …?",
  and a "yes" then answers it.
- **What any front desk answers.** The date and time (read off the clock in Florida, where Sona is), the caller's
  name (asked for, then remembered), "can you hear me", and "what can you do". A question it can't answer is
  admitted ("Hmm, I can't help with that one") with what it can do instead; a feeling said on its own is met
  ("That's great to hear", "I'm sorry about that"); a bare "okay" gets room ("I'm listening"). No line is said twice
  in a row: invitations and fallbacks rotate.
- **Mood.** Arousal (calm to excited) comes mostly from the tone, valence (negative to positive) mostly from the
  words, both smoothed over the call. Moods sour fast and recover slowly. The reply meets the caller and lifts: a
  happy caller gets the same energy and a little more warmth (bright register); a neutral caller warm; a frustrated
  caller is not mirrored but gets steady and calm, empathy first; a low caller gentle. Each register is a pace, a
  pitch lift along the measured style direction (Orion: +1 semitone when bright), and a level.
  The bright register needs positive words as well as energy: tone alone can't tell excited from annoyed.
- **Semantic endpointing.** After you stop, DUET answers after 0.25 s if you asked a question (the recogniser
  writes the "?") or said a short closing act ("hi", "thanks", "okay", "wait"), 0.5 s after a finished statement
  (people go on after a sentence), 0.55 s if unsure, 0.9 s if it sounds cut off ("I'd like to …"). It never ends a
  turn while you're still talking: turns longer than the recogniser's 10.24 s are transcribed in pieces cut at
  pauses (up to 25 s), and the text reflex reads a long turn from its end, where the question is.
- **Going on is one turn.** If you start again just as DUET decided to answer, or talk over the first 1.2 s of its
  answer without stopping it ("wait", "what?"), its answer is dropped whole (also what was being rendered or queued)
  and your words join your last turn: the brain reads both as one, from its state before the dropped answer. DUET's
  own sound never plays while it is listening to you.
- **Instant first sound.** Every sentence a reply can start with, and every answer's first line, is rendered per
  register and cached on disk (`models/opener-cache`), so the reply starts as soon as the decision is made; the rest
  renders while it plays.

## Measured end to end (simulated laptop)

`python -m duet sim <script>` runs the real threads on a 20 ms clock. DUET's output goes through a simulated room
(55 ms delay, 0.3 s reverb, speaker distortion) back into its own microphone, mixed with a scripted caller voiced
by Supertonic.

| Script | What happens | Reply latency* |
|---|---|---|
| `front-desk` | greeting, store price, "mm-hm" over DUET (kept), "Wait, how long…?" over DUET (stopped, answered), thanks, bye | 320-352 ms |
| `greet-vs-intent` | "Good morning, how are you?", SEO question, "Hold on, hold on" (stopped), "Sorry, go on" (resumed at the cut sentence), bye | 320 ms |
| `frustrated` | complaint (greeting + empathy + steady register), "real person?", "Okay, thank you" (calm, not bright) | 320-576 ms (the complaint ends in a statement: 0.5 s wait) |
| `long-talker` | the first live call's failure: a 20 s monologue with sentence pauses and a 0.4 s pause before more; one turn, transcribed whole, answered once at the end (empathy + timeline) | 320-416 ms |

`python tools/sim_check.py runs/sim-<script>-vN ...` checks a run for what the first live call got wrong: seconds
DUET played while the caller had the floor, DUET's own words inside a caller's transcript, and the latencies.
2026-11-03 (`runs/sim-long-talker-v5`, `sim-front-desk-v8`, `sim-frustrated-v6`, `sim-greet-vs-intent-v6`): 0.0 s
talked over and no echo in any of the four. (`sim-long-talker-v1`, `sim-front-desk-v5`, `sim-frustrated-v3` and
`sim-greet-vs-intent-v3` are crashed runs from a log-encoding bug fixed in `__main__.py`; kept, not valid.)

\* From the caller's last voiced frame to DUET's first sample leaving the player. On this laptop's speakers add the
60 ms WASAPI output latency (measured: 40 ms in, 60 ms out, no dropouts in a 3 s silent open).

Recordings and logs: `runs/sim-*/` (`call_mic_L_duet_R.wav`: left what DUET heard, right what it said;
`heard_after_echo_cancel.wav`; `call.json`).

## Run it

```
py -3.13 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m duet sim front-desk --out runs/my-run      # no microphone needed
.venv/Scripts/python -m duet live                                   # talk to it; Ctrl+C to stop
.venv/Scripts/python -m duet live --voice vera --record runs/live-1
.venv/Scripts/python -m duet devices
```

`DUET_DEBUG=1` prints every turn-taking state change. The first start renders 160 opener lines (~2-7 min,
depending on load); later starts load them from disk in about a second.

**Models** (in `models/`, synced to the vault, not in git): `turboctc/` (from frontdesk-bench's crate),
`vad/silero_vad.onnx` (snakers4/silero-vad v5.1.2), `supertonic3/` (HF Supertone/supertonic-3, from spark-2's cache),
`voices/` (our private folio-voices styles), `bge-small-en-v1.5/` (HF BAAI, for training only), and trained here:
`text-reflex-v4/` (front desk), `text-reflex-v6/` (friend), `prosody-head/`, `mood-voice/`; `opener-cache/` holds the
rendered openers.

**Tests:** `tests/test_aec.py` (echo cancellation and barge-in evidence), `tests/test_brain.py` (greeting vs intent,
interruption memory, mood policy).

## Not done or not verified

- **One live call so far** (2026-11-03, `runs/live-1`, 2 minutes on the laptop's speakers). Turn-taking held: 13 of
  19 replies 288-352 ms after the caller stopped, 6 of 6 talk-overs real, barge-in and backchannels right, although
  the laptop's speakers and microphone array allow only ~5 dB of echo cancellation (12-17 dB simulated). What broke
  there is fixed but not yet heard live: the 9.5 s turn cap and the echo leak it led to, the replies (re-greeting,
  "Ha, fair enough!" for any question, repeated lines), "I'm very excited" read as thanks, and octave jumps in the
  pitch track.
- **The tone reading is trained on acted speech** (CREMA-D, 91 actors, 12 sentences). On the live call its arousal
  followed real loudness and pitch, but its labels lean to angry and disgust for plain speech (it confuses happy
  with angry), so DUET uses the arousal and only a small valence hint, and goes bright only on positive words. In
  the simulations the caller is a TTS voice, so the tone readings there mean nothing.
- **English only** (TurboCTC and bge-small); Supertonic itself speaks cs, sk, de, es and more.
- **Replies are fixed lines** (`duet/replies.py`), without prices: Sona points to graphiq.art for those.
- The end-of-sentence pitch slope is still missing on half the CREMA-D clips (too few voiced frames in the last
  500 ms); the other feature issues in the mood head's report are fixed in head v3.

## Licences

Supertonic-3 is OpenRAIL-M: its use restrictions apply, including no impersonation (our voices are designed, not
cloned). TurboCTC Apache-2.0; Silero VAD MIT; bge-small MIT; CREMA-D ODbL 1.0 / DbCL 1.0 (the mood head is a
produced work). The voice styles are private (folio-voices) and must stay out of any public repo.
