# Luz as a teacher: what the learner's first two calls showed (2026-11-13)

Calls: `runs/live-20261113-211815` (3.5 min, level 2) and `runs/live-20261113-213237` (3.3 min, level 1). Both are
vault only, because they hold his voice. His side was transcribed again afterwards: the whole echo-cancelled mic
track, Whisper large-v3-turbo with 8 s chunks, auto and forced Spanish. That is how we know what he really said
where the live transcript was garbage.

## How he speaks

- **He asks a lot.** About a third of his turns are German or English meta questions:
  - what it means ("What means this?", "What means 'vos tenés razón'?", "Was heißt das nochmal?", "Was? Was? Noch
    mal auf Deutsch?")
  - how to say something ("Wie sage ich, ich hatte viel Arbeit?", "Wie sage ich nichts? Ich habe ein bisschen
    ferngeschaut.", "How do I say I miss you? Te extraño?")
  - steering the talk ("Ask me again what I was doing today.", "Okay, starten wir nochmal von neu.", "Can we start
    from you?")
- **He learns by repeating** what she just taught: "Se dice. Sí. Mucho trabajo.", "Se dice nada. Se dice nada.",
  "Así se dice nada.", "Te extraño?". Luz answered his repetition with "¡No! Vos tenés que decir." and he got
  confused ("no").
- **His Spanish** is 1–4 words a turn: "mi día es bien", "Sí, bien", "Todo bien. Muy bien.", "No planes de
  semana", "soy mucho triste", "Bien, ¿cómo estás tú?". He pauses in the middle (5 joined turns).
- **Recurring mistakes:**
  - ser for states and feelings: "es bien", "soy (mucho) triste", "es muy bien en Paraguay"
  - mucho for muy: "soy mucho triste"
  - the first-person ending: "no comprende" for comprendo
  - the verb dropped in questions: "¿Cómo tu estudio?", "No planes de semana"
- **What he doesn't understand:** past-tense questions ("¿Qué hiciste hoy?") until they are said in German.
  Sentences longer than about six words (level 2: "Soy de Asunción, pero vivo cerca de Luque con mi mamá…").

## Where DUET failed him

- **Whisper on short mixed turns.** Language detection on 1–3 s of speech (padded to 24 s) flips the language:
  - "¿Cómo estás tú?" came out as "como haistest du"
  - "Was? Was? Noch mal auf Deutsch?" came out as "Buzz? Buzz? No more of touch."
  - "What means 'vos tenés razón'?" came out as "What means boost in this rason?"
  - Luz then explained "Boost".
- **Her mistakes:**
  - She modelled an error herself: "¿Vos sos bien?"
  - She wrote a fake "Das heißt: ¿Qué hiciste hoy?" (Spanish) in a normal turn.
  - She pushed the same invitation (tereré, chipa) four times.
- **The coach corrected garbled or mixed turns** ("haistest → llamas", "of touch → de eso") and missed the
  ser/estar pattern.

## What changed (amiga.py, spanish_ears.py)

1. **Ears:**
   - When the automatic pass hears English that isn't one of his meta questions, a German-only Whisper server
     hears the turn again ("Buzz? Buzz?…" was German).
   - A word he asks about is matched to her last sentence by sound: "boost in this rason" becomes "vos tenés
     razón".
2. **Explain:** her last sentence or a word in German ("Das heißt: …"), then simple Spanish. More ways to ask are
   understood ("what this means", "auf Deutsch", "übersetzt", a bare "Was?").
3. **Say it / repeat it:**
   - "How do I say X?" gets "Se dice: <whole short sentence>." plus "Decilo vos." That becomes the phrase to
     practise.
   - When he repeats it, she praises him in two words and uses it in a tiny question.
   - When he gets it nearly right, she says it once more, slowly.
4. **Start again:** "von vorne", "start again" and the like make her start the talk again, simply.
5. **His profile** (`data/amiga/learner.json`, private):
   - It holds his recurring mistakes (with the right form) and the phrases he asked for or practised.
   - Each call's prompt carries three of them as today's practice. She brings each one in once, as a question he
     can answer with it.
   - The profile is updated after every call, from the coach's tips and what was practised.
6. **Goodbye:** "Hoy aprendiste: nada, te extraño." before her goodbye.
7. **Her Spanish:**
   - estar for feelings and states (never "sos bien").
   - Present-tense questions at level 1.
   - No repeated invitations.
   - No German in normal turns (the app does the explaining).
8. **Coach:** it skips turns that aren't mostly Spanish and knows the ser/estar and muy/mucho patterns.

## Call 3 (`runs/live-20261114-145620`, 5 min, 2026-11-14)

- **"Please translate."** became his way to ask, six times. Each time she translated well, then stopped, and he
  had to steer: "Ask me something.", "Mi bebe, give me a compliment."
- **His Spanish got longer:** "No, no, no. Es muy bonito. Mi energía es perfecta.", "Por trabajo, es mucho trabajo
  para mí.", "invierto mucho tiempo en ello" (a card of his).
- **"cansado means"** wasn't heard as a question about meaning ("Se dice: Estoy cansado.").
- **He asked about her Spanish:** "But is it a common answer, my day was tired?" He was right: she had said "Mi día
  está cansado".
- **The practice phrases were forced in:** "No bailo bien. Vos tenés razón.", "¿Vi un poco de tele hoy?",
  "Contigo me siento en casa".

So the talk now runs on **conversation lessons** (`lessons.py`): the app asks the questions, and she always ends on
the next one. "X means" is heard as a question about meaning. A question about the Spanish gets a German teacher
answer. The practice phrases left the prompt (the lessons' answers practise them), and only his usual mistakes stay.
The orb page shows the call.

## Call 4 (2026-11-14, 5 min, lessons + orb)

- **The lesson flow worked.** "Wie sage ich, ich werde einen Film schauen?" got "Se dice: Voy a ver una película.
  Decilo vos.", he said it, she praised him and asked the next question.
- **But she let his mistakes pass:**
  - "sí un poco estoy mucho trabajo" (tengo)
  - "Come la papa dulce." (comí)
  - "Sí, extraño mucho." (te extrañé)
  - "Mi bailar es mucho malo."
- **His German answers** ("Ich bin sehr schlecht… zwei P's") got English from her: "The app says you have two
  left feet."

Now every answer is checked before she replies:
- **Right:** gold and the next question.
- **Wrong:** red, the corrected sentence, the rule in German, and the same question again, up to 3 tries.

Asked inside her persona, the check passed all of the mistakes above. As a teacher request of its own, it caught
"estoy → tengo", "come → comí" and "extraño → te extrañé", and let the right answers through (replayed on Gemma 12B).
