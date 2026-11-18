# Reflexx Ear: our own small streaming recogniser (design, 2026-11-04)

Goal: replace TurboCTC (Granite Speech 5.0, 470M params, 950 MB fp16, 3.3 % WER on phone audio) in DUET with our own
recogniser of **~30M params, ~15-20 MB in ONNX**, that hears each 80 ms of audio **once** as it arrives, so partial
transcripts are free and the final one is ready the moment the turn ends. Built from our own findings:

| Finding | Where | Use in the ear |
|---|---|---|
| Gated delta-rule layers + a few recall (attention) layers beat a transformer at matched params, with a tiny fixed state (8.4 MB latents vs 1.2 GB KV at 32k) | KARST pilots 1-4, `karst/research/RESEARCH_STATE.md` | the encoder: streaming by construction, constant memory per frame |
| int4 weight PTQ on the hybrid costs +0.014 nats, no training, 1.7x faster | KARST E006/E009 (`hybrid_i4h`) | ship the ear as int4 weights: 30M params ~ 15 MB. Tested on r1c: holds for the body, not for the CTC head (int4 head doubles the error); int4 body + int8 head ~ 20 MB, see "int4 test on r1c"; on r1d it costs 2-3 points, see "Rung 1d result" |
| State as graph I/O, static shapes, prefix-state caching, DirectML export traps (opset 18, no Trilu, blocked inverse) | Reflexx paper, AQUIFER rung 0 | one ONNX graph per 160 ms chunk with the layer states in and out, on the laptop's GPU |
| Decide, don't generate | Reflexx | CTC: one pass, no autoregressive decoder (unlike Moonshine or Whisper) |

## Model
- Front end: TurboCTC's own features (`duet/asr.py features`: 80 log-mels + deltas, two 10 ms frames per 20 ms row), so
  teacher and student hear exactly the same thing.
- Subsampling: 2 conv layers, 20 ms rows -> 80 ms frames (TurboCTC's CTC rate, so its posteriors align 1:1).
- Encoder: d = 384, 12 layers in KARST's hybrid pattern (3 gated delta : 1 sliding-window attention over the last
  ~2 s), SwiGLU FFNs; streaming with a bounded state.
- Head: CTC over TurboCTC's 16,384 word-pieces (same tokenizer, same decoding in DUET). ~28M params total.

## Training: distilled from TurboCTC
- Teacher: TurboCTC on the Sparks (ORT CUDA) labels every clip: greedy transcript + top-16 CTC posteriors per 80 ms
  frame. Any English speech works, transcribed or not.
- Loss: CTC on the teacher's transcript + KL to the teacher's posteriors (frame-level distillation).
- Data (commercial-use licences only): LibriSpeech 960 h (CC BY 4.0) to start; then conversational and phone-like
  speech: our own TTS (Supertonic voices, Qwen3-TTS on spark-2) over generated everyday lines, VoxPopuli EN (CC0),
  plus phone-band, noise, reverb and level augmentation.

## Rungs and gates
| Rung | What | Gate |
|---|---|---|
| 0 | data + teacher labels for LibriSpeech-100; 8M ear trained ~2 h on spark-1; WER on LibriSpeech test-clean and the 310 Reflexx calls | pipeline works; WER falls steadily; streaming = full-context within 0.5 pt |
| 1 | 28M ear on 960 h + synthetic conversational speech + augmentation | <= 6 % WER on the 310 phone calls (TurboCTC 3.3 %) |
| 2 | int4 + chunked ONNX with state I/O on DirectML; drop-in `duet/ear.py` behind a flag; live A/B against TurboCTC | <= 10 ms per 160 ms chunk on the laptop; no loss in DUET's text-reflex scores |

Not claimed: anything above until measured. TurboCTC stays DUET's recogniser until the ear passes rung 2.

## Rung 0 result (2026-11-04, spark-1)
Code in `ear/` (`features.py`, `prep.py`, `model.py`, `train.py`, `evaluate.py`, `data.py`); run copied to `runs/ear-r0/`
(checkpoint, `log.json`, `eval.json`). Data: LibriSpeech train-clean-100 (md5 checked), the last 8 speakers held out.

- **Labelling:** TurboCTC on ORT CUDA labelled 100.9 h in 17 min (375x real time): 27,649 training clips, 890
  held-out, plus the 310 phone calls. Its WER: 1.1 % on the held-out speakers (likely seen LibriSpeech in training),
  3.4 % on the calls.
- **Ear:** 9.53M params (d 256, `DDDADD`, attention window 2.56 s, 4.2M of it the 16,384-way CTC head), delay 2
  frames (160 ms), CTC + 0.5 KL, SpecAugment, 17,296 steps in 71 min (60 passes over the data).

| | held-out speakers WER | 310 phone calls WER |
|---|---|---|
| TurboCTC (teacher, 470M) | 1.1 % | 3.4 % |
| Ear r0, best (step 4-5k) | 38.0 % | 80.5 % |
| Ear r0, final (step 17.3k) | 41.0 % | 81.1 % |

- **Streaming gate: passed.** 160 ms chunks with every layer's state carried (delta layers stepped one frame at a
  time) vs one pass over the whole clip, 100 held-out clips: 42.56 % vs 42.59 % WER, 93/100 identical transcripts,
  max logit gap 0.07 (fla's chunk kernel vs the fp32 recurrence). 2.3 ms per 160 ms chunk in plain PyTorch on the
  Spark.
- **WER gate: fell, then rose.** 47 % (1k steps) -> 38 % (4k) -> 41 % (17k) while training loss kept falling: the
  ear memorises. At step 7k it scored 8.3 % WER on its own training clips vs 38 % on unheard speakers. Missed words
  by how often the word appeared in training: 18.5 % (>= 100 times), 56 % (10-99), 88 % (1-9), 97 % (never).
- **Reading:** the pipeline and the streaming architecture work; 100 h from 243 speakers is far too little for a
  16,384-piece output. The phone calls fail because the ear has only heard clean read audiobooks.

## Rung 1 plan (from rung 0)
- More speakers and data first: LibriSpeech 960 h (2,338 speakers), then speech the teacher labels for free
  (VoxPopuli EN, our TTS conversational lines in many voices), phone-band + noise + reverb + speed augmentation.
- Early stopping on the held-out set; stronger SpecAugment.
- Then the 28M ear. Re-check whether a smaller output vocabulary beats distilling the 16,384 pieces.

## Rung 1a result: the same ear on 960 h (2026-11-04, spark-1)
Run copied to `runs/ear-r1/` (checkpoints, `log.json`, `ear_best.eval.json`, `r1.log`). One change from rung 0: the data.

- **Labelling:** TurboCTC labelled the other 860 h: train-clean-360 (364.7 h) in 56 min and train-other-500 (498.4 h)
  in 76 min, ~390x real time. The last 8 train-other-500 speakers are a second held-out set, `devother` (924 clips,
  3.4 h, harder recordings; teacher 0.9 % WER). Training set: 279,427 clips, 957 h, 2,322 speakers.
- **Ear:** unchanged (9.53M params, same recipe and 2 h budget): 23,208 steps in 98 min, about 9 passes. (The budget
  is set from the first 50 steps, which were slow while the 110 GB of features were first read from disk.)

| | held-out speakers (clean) | held-out speakers (other) | 310 phone calls |
|---|---|---|---|
| TurboCTC (teacher, 470M) | 1.1 % | 0.9 % | 3.4 % |
| Ear r0, 100 h (best) | 38.0 % | - | 80.5 % |
| Ear r1a, 960 h (final = best) | **20.7 %** | **24.1 %** | **61.1 %** |

- **Data was the bottleneck.** Ten times the data, same model and time: 38.0 % -> 20.7 % on the same held-out
  speakers. No memorising this time: held-out WER fell at every evaluation (48 % at 1k steps, 25 % at 10k, 20.7 % at
  23k) and was still falling when the budget ran out.
- **Streaming gate: passed again.** 100 held-out clips in 160 ms chunks with state vs one pass: 18.51 % vs 18.56 %
  WER, 96/100 identical transcripts, max logit gap 0.11, 2.2 ms per chunk in plain PyTorch.
- **The calls are a channel and style gap.** 21 % on audiobooks but 61 % on the calls, which are phone-band, short
  and conversational. The ear often gets the shape and misses the words: "Can you put me down for a web design
  consultation? Mornings work best for me." -> "you put me down for aineation. morning for best for me". Openers
  like "Yeah hey, um" are worst: audiobooks have none.

## Rung 1b plan (from rung 1a)
- Phone channel: run the 960 h through phone-band augmentation (8 kHz resample, band-pass, codec, noise, level) and
  recompute only the student features; the teacher labels stay valid because the 80 ms frames do not move.
- Conversational speech the teacher labels for free: our TTS everyday lines in many voices (spark-2), then VoxPopuli
  EN (CC0; download needs an OK).
- The 28M ear with a longer budget: the 9.5M curve was still falling at the end.

## Rung 1b result: phone line, conversational lines, 30M ear (2026-11-04, spark-1)
Run copied to `runs/ear-r1b/` (`ear_best.pt`, `log.json`, `ear_best.eval.json`, every transcript in
`ear_best.hyps.jsonl`, the error breakdown in `ear_best.errors.txt`, `r1b.log`). New code: `channel.py`, `phone.py`,
`tts.py`, `errors.py`.

- **Phone copies:** all 279,427 training clips through a random phone line (`channel.py`): room echo (30 %), caller
  noise (40 %), 8 kHz, a band-pass with random edges (150-450 Hz to 2.9-3.8 kHz), level, mu-law / A-law / no codec,
  line noise (70 %). The test calls' one setting (300-3400 Hz, mu-law, pink noise at 12-22 dB) lies inside that range.
  Only the student features are recomputed (`phone.py`, ~12 min on 16 cores); the teacher's labels are reused.
- **Conversational lines:** 19,666 everyday lines (39 dropped by the contamination guard: word overlap >= 0.6 with a
  test call), each spoken twice by Supertonic 3 (OpenRAIL-M; the test calls are Qwen3-TTS) in 300 voices mixed from
  its 10 stock styles, at random speeds (`tts.py`): 39,332 clips, 74.9 h, ~125x real time. TurboCTC's WER on them
  against the scripts: 2.8 %, its only misses numbers spelled out in the script. Phone copies of these too.
- **Mix:** 794,846 clips, 2,364 h: the 960 h and its phone copy, plus the lines and their phone copy three times each
  (about a fifth of training is conversational).
- **Ear:** 29.77M params (d 384, `DDDADDDADDDA`, 6 heads, FFN 1024; 6.3M of it the CTC head), lr 1.5e-3, 36,383 steps
  in 328 min, just over 5 passes. On the way: the input mean and std now come from 24 shards, because a strided read of
  all 229 GB of features (more than RAM) stalled the first launch.

| | held-out clean | held-out other | 310 calls, clean | 310 calls, phone |
|---|---|---|---|---|
| TurboCTC (teacher, 470M) | 1.1 % | 0.9 % | 2.4 % | 3.4 % |
| Ear r0, 9.5M, 100 h (best) | 38.0 % | - | - | 80.5 % |
| Ear r1a, 9.5M, 960 h | 20.7 % | 24.1 % | 40.4 % | 61.1 % |
| Ear r1b, 29.8M, 2,364 h (final = best) | **14.5 %** | **17.6 %** | **21.8 %** | **26.6 %** |

- **WER gate: failed** (26.6 % vs <= 6 % on the phone calls), but the calls error fell by more than half.
- **The phone gap is mostly closed:** clean -> phone calls cost 20.7 points in r1a, 4.8 now.
- **The style gap is down by two thirds:** audiobooks -> clean calls cost 19.7 points in r1a, 7.3 now. Size and data
  changed together, so this run cannot split the credit. A clue for size: at step 23k the 30M ear scored 16.5 % on
  audiobooks vs the 9.5M ear's 20.7 %, while spending 60 % of its training on other audio.
- **Still learning, slower on calls:** from step 18k to 36k audiobook WER fell 18 % relative (17.6 -> 14.5), the phone
  calls 10 % (29.7 -> 26.6).
- **Streaming gate: passed.** 100 held-out clips in 160 ms chunks with state vs one pass: 12.05 % vs 12.05 % WER,
  100/100 identical transcripts, max logit gap 0.09, 4.2 ms per chunk in plain PyTorch.

Where the phone-call errors are (`errors.py`; 874 errors on 3,284 words):
- **Not vocabulary.** 87.5 % of the errors are on words the ear heard 10+ times in training. Unseen and rare words
  are 5.4 % of errors, numbers 1.3 %. The rung 1a guess (names and numbers) was wrong.
- **Clip edges.** First and last words are wrong 35 % of the time vs 22 % mid-clip: a quarter of all errors.
  Audiobooks show no edge effect (12 % first word vs 13 % middle). The test calls start speaking at once (median
  0-20 ms of lead-in, 97-100 % under 100 ms), while our TTS lines had a median 620 ms (1 % under 100 ms) and
  LibriSpeech 180 ms. The ear starts every clip from an empty state, so it has rarely had to catch a first word cold.
- **Small words and word pieces.** The commonest misses: and <-> in, a <-> the, dropped "do", "is", "a", homophones
  (site -> sight x8), words that lose a piece (tomorrow -> morrow, today -> day, someone -> some). Context fixes
  these; greedy CTC uses none.
- **Fillers:** "um" dropped 16 times (2.5 % of errors); the teacher drops them too.
- **Reading:** the call gap is the ear's general weakness on common words (82 % of the audiobook errors too), made
  worse by abrupt clip edges and the phone line. Mid-clip, a 6-point style gap remains (13.3 % audiobooks, 19.0 %
  clean calls, 22.4 % phone calls).

## Rung 1c plan (from rung 1b)
- A word language model in the decoder first, on the r1b checkpoint, no retraining: CTC beam search with a small
  n-gram on conversational and LibriSpeech text (contamination guard applied). Measures how much is decoding.
- Clip edges: train on clips with little or no lead-in and tail (cut on 80 ms steps, so the labels stay aligned), and
  also score the calls the way DUET hears them: one stream with line noise between turns, not each clip from an empty
  state.
- More, and more varied, speech for the teacher to label (VoxPopuli EN and others, each download needs an OK; more
  permissively licensed TTS engines), and a ~24 h budget.
- Stay at 30M (int4 ~15 MB); grow to 60-80M only if these stall.

## Rung 1c, step 1 result: a language model in the decoder (2026-11-04, spark-1)
Same r1b checkpoint, no retraining. New code: `lm.py` (our own interpolated Kneser-Ney word 4-gram), `decode.py` (CTC
prefix beam search, width 16, each finished word scored alpha x ln P + beta), `leadin.py`. Outputs in `runs/ear-r1b/`
(`ear_best.decode.json`, `decode.log`, every transcript in `ear_best.decode.hyps.jsonl`, `ear_best.decode.errors.txt`).
One fix on the way: a word is scored only once the next one starts, so a spelling that can no longer become a known
word ("thecat", "ation") now ranks as an unknown word at once. Without that, prefixes that glued words together dodged
the LM, crowded out the beam, and tripled the WER.

- **Language models, no test text in either.** Books: the LibriSpeech transcripts of the training audio (279,427
  sentences, 9.3M words, 15.5M n-grams). Lines: 17,700 of our conversational lines (335k words), with 1,966 more held
  out. Mix: 0.9 lines + 0.1 books, the weight with the lowest perplexity on the held-out lines (7, vs 1,000 for books
  alone: the lines are formulaic). Leak check: the books contain 17 % / 16 % of the held-out audiobook 4-grams and 8 %
  of the calls'.
- **Tuning.** alpha and beta (6 x 5 grid) picked on 600 held-out audiobook clips: alpha 0.3, beta 2 for both models.
  Tuned on the phone calls themselves, the grid picks the same setting, so the call numbers carry no tuning on the test.
- **Lead-in.** The calls with 0.5 s before and after (`leadin.py`). Phone: through frontdesk-bench's own phone line,
  its noise running on through the padding (speech part matches the official clips to >= 49.6 dB). Clean: digital
  silence.

| decoder | held-out clean | held-out other | calls, clean | calls, phone | phone with lead-in |
|---|---|---|---|---|---|
| greedy (rung 1b) | 14.5 % | 17.6 % | 21.8 % | 26.6 % | 26.6 % |
| beam, no LM | 13.6 % | 16.9 % | 20.6 % | 25.5 % | 25.2 % |
| beam + books LM | **10.6 %** | **12.9 %** | 20.4 % | 25.0 % | 24.5 % |
| beam + mix LM | 11.9 % | 14.5 % | **15.5 %** | **19.9 %** | **18.4 %** |

- **Decoding was a quarter of the call errors:** 26.6 -> 19.9 % on the phone calls, 21.8 -> 15.5 % clean. The WER
  gate (<= 6 %) is still failed.
- **The call gain comes from the conversational text.** Audiobook text helps audiobooks (-27 %) and barely moves the
  calls; the 335k words of lines do it. The front desk has a known domain, so this fits Sona. The listening friend's
  talk is broader, so expect less there.
- **Lead-in helps the first word, not the last.** With the line running 0.5 s before the caller speaks (how DUET hears
  a turn), first words go from 26 % to 19 % wrong (beam + mix LM), WER 19.9 -> 18.4 %. Last words stay at 28 % with or
  without the tail, so the clip end is not a cold-state problem; more likely it is how utterance-final words sound
  ("only by card?" -> "only by c.", "the thirty minute one" -> "the thirty minute it was"). Digital silence as lead-in
  made the clean calls worse (21.8 -> 23.8 % greedy): the ear never heard pure zeros before speech, and a live
  microphone never produces them.
- **What is left is acoustic.** After the LM, mid-clip words are 16 % wrong. The commonest misses are still is -> if,
  site -> sight, and <-> in, a <-> the, someone -> some / one, "Friday" -> "fighting", and "PayPal" -> "paper" (in
  neither model's vocabulary).

## Rung 1c, next (from step 1)
- More and more varied audio for the teacher to label, then a longer run (the audiobook curve was still falling).
  Needs download OKs: VoxPopuli EN (CC0), more permissively licensed TTS engines for conversational lines.
- Abrupt-start training moves down the list: lead-in recovers part of the first-word excess, and inside DUET the ear
  always has one.
- Keep the LM: books + lines for general speech, with a line set per persona (front desk, friend) as the domain part.
- The beam runs frame by frame, so it streams; its cost on the laptop is not measured yet.

## Rung 1c result: VoxPopuli and a 24 h run (2026-11-06, spark-1)
Run copied to `runs/ear-r1c/` (`ear_best.pt` = step 136,000, `log.json`, `ear_best.eval.json`, `ear_best.decode.json`,
every transcript in `ear_best.hyps.jsonl` and `ear_best.decode.hyps.jsonl`, both error breakdowns, `r1c.log`, the lost
first run in `r1c-first-run.log`, the launch and scoring scripts `r1c.sh` and `r1c_score.sh`). New code:
`voxpopuli.py`, `prep.py --first-shard`, `phone.py` over several roots, `--resume` in `train.py`.

- **VoxPopuli EN** (CC0, European Parliament speeches): 182,482 clips, 524 h, labelled by the teacher, plus a phone
  copy. TurboCTC scores 7.5 % against the official transcripts, part of it their style ("EUR 2" for "two euros").
- **Mix:** 1,081,146 clips, 3,263 h: the 960 h and VoxPopuli, each with its phone copy, plus the conversational lines
  and their phone copy twice each. Conversational audio fell from 19 % of training (r1b) to 9 %.
- **Ear:** the same 29.77M ear and lr 1.5e-3, 150,000 steps with a cosine to 0.1x, 16 passes over the mix, 24.1 h.
  Best audiobook score at step 136,000.
- **Lost first run:** spark-1 is shared, and its GPU memory is the machine's RAM. A ComfyUI job ran out of memory
  and the first run died at step 56,150 after 8.1 h; it started again from zero. `train.py` now writes `resume.pt` at
  every eval (weights, optimizer, data position, random state), and `--resume` goes on from it. Tested: a resumed run
  matched an unbroken one loss for loss.

| | held-out clean | held-out other | VoxPopuli test | 310 calls, clean | 310 calls, phone | phone with lead-in |
|---|---|---|---|---|---|---|
| TurboCTC (teacher, 470M) | 1.1 % | 0.9 % | 7.6 % | 2.4 % | 3.4 % | - |
| r1b greedy | 14.5 % | 17.6 % | 46.7 % | 21.8 % | **26.6 %** | 26.6 % |
| r1c greedy | **13.0 %** | **15.8 %** | **22.4 %** | **20.3 %** | 27.5 % | **23.8 %** |
| r1b beam + mix LM | 11.9 % | 14.5 % | 42.3 % | 15.5 % | **19.9 %** | 18.4 % |
| r1c beam, no LM | 12.3 % | 14.9 % | 21.0 % | 19.2 % | 26.5 % | 22.9 % |
| r1c beam + books LM | **9.5 %** | **11.5 %** | **19.7 %** | 18.9 % | 25.7 % | 21.5 % |
| r1c beam + mix LM | 10.6 % | 12.8 % | 20.8 % | **14.6 %** | 21.7 % | **18.1 %** |

- **WER gate: failed** (18.1 % on the phone calls as DUET hears them, vs <= 6 %).
- **More speech helped everything but the cold phone clips.** Audiobooks -10 % (greedy), VoxPopuli halved (now in
  training), clean calls -7 % greedy and -6 % with the mix LM.
- **The phone calls only improved with the line running first.** Cut cold (the official clips), they sat at 27-29 %
  greedy from step 50,000 on while audiobooks kept falling, and end at 27.5 % vs r1b's 26.6 %: more first and middle
  words wrong, 319 words dropped vs 243. With 0.5 s of line before and after (how DUET hears a turn) r1c is better at
  every position: greedy 26.6 -> 23.8 %; with the mix LM first words 19.4 -> 18.7 %, middle 14.4 -> 12.6 %, last
  words 28.7 -> 21.9 %. r1b gained nothing from the lead-in greedy, r1c gains 3.7 points: it leans on hearing the line before
  the speech. Inside DUET the line is always running, so the lead-in column is the one that counts.
- **The LM's word bonus, tuned on audiobooks, now overshoots on calls.** The held-out audiobooks picked alpha 0.5,
  beta 4 for the mix LM (r1b: 0.3, 2). On the phone calls that adds words: 157 extra vs r1b's 62, 57 of them at the
  clip start ("I" x22, "we" x13: how our lines open). At r1b's setting the phone calls score 20.8 %; tuned on
  themselves with lead-in, 17.2 %. Tuning on audiobooks no longer transfers to calls.
- **The curve has flattened.** Audiobooks 13.5 % at step 120k, 13.0 % at 136k, 13.3 % at 150k, with the learning
  rate at its floor. A longer run of this ear on this mix will not move much.
- **Calls stalled early, and they are mostly acoustic.** Each conversational clip was replayed about 64 times (r1b
  about 30), at half r1b's share of training, and the calls stopped improving by step 50k. 300 mixed Supertonic voices
  over 19.7k lines are likely learned by then: the calls need more varied conversational audio, not more passes or
  more read speech. The commonest misses with lead-in and the mix LM: a -> the, is -> if, um -> ah, I'm -> am, we're ->
  where, someone -> one, site -> sight.
- **Streaming gate: passed.** 100 held-out clips in 160 ms chunks with state vs one pass: 10.24 % vs 10.24 % WER,
  99/100 identical transcripts, max logit gap 0.10, 4.2 ms per chunk in plain PyTorch.

## Rung 1c, next (from the 24 h run)
- **Free:** tune alpha and beta on spoken held-out lines (the 1,966 held-out lines through TTS and the phone line,
  with lead-in), not on audiobooks. Check whether the ear has memorised its training lines: WER on training TTS clips
  vs fresh TTS clips of held-out lines.
- **The recipe:** 13 % on held-out audiobooks after 3,263 h is weak for a 30M recogniser, with the curve flat. Test a
  smaller output vocabulary (the 16,384-piece CTC head is 6.3M of the 29.8M params and every frame has to choose
  among all of them): a short A/B at a fixed budget against the current head.
- **Data:** more varied conversational speech, back to at least a fifth of training: more TTS engines and voices with
  permissive licences, and real conversational speech where the licence allows it (each download needs an OK).
- Grow to 60-80M only after the recipe test.

## Rung 1c checks: memorisation, and the LM tuned on spoken lines (2026-11-06, spark-1)
Same r1c checkpoint (step 136,000), no retraining. Outputs in `runs/ear-r1c/checks/` (`memo.sh`, the three line sets,
every transcript, `memo.memo_report.txt`, `memo.ear_best.decode.json`, `calls_supertonic.*`).

- **Sets, all in voices the ear never heard** (`tts.py render --seed 1`: 300 new mixes of Supertonic's ten styles, one
  take per line), each clean and through a random phone line (`phone.py`):
  - 600 of the 19,666 lines the ear trained on;
  - 1,558 lines it never heard, from DUET's validation text (text reflex and friend), test-call guard applied. Our
    text is templated: 965 share >= 60 % of their words with a training line ("near"), 593 do not ("far", like the
    test calls);
  - the 310 test-call scripts themselves, spoken by Supertonic instead of Qwen3-TTS (a diagnostic only, nothing
    tuned on it).

| greedy WER vs the script | clean | phone | teacher | clips |
|---|---|---|---|---|
| training clips (trained line, trained voice) | 3.0 % | 3.3 % | 2.8 % | 39,332 |
| trained lines, new voice | 3.8 % | 4.9 % | 2.6 % | 600 |
| new lines near a trained one | 4.3 % | 5.9 % | 2.9 % | 965 |
| new lines far from all | 7.1 % | 10.3 % | 2.4 % | 593 |
| the 310 call scripts, Supertonic voices | 8.7 % | 10.8 % | 2.7 % | 310 |
| **the 310 test calls, Qwen3-TTS voices** | **20.5 %** | **27.4 %** | 2.4 / 3.4 % | 310 |

- **Memorised, but that is not what fails.** The ear reproduces its training clips about as well as the teacher
  labelled them (3.0 % vs 2.8 %). A new Supertonic voice costs little (3.8 %); new lines cost more (7.1 % far). All of
  that is far below the calls.
- **The calls fail on their voices, not their words.** The same 310 scripts score 8.7 % in Supertonic voices and
  20.5 % in the Qwen3-TTS voices, with no phone line in either. The teacher finds both equally easy (2.7 / 2.4 %).
  LibriSpeech's and VoxPopuli's thousands of speakers did not carry over to these voices; the only conversational
  speech the ear has heard is in Supertonic's ten styles.
- **The LM tuned on spoken lines changes nothing.** Tuned on the 593 far lines through the phone line, the mix LM picks
  alpha 0.5, beta 4, the setting the audiobooks picked. Calls: 21.7 % phone, 17.8 % with lead-in, 14.7 % clean (vs
  21.7 / 18.1 / 14.6 % tuned on audiobooks; reshuffling the LM's held-out tenth moves these by about 0.3). On the far
  lines themselves it takes 10.3 -> 7.5 %. A setting learned on voices the ear knows does not fit the voices it does
  not: there it drops opening words and the word bonus fills them with "I" and "we". Decoding cannot tune past this.

## Rung 1c, next (from the checks)
- **Voices first.** Conversational speech in many more voices than Supertonic's ten styles. Qwen3-TTS VoiceDesign is
  already on spark-2 (no download), but it rendered the test calls, so training on it would make the 310 calls a test
  of one engine. So first a test set of real voices: our own DUET calls (the author's `runs/live-*`, and a teammate's with his OK),
  the teacher's transcript checked by hand. Then any engine is fair for training: Qwen3-TTS with hundreds of designed
  voices, and other permissively licensed engines (download OKs).
- The vocabulary A/B stays on the list for the audiobook gap (13 % on held-out speakers), after the voices.

## Real-voice test set (2026-11-06, references checked by ear)
`ear/realcalls.py`: every caller turn from the author's 8 DUET calls of 2026-11-03 (`runs/live-*`), cut from what DUET
heard after echo cancelling: a laptop microphone, Czech-accented English, real conversation. 99 of 103 turns, 4.9 min,
829 words, in `runs/ear-real/` (the vault, not git: it is the author's voice). On spark-1 in `~/ear/data/real_draft`
(owner-only; the checked set in `~/ear/data/real`), scored with `~/ear/real_score.sh real`.

- **Cutting:** call.json logs when a caller turn ended, not when it began. DUET's Silero VAD on the echo-cancelled
  audio finds the speech; while DUET's own channel plays (and 0.4 s after), a frame counts as the caller's only if it
  is 6 dB above the echo DUET's level predicts. The turn is the speech since the previous caller turn or backchannel,
  pauses of up to 2 s included, plus 0.3 s each side. 4 turns had nothing separable from DUET's echo (one-word turns
  said over DUET). About 5 clips, most from the first call on laptop speakers, still hold a few of DUET's words.
- **References:** the teacher's transcript of each clip, being checked by ear on a local page (`realcalls.py review`,
  http://localhost:4310): fix the words, drop clips with DUET's voice or cut mid-word. The teacher's draft and DUET's
  live transcript agree word for word on 63 of 99 clips (11.5 % apart, mostly where a live turn was cut differently).
  the author's check: 96 kept (3 references corrected), 3 dropped (DUET's voice). Four kept references are doubtful
  (`1_02` holds DUET's words, `1_10`, `1_12`, `1_15` read oddly); without them the ear's WER moves by 0.1-0.5 points.
  The teacher's 1.7 % against the checked references is flattering: they started as its own drafts.

| | real voice, 96 turns | synthetic phone calls, 310 |
|---|---|---|
| TurboCTC (teacher) | 1.7 % | 3.4 % |
| r1b greedy / beam + mix LM | 54.9 % / 51.3 % | 26.6 % / 19.8 % |
| r1c greedy / beam + mix LM | **49.7 % / 46.4 %** | 27.5 % / 21.9 % |

(Scored against the unchecked drafts the numbers were within 0.4 points of these.)

- **On a real voice the ear gets half the words wrong**, twice its error on the synthetic calls. r1c's extra speech
  helped (-5 points), the LM barely (-3).
- **Everywhere in the turn, on common words:** first word 61 % wrong, middle 46 %, last 43 %; 97 % of the words are
  common ones. Most errors are dropped words (192 of 413): "Hey bro, how are you?" -> "Ab are you?", "I missed
  completely our appointment" -> "I may prof completely important".
- **Caveats:** one speaker, 5 minutes; each clip starts cold with 0.3 s before the speech, while inside DUET the ear
  would run on through the whole call.
- **Reading:** this is the number that matters for DUET, and the ear is far from replacing TurboCTC on it. It also
  frees the training data: with real voices as the test, any TTS engine is fair to train on, Qwen3-TTS included.
  Report this set at every rung from now on, next to the 310 calls.

## Rung 1d plan (from the real-voice test)
The ear fails on voices it has not heard: 8.7 % on the call scripts in Supertonic voices, about 20 % in the test's
Qwen3-TTS voices, 49.7 % on the author. Rung 1d gives it many more voices speaking our conversational lines, in DUET's own
acoustic conditions, and measures what that buys on the real voice.

**Pilot (2026-11-06, spark-2, no download):** Qwen3-TTS 1.7B (Apache-2.0) is in `/opt/voice/models`: VoiceDesign
(a voice from a description) and Base (a voice cloned from a reference clip, here from its speaker embedding only).
Both batch: 6-7x real time at batch 8, 15x at 32, 26x at 64 (15 GB). The teacher hears 40 pilot clips at 0-4 % WER
against their scripts (designed Czech-accented, teen, Scottish elder; two VoxPopuli speakers cloned). Samples in
`runs/ear-r1d-pilot/`.

**Data**
- **Designed voices, ~1,000:** descriptions built from gender x age x accent (about 40: US regions, UK, Irish,
  Scottish, Australian, Indian, Nigerian, Filipino, and English as a second language with Czech, Polish, German,
  Spanish, Italian, French, Russian, Ukrainian, Portuguese, Chinese, Japanese, Korean, Vietnamese, Arabic, Turkish...)
  x manner (fast, slow, mumbly, tired, excited, hesitant) x setting (phone, laptop microphone, car, kitchen). The 12
  test-call descriptions and close paraphrases are excluded.
- **Cloned voices, ~2,000 real speakers:** one 6-12 s clip per speaker from VoxPopuli EN (CC0; many speak English
  as a second language) and LibriSpeech train (CC BY 4.0). The voices are only used to train the ear, never published.
- **Lines:** the same 19,666 (the checks showed the words are not the problem), each spoken by two designed and two
  cloned voices: ~78,700 clips, ~150 h, ~6 h on spark-2 at batch 64. Clips the teacher hears at > 20 % WER against
  the script are dropped (the TTS skipped or added words). 100 designed and 200 cloned voices are held out for a
  "new TTS voices" test.
- **DUET's acoustics:** a `laptop` line in `channel.py` next to `phone`: a room, a small microphone's band and tilt,
  background noise, and an echo-canceller residual (another clip of DUET's Supertonic voice 15-30 dB under the
  speech, sometimes overlapping the first word). Every new clip gets one copy through either line.

**Training:** start from r1c (`ear_best.pt`; a new `--init` in `train.py`), mix the 3,263 h with the ~300 h of new
clips weighted so conversational speech is ~30 % of training (it was 9 %), lr 1e-3 with a fresh warm-up, 40k steps
(~8 h on spark-1). Same 29.8M ear. If the real voice moves, a from-scratch run on the full mix follows; if it does
not, the voice hypothesis is wrong and the recipe (vocabulary, size) is next.

**Tests, every rung from now:** real voice (the author's 96 turns; a teammate's calls too if he agrees), the held-out TTS
voices, the 310 call scripts in Supertonic voices, held-out audiobooks and VoxPopuli test. The 310 Qwen3-TTS calls
are still reported, but with Qwen3-TTS in training they test known engine, new voice, not a new engine.

**Gate:** real-voice WER <= 30 % greedy (r1c 49.7 %). Under that, voices were the lever; the 6 % phone gate stays
the end goal.

**Cost:** ~6 h spark-2 rendering, ~1 h labelling and ~8 h training on spark-1, no downloads.

## int4 test on r1c (2026-11-06, spark-1)
The ear is built on KARST's hybrid. The other half of KARST's best measured point, `hybrid_i4h` (int4 weights, no
training, +0.013 to +0.024 nats on the 125M language model), had not been tried on the ear. `ear/quant.py` rounds a
checkpoint the same way: W4A16 round-to-nearest, symmetric, one scale per 128 inputs, the same layers (delta qkv, gate
and o; attention qkv and o; both FFN projections; the head). It writes the rounded weights as floats, so `evaluate.py`
and `decode.py` score it unchanged. Decoding stays at r1c's alpha 0.5, beta 4, so only the weights differ. Scripts
`~/ear/int4_score.sh`, `int4_sweep.sh` and `int4_ship.sh` on spark-1; results in `~/ear/runs/r1c/int4/` (owner only:
it holds real-voice transcripts) and `runs/ear-r1c/int4/`.

| WER %, greedy / beam + mix LM | bf16 (59.5 MB) | int4, KARST's recipe (16.7 MB) | int4 body + int8 head (19.9 MB) |
|---|---|---|---|
| real voice (96 turns) | 49.6 / 46.2 | 59.0 / 51.2 | 50.3 / 47.8 |
| new Qwen3-TTS voices (r1d's held-out test, 1,131 clips) | 17.5 | 27.7 | 19.2 |
| phone calls | 27.4 / 21.7 | 37.9 / 25.6 | 29.2 / 22.9 |
| clean calls | 20.5 / 14.6 | 30.2 / 18.3 | 21.4 / 15.5 |
| audiobooks, clean | 13.0 / 10.6 | 26.3 / 17.1 | 14.9 / 11.9 |
| audiobooks, other | 15.8 / 12.8 | 29.5 / 19.8 | 17.7 / 14.4 |
| VoxPopuli | 22.4 / 20.7 | 33.5 / 28.2 | 24.7 / 22.9 |

**Which part takes int4 badly** (one part at a time, greedy, audiobooks clean / real voice; bf16 13.0 / 49.6):

| Rounded to int4 (groups of 128 unless noted) | Weights | Audiobooks | Real voice |
|---|---|---|---|
| CTC head only | 6.3M | 23.1 | 56.4 |
| FFN only | 14.2M | 14.2 | 49.2 |
| delta mixers only | 6.6M | 13.6 | 49.8 |
| attention only | 1.8M | 13.1 | 49.3 |
| everything but the head | 22.6M | 14.8 | 50.4 |
| everything, groups of 32 | 28.9M | 18.0 | 55.0 |
| everything as int8 instead (31.1 MB) | 28.9M | 13.1 | 49.3 |

Also rounding the two front-end convs saves 1.2 MB and costs +0.2 on audiobooks and +1.2 on the real voice (12 words),
so they stay bf16. Streaming is unchanged in every version (20 of 20 clips give the same words chunked and whole).

**Reading**
- KARST's int4 result holds for the ear's body. Its 22.6M delta, attention and FFN weights at int4 cost +1.8 points
  on audiobooks, +0.7 on the real voice and +1.8 on phone calls (greedy), with the LM about +1.2 to +1.6.
- It does not hold for the 16,384-way CTC head. Rounding the head alone takes the audiobook error from 13.0 to 23.1 %.
  Its weights round worst (relative error 0.26, against 0.12-0.13 everywhere else). In KARST the LM head added only
  +0.008 nats. Why the CTC head is so sensitive is not measured. Finer groups (32) do not rescue it; int8 does, at no
  loss.

**For rung 2:** ship the int4 body with an int8 head (19.9 MB, against TurboCTC's 950 MB), or int8 everywhere
(31.1 MB, no loss) if int4 is slow or unsupported on DirectML. That needs checking on the laptop. A short fine-tune with
the rounding in the loop could win back the body's 1-2 points (not tested). The same test runs on r1d when it lands.

## Rung 1d result: Qwen3-TTS voices in DUET's acoustics (2026-11-07, spark-1)
Data as planned: 78,664 Qwen3-TTS clips in designed and cloned voices speaking our lines (152.8 h; 2,841 dropped where
the teacher heard > 20 % WER against the script), each clean, through the phone line and through the new `laptop` line
with DUET's echo; 1,131 clips in held-out voices for the test (`qwentest`, `qwentestp`, `qwentestl`). Trained from r1c,
40k steps, lr 1e-3, Supertonic's and the new clips weighted x2: 333 min on spark-1. Best by `qwentestl` at step
38,000. Run in `runs/ear-r1d/` (the vault: it holds real-voice transcripts; owner-only on spark-1).

| WER % | r1c greedy | r1c beam + mix LM | r1d greedy | r1d beam + mix LM |
|---|---|---|---|---|
| **real voice, the author (96 turns)** | 49.7 | 46.4 | **46.2** | **43.1** |
| new Qwen3-TTS voices, clean | 17.5 | | 13.1 | |
| new Qwen3-TTS voices, laptop line | | | 17.3 | 14.2 |
| call scripts, Supertonic voices, clean / phone | 8.7 / 10.8 | | 8.6 / 10.8 | 7.7 (phone) |
| test calls, Qwen3-TTS voices, phone line | 27.4 | 21.7 | 15.2 | 12.3 |
| test calls, Qwen3-TTS voices, clean | 20.5 | 14.6 | 12.5 | 9.4 |
| audiobooks, clean | 13.0 | 10.6 | 12.8 | 10.6 |
| audiobooks, other | 15.8 | 12.8 | 15.5 | 12.5 |
| VoxPopuli test | 22.4 | 20.8 | 22.1 | 20.8 |

(Beam + mix LM at r1c's alpha 0.5, beta 4, which the audiobooks picked again. With the books LM the real voice is 42.9 %.)

- **The gate is missed: 46.2 % greedy on the real voice, against <= 30 %.** It moved 3.5 points, greedy and beam
  alike (about 28 of 809 words), less than r1b -> r1c's 5. And not steadily: per call from -9 to +5 points, and from
  one checkpoint to the next 45.9-52.4 % during the run.
- **What the voices bought, they bought on synthetic voices.** New Qwen3-TTS voices 17.5 -> 13.1 %; the Qwen3-TTS test
  calls 27.4 -> 15.2 % (that engine is now in training, so part of this is familiarity). Supertonic call scripts,
  audiobooks and VoxPopuli are unchanged (within 0.3).
- **The same errors as r1c, a few fewer:** 374 (r1c 413), 175 of them dropped words (192); first word wrong 54 % (61 %),
  middle 42 % (46 %), last 47 % (43 %). Short common words go missing: you x10, bro x8, no x7, I'm x7, thank x6.
- Streaming unchanged: 100 of 100 clips give the same words chunked and whole, 4.1 ms per 160 ms chunk (GPU, Python).
- **Incidents while the data was made (2026-11-06):** spark-2's render OOM at 11:39; on spark-1 the kernel's OOM killer
  took a teammate's ComfyUI (13:01) and tailscaled (13:47). Since then `~/ear/memguard.sh` stops our own (resumable) ear jobs
  when available memory falls under 12 GB; it never had to during training.

**Reading:** by the plan's own test (a from-scratch run if the real voice moves, the recipe if it does not), it barely
moved. Thousands of synthetic voices, a laptop line and DUET's echo taught the ear synthetic voices, not a human one on a
laptop microphone. The voice-count hypothesis is out; the synthetic-vs-human gap is the open one. Also open: whether
29.8M params with a 16,384-piece head is too small (the audiobooks sit at 12.8 % against the teacher's 1.1 %, unchanged
since r1c).

**int4 on r1d** (the ship candidate from r1c's sweep: int4 body, int8 head, 19.9 MB; `~/ear/int4_r1d.sh`, results in
`runs/ear-r1d/int4/`). It costs a little more than on r1c, most on the real voice:

| WER %, greedy / beam + mix LM | r1d bf16 | r1d int4 body + int8 head | r1c: the same change |
|---|---|---|---|
| real voice (96 turns) | 46.2 / 43.1 | 48.8 / 45.0 | +0.7 / +1.6 |
| new Qwen3-TTS voices, laptop line | 17.3 / 14.2 | 20.5 / 16.0 | |
| test calls, Qwen3-TTS, phone line | 15.2 / 12.3 | 17.3 / 13.4 | +1.8 / +1.2 |
| call scripts, Supertonic voices, phone | 10.8 / 7.7 | 12.5 / 9.1 | |
| audiobooks, clean | 12.8 / 10.6 | 15.1 / 11.8 | +1.9 / +1.3 |
| VoxPopuli test | 22.1 / 20.8 | 24.8 / 22.5 | +2.3 / +2.2 |

Streaming still matches (20 of 20 clips). So int4 costs 2-3 points greedy and 1-2 with the LM, on every set: rung 2
needs the short fine-tune with the rounding in the loop, or int8 (31.1 MB).

**Replay through DUET** (`tools/ear_replay.py`, call 20261103-230111, 22 caller turns, beam + mix LM; report in
`runs/ear-real/replay_20261103-230111_r1d.md`, vault only): the friend decides differently from the words as said on
11 of 22 turns with r1d's transcripts (r1c 14). TurboCTC 0 of 22 this time, 2 last time: that much is the model's own
run-to-run noise. "Joke" heard as "job" three times in a row means DUET never tells the joke it was asked for.

## Rung 1d, next (from the result)
- **Real human speech, labelled by the teacher** (distillation needs no transcripts). Candidates with commercial-use
  licences, each a download that needs an OK: Mozilla Common Voice English (CC0; real people on their own microphones,
  many accents, Czech among them; its terms must be accepted) and the AMI meetings (CC BY 4.0; real conversation,
  headset and room microphones). The real-voice test stays the gate.
- **the author's own calls as training data**, the 96 turns kept as the test: the most direct lever for DUET's one current
  user, and it needs only calls he makes anyway. The weights would then hold his voice: vault only, like the clips.
- Then the recipe A/B (a smaller vocabulary, a bigger ear) for the audiobook gap, and int4 with the rounding in the
  loop for rung 2.
