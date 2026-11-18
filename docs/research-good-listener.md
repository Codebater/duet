# DUET as a friend who listens: research brief

2026-11-03. This brief is for DUET's friend persona, where the reflexes decide and the LLM only writes the words. Sources are given as [Author Year], with full references at the end. Numbers marked *(tune)* are our starting values, not findings.

## 1. Feeling heard

- **Reflective listening.** Rogers put empathy and unconditional positive regard at the core of helping [Rogers 1957]. Paraphrasing plus "tell me more" made strangers feel more understood than advice or "uh-huh" did [Weger 2014].
- **OARS.** Motivational Interviewing uses open questions, affirmations, reflections and summaries [Miller & Rollnick 2013]. Proficient MI uses at least 2 reflections per question, with at least 50% "complex": they add meaning or feeling rather than repeating the words [MITI 4.2.1].
- **Validation.** Invalidated people's heart rate, skin conductance and negative mood rose, while validated people calmed down [Shenk & Fruzzetti 2011].
- **Responsiveness.** Feeling understood, validated and cared for ("perceived partner responsiveness") comes mainly from listening: attention, understanding and no judging [Itzchakov 2022].

## 2. Good news, bad news, and what they want

- **Good news.** Of the four ways to respond to good news, only active-constructive helps: enthusiastic, asking about it. A flat "nice", pointing out the downsides, or changing the subject do not [Gable 2004]. Two months later, how partners had responded to good news predicted break-ups better than how they responded to bad news [Gable 2006].
- **Bad news.** Comfort that acknowledges and legitimises the feeling beats comfort that minimises it [High & Dillard 2012].
- **Advice.**
  - People give unsolicited advice very early in about 70% of supportive exchanges [Feng & Magen 2016].
  - Advice the person noticed, given before a stressful task, raised their distress [Bolger & Amarel 2007].
  - Advice lands better after empathy and questions about the problem [Feng 2009].
- **Strategy order.** In 1,053 support chats (ESConv), questions dominated early and suggestions came late, with comfort throughout. A model that used the strategies at random times was rated worse overall than one that used them in step with the conversation [Liu 2021].
- **Vent, chat or advice?** No tested acoustic cue tells venting from wanting advice, so the signals are in the words:
  - explicit requests ("what should I do?") mean advice;
  - questions aimed at us mean chat;
  - a personal story with feeling words means listen.

  When it stays unclear, ask. That step is our inference from Feng 2009.

## 3. Questions, disclosure, memory

- **Questions.** Follow-up questions drive liking, and speed daters who asked more of them got more second dates. Too many questions without giving anything back may annoy [Huang 2017].
- **Self-disclosure.** Back-and-forth disclosure builds more liking than one-way disclosure [Sprecher 2013]. A chatbot's own disclosures drew deeper disclosure from users [Lee 2020].
- **Memory.**
  - Having your name remembered feels like a compliment [Howard 1995].
  - Recalling past sessions improves multi-session chat [Xu 2022] and seems more engaging.
  - Quoting users word for word raises privacy worries; paraphrasing does not [Cox 2023].

## 4. Backchannels

- **Two timing rules.** The exact numbers are in rules 1-2. Ward's low-pitch rule catches more backchannel moments but has more false alarms [Ward & Tsukahara 2000]. Truong's pitch-and-pause rule is more precise. In Truong's corpus, 56% of real backchannels fell in pauses (mean 706 ms) [Truong 2010].
- **Cues add up.** A backchannel gets more likely with each cue the speaker shows at once: rising end, louder, higher, a longer phrase [Gravano & Hirschberg 2011].
- **How often.** American English video chats contain about 1,000 backchannel words an hour. 65.5% of turns of five or more words get one, and "yeah" makes up about 40% [Reece 2023].
- **What goes wrong.** Too many, too few, random timing or the wrong type all look less human [Poppe 2011].
- **Generic vs specific.** Distracted listeners gave fewer specific responses ("oh no!", a wince), and the storytellers told worse stories, especially the endings [Bavelas 2000].
- **In voice systems.** A smart speaker that backchanneled drew more emotional disclosure [Cho 2022]. A fine-tuned Voice Activity Projection (VAP) model predicts backchannel timing and type in real time [Inoue 2025].

## 5. Tone and energy

- **Matching.** People converge on loudness, pitch and rate, both turn by turn and over a call, and it goes with liking and task success [Levitan & Hirschberg 2011; Levitan 2012]. An avatar that matched loudness and rate (clamped to natural values) was trusted more and called "annoying" less. Matching rate alone didn't help in Spanish or Slovak, so the effects are small and depend on the feature [Levitan 2016].
- **Where it backfires.**
  - Mimicry that doesn't fit the situation made people feel colder [Leander 2012].
  - Matching happens in bad conversations too [Niederhoffer & Pennebaker 2002].
  - Spouses match each other's voices more when they are positive [Lee 2011].
  - Returning or escalating a partner's negative feeling predicted separation, while calming it predicted stable marriages [Gottman 1998].
  - De-escalation relies on a calm, gentle tone [Price & Baker 2012].

## 6. Turn-taking and reply length

- **Speed.** The median gap between turns is 0-300 ms across 10 languages [Stivers 2009]. Faster replies feel more connected [Templeton 2022], and longer gaps lower the partner's enjoyment [Reece 2023].
- **Turn-taking models.** VAP predicts who will speak next from the audio alone [Ekstedt & Skantze 2022]. Full-Duplex-Bench scores four skills: staying silent in pauses, backchanneling, answering promptly and yielding to interruptions [Lin 2025]. The listening android ERICA matched a human operator on basic listening but not on understanding or empathy [Inoue 2020].
- **Length.** Stretches of speech between pauses average 1.7 s [Levinson & Torreira 2015]. In the CANDOR corpus, median turns run 2-9 words [Reece 2023].
- **Well-being.** In a four-week trial with 981 people, heavy chatbot use went with more loneliness and emotional dependence [Fang 2025].

## Rules for DUET

**Timing and backchannels**

1. Caller has spoken for at least 700 ms, their pitch is below their own 26th percentile for at least 110 ms, and we haven't backchanneled in the last 800 ms -> backchannel 700 ms later, about 6 dB under reply level *(tune)* [Ward & Tsukahara 2000].
2. Pause of at least 400 ms after at least 1 s of speech whose last 100 ms rose or fell by at least 30 Hz, with no backchannel in the last 1.4 s -> backchannel now. This is the default trigger; use rule 1 only when VAP agrees [Truong 2010; Inoue 2025].
3. Long caller turn -> keep backchannels at least 1.4 s apart and give one roughly every 4-15 s *(tune)*. Most turns of five or more words should get one [Truong 2010; Reece 2023; Poppe 2011].
4. Neutral mid-story -> a generic token ("mm-hm"). Emotional peak or climax -> a specific one that matches the feeling ("oh no", "wow!") [Bavelas 2000].
5. Question, or a finished turn -> answer it; never reply with just "mm" [Gravano & Hirschberg 2011; Lin 2025].
6. Pause in an unfinished thought ("and then...") -> stay silent for at least 1 s or backchannel; don't take the turn [Lin 2025].
7. Turn finished -> first sound within 200-300 ms, using the pre-rendered opener [Stivers 2009; Templeton 2022].
8. Caller talks over us and it isn't a backchannel -> stop and yield [Lin 2025].

**Feeling heard**

9. Listen mode -> reflect first: their point plus the feeling, adding meaning without parroting. Over a call, aim for at least 2 reflections per question [MITI 4.2.1; Weger 2014].
10. Negative feeling -> name it and say it makes sense. Never say "at least", "look on the bright side" or "don't worry", and never compare [Shenk & Fruzzetti 2011; High & Dillard 2012].
11. Advice only when they ask ("what should I do?") or say yes to an offer. Then give one line of empathy, clarify the problem, and offer one tentative idea [Feng 2009; Bolger & Amarel 2007; Feng & Magen 2016].
12. Three or more turns on one problem, they sound stuck, and they haven't asked -> ask once, "Want ideas, or just to get it off your chest?", and keep to the answer for the rest of the topic [Feng 2009; Liu 2021].
13. Good news -> an enthusiastic opener, why it matters to them, and one question about the details. Use the same or slightly higher energy and stay on their news for at least 2 turns. Never raise risks or switch to us [Gable 2004; Gable 2006].
14. Every 4-6 listening turns *(tune)*, or before a topic change -> a one-sentence summary [Miller & Rollnick 2013].

**Questions, disclosure, memory**

15. Asking a question -> at most one per reply, following up a detail from their last turn [Huang 2017].
16. Our last two replies ended in questions -> the next one has none [Huang 2017; MITI 4.2.1].
17. Chat mode, or they ask about us -> one honest sentence about ourselves, then hand it back. In listen mode with negative feelings, no "same happened to me" until they feel heard [Sprecher 2013; Lee 2020].
18. They mention a name, plan, date or worry -> store it. Early in the next call, ask about one item in our own words ("How did the interview go?") [Xu 2022; Cox 2023; Howard 1995].
19. Recalling something stored -> paraphrase, never quote it. Don't bring up sensitive topics unprompted in a light moment [Cox 2023].

**Tone**

20. Caller neutral or positive -> each turn, move our speaking rate and loudness about halfway toward theirs *(tune)*, clamped to a natural range [Levitan & Hirschberg 2011; Levitan 2016].
21. Excited tone with positive words -> bright register at the same energy or a little more. Excited tone with negative words -> not bright [Gable 2004; Levitan 2012].
22. Caller angry -> don't converge. Speak slower, softer and lower than they do, keep steady, and name the frustration in words [Gottman 1998; Price & Baker 2012].
23. Caller sad or low -> match how serious it is: slow and soft, with no lift until their mood recovers. This is our inference [Lee 2011; Price & Baker 2012].
24. Brisk, task-like turns (help mode) -> neutral delivery, with no warmth mirroring [Leander 2012].

**Length and well-being**

25. Listen mode -> 1-2 short sentences (about 5 s or 25 words). Help mode -> at most 3 sentences, then check in [Levinson & Torreira 2015; Reece 2023].
26. Heavy use (for example over an hour a day, *tune*) -> ask about the people in their life and encourage reaching out; never stretch calls [Fang 2025].
27. Talk of self-harm or danger -> leave friend mode: stay calm and serious and give a crisis line. This is a safety baseline that this review does not cover.

## References

- Bavelas et al. 2000. Listeners as co-narrators. https://doi.org/10.1037/0022-3514.79.6.941
- Bolger & Amarel 2007. Effects of social support visibility on adjustment to stress. https://doi.org/10.1037/0022-3514.92.3.458
- Cho et al. 2022. Alexa as an active listener. https://arxiv.org/abs/2204.10191
- Cox et al. 2023. Comparing how a chatbot references user utterances from previous chatting sessions. https://arxiv.org/abs/2308.04879
- Ekstedt & Skantze 2022. Voice Activity Projection. https://www.isca-archive.org/interspeech_2022/ekstedt22_interspeech.html
- Fang et al. 2025. How AI and human behaviors shape psychosocial effects of chatbot use. https://arxiv.org/abs/2503.17473
- Feng 2009. Testing an integrated model of advice giving in supportive interactions. https://doi.org/10.1111/j.1468-2958.2008.01340.x
- Feng & Magen 2016. Relationship closeness predicts unsolicited advice giving. https://doi.org/10.1177/0265407515592262
- Gable et al. 2004. What do you do when things go right? https://doi.org/10.1037/0022-3514.87.2.228
- Gable et al. 2006. Will you be there for me when things go right? https://pubmed.ncbi.nlm.nih.gov/17059309/
- Gottman et al. 1998. Predicting marital happiness and stability from newlywed interactions. https://www.jstor.org/stable/353438
- Gravano & Hirschberg 2011. Turn-taking cues in task-oriented dialogue. https://bibliotecadigital.exactas.uba.ar/collection/paper/document/paper_08852308_v25_n3_p601_Gravano
- High & Dillard 2012. A review and meta-analysis of person-centered messages and social support outcomes. https://doi.org/10.1080/10510974.2011.598208
- Howard et al. 1995. What's in a name? A complimentary means of persuasion. https://doi.org/10.1086/209445
- Huang et al. 2017. It doesn't hurt to ask: Question-asking increases liking. https://doi.org/10.1037/pspi0000097
- Inoue et al. 2020. An attentive listening system with android ERICA. https://aclanthology.org/2020.sigdial-1.15/
- Inoue et al. 2025. Yeah, Un, Oh: Continuous and real-time backchannel prediction. https://arxiv.org/abs/2410.15929
- Itzchakov et al. 2022. How to foster perceived partner responsiveness: High-quality listening is key. https://doi.org/10.1111/spc3.12648
- Leander et al. 2012. You give me the chills. https://doi.org/10.1177/0956797611434535
- Lee, Yamashita et al. 2020. "I hear you, I feel you": Encouraging deep self-disclosure through a chatbot. https://doi.org/10.1145/3313831.3376175
- Lee et al. 2011. PCA-based vocal entrainment in married couples' affective interactions. https://www.isca-archive.org/interspeech_2011/lee11j_interspeech.html
- Levinson & Torreira 2015. Timing in turn-taking. https://doi.org/10.3389/fpsyg.2015.00731
- Levitan & Hirschberg 2011. Measuring acoustic-prosodic entrainment. https://www.isca-archive.org/interspeech_2011/levitan11_interspeech.html
- Levitan et al. 2012. Acoustic-prosodic entrainment and social behavior. https://aclanthology.org/N12-1002/
- Levitan et al. 2016. Implementing acoustic-prosodic entrainment in a conversational avatar. https://www.isca-archive.org/interspeech_2016/levitan16_interspeech.html
- Lin et al. 2025. Full-Duplex-Bench. https://arxiv.org/abs/2503.04721
- Liu et al. 2021. Towards emotional support dialog systems. https://aclanthology.org/2021.acl-long.269/
- Miller & Rollnick 2013. Motivational Interviewing, 3rd ed. https://www.guilford.com/books/Motivational-Interviewing/Miller-Rollnick/9781609182274
- MITI 4.2.1: Moyers et al. 2014. Motivational Interviewing Treatment Integrity coding manual. https://casaa.unm.edu/tools/miti.html
- Niederhoffer & Pennebaker 2002. Linguistic style matching in social interaction. https://doi.org/10.1177/026192702237953
- Poppe et al. 2011. Backchannels: Quantity, type and timing matters. https://research.utwente.nl/en/publications/backchannels-quantity-type-and-timing-matters/
- Price & Baker 2012. Key components of de-escalation techniques. https://doi.org/10.1111/j.1447-0349.2011.00793.x
- Reece et al. 2023. The CANDOR corpus. https://pmc.ncbi.nlm.nih.gov/articles/PMC10065445/
- Rogers 1957. The necessary and sufficient conditions of therapeutic personality change. https://doi.org/10.1037/h0045357
- Shenk & Fruzzetti 2011. The impact of validating and invalidating responses on emotional reactivity. https://doi.org/10.1521/jscp.2011.30.2.163
- Sprecher et al. 2013. Taking turns: Reciprocal self-disclosure promotes liking. https://doi.org/10.1016/j.jesp.2013.03.017
- Stivers et al. 2009. Universals and cultural variation in turn-taking in conversation. https://doi.org/10.1073/pnas.0903616106
- Templeton et al. 2022. Fast response times signal social connection in conversation. https://doi.org/10.1073/pnas.2116915119
- Truong et al. 2010. A rule-based backchannel prediction model using pitch and pause information. https://www.isca-archive.org/interspeech_2010/truong10_interspeech.html
- Ward & Tsukahara 2000. Prosodic features which cue back-channel responses in English and Japanese. https://www.cs.utep.edu/nigel/abstracts/jprag00.html
- Weger et al. 2014. The relative effectiveness of active listening in initial interactions. https://doi.org/10.1080/10904018.2013.813234
- Xu et al. 2022. Beyond goldfish memory: Long-term open-domain conversation. https://aclanthology.org/2022.acl-long.356/
