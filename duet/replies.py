"""What DUET says, as Sona at GRAPHIQ's front desk. No prices here: they change, so Sona points to graphiq.art.
Numbers are written the way they should be spoken.

Answers are fixed lines per intent (Reflexx's zero-token rule: pick, don't generate). Mood changes the opener and
the delivery, never the facts.
"""

ANSWERS = {
    "book_consultation": "You can book a consultation at graphiq dot art slash contact. Pick the service, then a day "
                         "and time that suits you. Would you like the team to email you the link?",
    "reschedule_booking": "No problem. Just reply to your booking confirmation, or write to hello at graphiq dot art "
                          "with the new time, and the team will move it for you.",
    "consultation_fee": "The consultation fee is on graphiq dot art, and it's paid when you book. If you only have a "
                        "quick question, you don't need a paid consultation. I can pass it to the team.",
    "website_price": "Our Launch offer is a complete business website, live in ninety-six hours, with the design, "
                     "copy, domain, hosting, email and booking forms. The current price is on graphiq dot art.",
    "store_price": "A complete Shopify store is ready in about a week, with up to fifty products loaded and "
                   "payments and shipping set up. You'll find the price on graphiq dot art.",
    "software_quote": "Custom software starts with a consultation. A SaaS MVP usually takes six to ten weeks, and "
                      "the team gives you a fixed quote after the first call.",
    "timeline": "A business website goes live in ninety-six hours with our Launch offer, if the current slot is free. "
                "A store takes about a week, and custom software six to ten weeks.",
    "automation": "We set up AI assistants, WhatsApp and email automations, and connect your CRM and invoicing. It's "
                  "a one-time setup, and it usually saves ten hours a week or more.",
    "lead_generation": "For more leads, we build a landing page and funnel with email and text follow-ups, CRM setup "
                       "and conversion tracking, as a one-time project.",
    "seo": "Yes. Our Get Found on Google plan covers keywords, on-page and technical fixes, content, local "
           "visibility, and a monthly ranking report.",
    "care_plan": "Our monthly care plan covers hosting, backups, security, updates, uptime monitoring, and support "
                 "around the clock. It's completely optional.",
    "ownership": "You own everything. We deploy to your own accounts, so the code, hosting, domain and database are "
                 "yours from day one.",
    "ai_and_templates": "We use AI openly to move fast, but every site is custom designed by a person. No themes, and "
                        "no page builders.",
    "portfolio": "You can see our work on graphiq dot art. I can also ask the team to send you examples from your "
                 "industry.",
    "location": "We're a US company based in Fort Lauderdale, Florida, and we work with clients across the US and "
                "Europe.",
    "contact_details": "You can email hello at graphiq dot art, or call or WhatsApp us at two one three, two three "
                       "two, two two two seven.",
    "speak_to_person": "Of course. The fastest way to reach the team is hello at graphiq dot art, or you can book a "
                       "call at graphiq dot art slash contact.",
    "site_problem": "Let's get that fixed. Please email hello at graphiq dot art with what you're seeing, and the "
                    "team will jump on it. Care plan clients have support around the clock.",
    "billing": "For billing, please send your invoice number to hello at graphiq dot art, and the team will sort it "
               "out with you.",
    "refund_cancellation": "I can't decide that myself, but I'll make sure the team sees it. Please email hello at "
                           "graphiq dot art with your details, and they'll get back to you.",
    "partnership": "We'd be happy to hear about it. Please send the details to hello at graphiq dot art.",
    "press": "Thanks for reaching out. Please send your request to hello at graphiq dot art, and we'll get back to "
             "you.",
    "job_inquiry": "Thanks for your interest! Please send your portfolio to hello at graphiq dot art.",
    "vendor_pitch": "Thanks. Please send the details to hello at graphiq dot art, and the team will take a look.",
    "sister_brands": "Yes, Clausign and Fregio are part of the same company. Clausign does e-signatures, and Fregio "
                     "is our AI website builder.",
    "out_of_scope": "Hmm, I don't think that's something we can help with. We're GRAPHIQ, a web design and software "
                    "studio. Is there anything about a website, a store, or automation I can help with?",
}

# Openers by register (mood.respond picks the register). The first sentence of a reply is always one of these,
# rendered once at startup, so a reply starts sounding at once.
GREET_BACK = {"bright": "Hi there!", "warm": "Hi!", "calm": "Hello.", "steady": "Hello."}
GREET_AGAIN = {"bright": "Hi again!", "warm": "Hi again.", "calm": "Hello again.", "steady": "Hello again."}
ACK = {"bright": "Sure thing!", "warm": "Sure!", "calm": "Of course.", "steady": "I understand."}
GLAD = {"bright": "That's great to hear!", "warm": "That's great to hear.", "calm": "Good to hear.",
        "steady": "Good to hear."}
NOTED = {"bright": "Got it!", "warm": "Got it.", "calm": "I see.", "steady": "I see."}
LOOKUP = {"bright": "Let me see!", "warm": "Let me see.", "calm": "Let me see.", "steady": "Let me see."}
EMPATHY = {"bright": "Oh no, sorry about that.", "warm": "I'm sorry about that.", "calm": "I'm sorry to hear that.",
           "steady": "I'm sorry about that."}
HOW_ARE_YOU = {"bright": "I'm doing great, thanks for asking!", "warm": "I'm doing well, thank you!",
               "calm": "I'm well, thank you.", "steady": "I'm well, thank you."}
INVITE = {"bright": "What can I do for you today?", "warm": "How can I help you today?",
          "calm": "How can I help?", "steady": "How can I help?"}
WELCOME = {"bright": "You're very welcome!", "warm": "You're welcome!", "calm": "You're welcome.",
           "steady": "Of course."}
GO_AHEAD = {"bright": "Sure, go ahead!", "warm": "Sure, go ahead.", "calm": "Of course, go ahead.",
            "steady": "Of course, go ahead."}

OPENING = "Hi, this is Sona at GRAPHIQ."
OPENING_INVITE = "How can I help you today?"
ANYTHING_ELSE = "Is there anything else I can help with?"
GOODBYE = "Thanks for calling GRAPHIQ. Have a great day!"
NOT_SURE = "Sorry, I want to make sure I get this right. Could you tell me a little more?"
BOT = "I'm Sona, GRAPHIQ's AI assistant. I can answer questions and pass anything on to the team."
# what Sona can do, for "talk to me", "what can you do" and questions she can't answer (two ways to say it, so a
# repeated question never gets the same words twice)
CAN_DO = ["I can tell you about our websites, online stores and custom software, what they cost and how long they "
          "take, or set up a call with the team.",
          "Ask me about websites, stores, software, prices or timelines, or I can book you a call with the team."]
CANT_ANSWER = ["Hmm, I can't help with that one.", "Sorry, that one's outside what I know."]
HEAR_YOU = "Yes, I can hear you."
NAME_ASK = ["Not yet! What's your name?", "I don't know it yet. What's your name?"]
NAME_KNOWN = "You're {name}."
NICE_TO_MEET = "Nice to meet you, {name}!"
LISTENING = ["I'm listening.", "Take your time."]
AFFIRMED = "Great! I'll pass that on, and the team will email you from hello at graphiq dot art."
DENIED = "No problem. What can I help you with instead?"
RESUME = "Sure."
CLARIFY = "Just to check, is this about {topic}?"

TOPICS = {
    "book_consultation": "booking a consultation", "reschedule_booking": "moving a booking",
    "consultation_fee": "the consultation fee", "website_price": "the price of a website",
    "store_price": "an online store", "software_quote": "custom software", "timeline": "how long it takes",
    "automation": "automating your business", "lead_generation": "getting more leads", "seo": "Google rankings",
    "care_plan": "hosting and maintenance", "ownership": "who owns the site", "ai_and_templates": "how we use AI",
    "portfolio": "our past work", "location": "where we're based", "contact_details": "how to reach us",
    "speak_to_person": "talking to someone on the team", "site_problem": "a problem with your site",
    "billing": "a bill or invoice", "refund_cancellation": "a refund or cancellation",
    "partnership": "a partnership", "press": "a press request", "job_inquiry": "working with us",
    "vendor_pitch": "a business offer", "sister_brands": "Clausign or Fregio",
}

# How the voice should say a few written words.
SPOKEN = {"GRAPHIQ": "Graphic", "graphiq": "graphic", "Clausign": "Clause-sign", "SaaS": "sass", "MVP": "M V P",
          "CRM": "C R M", "AI": "A I"}

AFTER_CUT = {"bright": "Sure!", "warm": "Sure.", "calm": "Of course.", "steady": "Of course."}
INVITE_AGAIN = "What can I do for you?"
# invitations in each register, said in turn so the caller never hears the same one twice in a row
INVITES = {r: [INVITE[r], INVITE_AGAIN, "What would you like to know?"] for r in INVITE}
GREAT = "Great!"
NO_PROBLEM = "No problem."

# Every sentence a reply can start with, with the register it is said in, so all of them are rendered (and cached)
# before the call: a reply's first sound never waits for the voice model.
REGISTERS = ("bright", "warm", "calm", "steady")
OPENERS = [(d[r], r) for d in (GREET_BACK, GREET_AGAIN, ACK, GLAD, NOTED, LOOKUP, EMPATHY, HOW_ARE_YOU, WELCOME,
                               GO_AHEAD, AFTER_CUT) for r in d] + [
    (t, r) for t in ("Hi, this is Sona at GRAPHIQ.", RESUME, "Hmm.", "Thanks for calling GRAPHIQ.", GREAT, NO_PROBLEM,
                     HEAR_YOU, *CANT_ANSWER, *NAME_ASK, *LISTENING) for r in REGISTERS]

# ------------------------------------------------------------------ facts read off the clock, spoken as words
ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen").split()
TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty"}
ORD = ("first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth "
       "fifteenth sixteenth seventeenth eighteenth nineteenth twentieth").split()


def number_words(n):
    return ONES[n] if n < 20 else TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")


def ordinal_words(n):
    if n <= 20:
        return ORD[n - 1]
    if n % 10 == 0:
        return "thirtieth"
    return f"{TENS[n // 10]}-{ORD[n % 10 - 1]}"


def spoken_date(dt):
    return f"It's {dt.strftime('%A')}, {dt.strftime('%B')} {ordinal_words(dt.day)}."


def spoken_time(dt):
    h, m = dt.hour, dt.minute
    if (h, m) == (12, 0):
        return "It's noon here in Florida."
    if (h, m) == (0, 0):
        return "It's midnight here in Florida."
    part = "in the morning" if h < 12 else "in the afternoon" if h < 18 else "in the evening" if h < 22 else "at night"
    hw = number_words(h % 12 or 12)
    mw = "o'clock" if m == 0 else f"oh {ONES[m]}" if m < 10 else number_words(m)
    return f"It's {hw} {mw} {part} here in Florida."
