"""Conversation lessons for the Spanish partners: the questions Luz or Sofía walk him through, one at a time.

Each step is her question (q, with its German), and an answer he can borrow (a, with its German) when he doesn't
know what to say. The answers come from his own cards where they fit ("Tuve mucho trabajo", "Invierto mucho tiempo
en ello", "Tengo dos pies izquierdos"). Questions are present tense or the few past-tense ones a day needs, short,
vos (Paraguay). After he answers, she reacts in one sentence and asks the next; when he asks what it means, she
explains in German and asks the same question again (amiga.py). See docs/amiga-teacher.md.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Step:
    q: str
    de: str
    a: str
    a_de: str


@dataclass(frozen=True)
class Lesson:
    title: str
    title_de: str
    steps: tuple


def L(title, title_de, *steps):
    return Lesson(title, title_de, tuple(Step(*s) for s in steps))


NOVIA = (
    L("Tu día", "Dein Tag",
      ("¿Cómo estás?", "Wie geht es dir?", "Estoy bien, gracias.", "Mir geht es gut, danke."),
      ("¿Qué hiciste hoy?", "Was hast du heute gemacht?", "Trabajé mucho.", "Ich habe viel gearbeitet."),
      ("¿Estás cansado?", "Bist du müde?", "Sí, un poco cansado.", "Ja, ein bisschen müde."),
      ("¿Qué comiste hoy?", "Was hast du heute gegessen?", "Comí chipa.", "Ich habe Chipa gegessen."),
      ("¿Qué hacés esta noche?", "Was machst du heute Abend?", "Veo un poco de tele.", "Ich schaue ein bisschen fern."),
      ("¿Me extrañaste?", "Hast du mich vermisst?", "Sí, te extrañé mucho.", "Ja, ich habe dich sehr vermisst.")),
    L("Lo que te gusta", "Was du magst",
      ("¿Te gusta bailar?", "Tanzt du gern?", "Sí, pero tengo dos pies izquierdos.",
       "Ja, aber ich habe zwei linke Füße."),
      ("¿Qué música te gusta?", "Welche Musik magst du?", "Me gusta un poco de todo.", "Ich mag ein bisschen von allem."),
      ("¿Te gusta el tereré?", "Magst du Tereré?", "Sí, me gusta mucho.", "Ja, sehr gern."),
      ("¿Qué comida te gusta?", "Welches Essen magst du?", "Me gusta la pizza.", "Ich mag Pizza."),
      ("¿Qué hacés en tu tiempo libre?", "Was machst du in deiner Freizeit?", "Hago deporte.", "Ich mache Sport.")),
    L("Planes", "Pläne",
      ("¿Qué hacemos el sábado?", "Was machen wir am Samstag?", "¿Vamos a la Costanera?", "Gehen wir an die Costanera?"),
      ("¿Tomamos un tereré?", "Trinken wir einen Tereré?", "Dale, me encanta.", "Okay, sehr gern."),
      ("¿A qué hora nos vemos?", "Um wie viel Uhr sehen wir uns?", "A las cinco.", "Um fünf."),
      ("¿Dónde nos encontramos?", "Wo treffen wir uns?", "En mi casa.", "Bei mir zu Hause."),
      ("¿Cenamos juntos después?", "Essen wir danach zusammen zu Abend?", "Sí, me encantaría.", "Ja, sehr gern.")),
    L("Tu trabajo", "Deine Arbeit",
      ("¿En qué trabajás?", "Was arbeitest du?", "Tengo un estudio creativo.", "Ich habe ein Kreativstudio."),
      ("¿Qué hacés en tu estudio?", "Was machst du in deinem Studio?", "Hago páginas web y videos.",
       "Ich mache Webseiten und Videos."),
      ("¿Te gusta tu trabajo?", "Magst du deine Arbeit?", "Sí, me gusta lo que hago.", "Ja, ich mag, was ich mache."),
      ("¿Trabajás mucho?", "Arbeitest du viel?", "Sí, invierto mucho tiempo en ello.", "Ja, ich stecke viel Zeit hinein."),
      ("¿Estás contento en Paraguay?", "Bist du glücklich in Paraguay?", "Sí, estoy muy contento acá.",
       "Ja, ich bin hier sehr glücklich.")),
)

CITA = (
    L("Conocerse", "Sich kennenlernen",
      ("¿Esta silla está libre?", "Ist dieser Stuhl frei?", "Sí, claro, está libre.", "Ja, klar, er ist frei."),
      ("¿Cómo te llamás?", "Wie heißt du?", "Me llamo {name}. ¿Y vos?", "Ich heiße {name}. Und du?"),
      ("¿De dónde sos?", "Woher kommst du?", "Soy de Europa.", "Ich komme aus Europa."),
      ("¿Vivís acá en Asunción?", "Wohnst du hier in Asunción?", "Sí, vivo acá.", "Ja, ich wohne hier."),
      ("¿En qué trabajás?", "Was arbeitest du?", "Tengo un estudio creativo.", "Ich habe ein Kreativstudio."),
      ("¿Te gusta Paraguay?", "Gefällt dir Paraguay?", "Sí, me gusta mucho.", "Ja, sehr."),
      ("¿Venís seguido a este café?", "Kommst du oft in dieses Café?", "No, es mi primera vez.", "Nein, zum ersten Mal.")),
    L("Gustos", "Vorlieben",
      ("¿Qué te gusta hacer?", "Was machst du gern?", "Me gusta un poco de todo.", "Ich mag ein bisschen von allem."),
      ("¿Te gusta bailar?", "Tanzt du gern?", "Sí, pero bailo muy mal.", "Ja, aber ich tanze sehr schlecht."),
      ("¿Hacés deporte?", "Machst du Sport?", "Antes hacía mucho deporte.", "Früher habe ich viel Sport gemacht."),
      ("¿Qué música escuchás?", "Welche Musik hörst du?", "Escucho de todo.", "Ich höre alles."),
      ("¿Cocinás?", "Kochst du?", "Sí, un poco.", "Ja, ein bisschen.")),
    L("Una cita", "Eine Verabredung",
      ("¿Tomamos algo?", "Trinken wir etwas?", "Sí, ¿un café?", "Ja, einen Kaffee?"),
      ("¿Qué tomás?", "Was trinkst du?", "Un café, por favor.", "Einen Kaffee, bitte."),
      ("¿Tenés planes para el sábado?", "Hast du Pläne für Samstag?", "No, ¿y vos?", "Nein, und du?"),
      ("¿Querés salir conmigo?", "Willst du mit mir ausgehen?", "¡Sí, me encantaría!", "Ja, sehr gern!"),
      ("¿Me das tu número?", "Gibst du mir deine Nummer?", "Claro, acá está.", "Klar, hier ist sie.")),
)

LESSONS = {"novia": NOVIA, "cita": CITA}
