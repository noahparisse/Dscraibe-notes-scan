import datetime
import random
import re

LABELS = ["datetime", "duration", "geographical_region", "border", "third-party_actor", "power", "voltage_level", "voltage", "infrastructure", "substation", "network_event", "center", "operation", "external_event", "criticality",
          "event_impact", "moment", "event", "country", "value", "actor", "app", "TOPOLOGY", "STUDY", "LINE", "OPERATING CONDITIONS", "TRANSIT", "DECISION", "LIMIT", "OPERATING MODE", "DATA", "IT ISSUE", "TOOL ACTION", "TO DO", "CALL"]

ARROWS = ["->", "=>", "→", "-->", "⇨", "—>"]

# --- Date & Hours ---


# --- Geographical region & Country---


geographical_region = ["Rouen",
                       "Caen",
                       "Le Havre",
                       "Cherbourg",
                       "Deauville",
                       "Rennes",
                       "Saint-Malo",
                       "Brest",
                       "Saint-Brieuc",
                       "Nantes",
                       "Lille",
                       "Dunkerque",
                       "Calais",
                       "Amiens",
                       "Valenciennes",
                       "Paris",
                       ]

country = ["France", "Allemagne", "Italie", "Belgique",
           "Suisse", "Espagne", "Luxembourg", "Royaume-Uni", "UK", "Pays-Bas"]

# --- Third party actor ---
third_party_actor = ["SNCF", "Enedis", "Renault", "EDF"]


# --- Power & Voltage ---


# --- Infrastructure ---
infrastructure = ["CSS", "Central sous station (SNCF)", "GEH",
                  "Groupe exploitation hydraulique", "TR", "Transformateur", "BR", "Bâtiment de relayage", "GT", "Groupe de traction"]


# --- Center ---
center = ["COSE", "Centre opérationnel du système électrique",
          "COSE-P", "Centre opérationnel du système électrique de Paris",
          "COSE Paris", "Centre opérationnel du système électrique de Paris"]

# --- Operation ---
operation = ["MNV", "Manoeuvre", "RACR", "Retour à la conduite des réseaux",
             "RDCR", "Retrait de la conduite des réseaux",  "CTS", "Cycle triphasé supplémentaire"]

# --- External Event ---
external_event = ["TST", "Travaux sous tensions"]

# --- Criticality ---
criticality = ["URGENT", "Attention", "Prioritaire",
               "Blocant", "Immédiat", "Important"]


# --- Event Impact ---


# --- Moment ---
moment = ["HO", "Heures ouvrées"]

# --- Event ---
event = ["DIFFB", "Protection différentielle de barre",
         "RADA", "Retrait agile suite à détection d'anomalie", "Panne réseau", "Incident"]


# --- Actor ---
actor = ["PO", "Poste opérateur", "PDM", "Personnel de manoeuvre", "CCO",
         "Chargé de conduite", "CEX", "Chargé d'exploitation", "ACR", "Agence de conduite régionale"]

# --- App ---

# --- Topology ---

# --- Study ---
STUDY = ["Etude"]

# --- Line ---

# --- Operating conditions ---

# --- Transit ---

# --- Decision ---

# --- Limit ---

# --- Operating mode ---
operating_mode = ["SUAV", "Sous tension à vide", "TIR", "Thermographie infrarouge",
                  "RSD", "Régime surveillance dispatching", "SMACC", "Système avancé pour la commande de la compensation", "RA", "Réenclencheur automatique", "FDE", "Filet d'essai", "PF", "Point figé"]

# --- Data ---

# --- IT issue ---

# --- Tool action ---

# --- To do ---


# --- Call ---


# POSSIBLE NOTES
NOTES = [
    # --- Rdv & Appels classiques ---
    "Rdv avec {PERSO} à {HEURE}",
    "Rdv avec {PERSO} {DATE} à {HEURE} {CRITICALITY}",
    "Appeler {PERSO} à {HEURE} {CRITICALITY}",
    "Appeler {PERSO} {DATE}",
    "Prévoir {TACHE} à {HEURE}",
    "Prévoir {TACHE} {DATE} {CRITICALITY}",
    "Rappel {PERSO} {HEURE} {CRITICALITY}",
    "Validation avec {PERSO} sur {TACHE}",
    "Validation avec {PERSO} sur {TACHE} {DATE} {CRITICALITY}",
    "Préparer {TACHE} pour {PERSO}",
    "Réaliser {TACHE} car {EVENT} {CRITICALITY}",

    # --- Tâches courtes, style "prise de note rapide" ---
    "{TACHE} {HEURE}",
    "{TACHE} {DATE}",
    "rappel {PERSO} {HEURE} {CRITICALITY}",
    "{PERSO} {TACHE}",
    "à faire : {TACHE} {CRITICALITY}",
    "{TACHE} {LIEU} à prévoir",
    "{TACHE} {LIEU} terminé",
    "{TACHE} {LIEU} en cours",
    "vérif {TACHE} {CRITICALITY}",

    # --- Variantes avec lieu / infrastructure ---
    "{TACHE} {FLECHE} {LIEU} {CRITICALITY}",
    "{TACHE} {FLECHE} {INFRA} {CRITICALITY}",
    "Rdv avec {PERSO} {FLECHE} {LIEU}",
    "Rdv avec {PERSO} {FLECHE} {INFRA}",
    "Intervention {PERSO} {TACHE} {LIEU} {CRITICALITY}",
    "Intervention {PERSO} {TACHE} {INFRA} {CRITICALITY}",

    # --- Téléphone / numéros ponctuels ---
    "Appeler {PERSO} {TEL} {CRITICALITY}",
    "{PERSO} {TEL}",
    "Changement numéro {PERSO} {FLECHE} {TEL}",
    "Nouveau numéro {PERSO} : {TEL}",
    "{PERSO} nouveau tel {TEL}",

    # --- Vocabulaire technique / signalement ---
    "Rdv pour {TECH} à {HEURE}",
    "Validation {TECH} {DATE}",
    "{TECH} {LIEU} prévu {DATE} {HEURE} {CRITICALITY}",
    "{TECH} {LIEU} ok",
    "Problème {TECH} signalé {CRITICALITY} à {INFRA}",
    "Signalement {EVENT} {DATE} {CRITICALITY} sur {INFRA}",
    "Signalement {EVENT} {DATE} {HEURE} {CRITICALITY} à {INFRA}",
    "Appeler {PERSO} pour {EVENT} {CRITICALITY}",

    # --- Événements & incidents ---
    "{EVENT} {HEURE}",
    "{EVENT} {DATE} {CRITICALITY}",
    "{EVENT} {DATE} {HEURE} {CRITICALITY}",
    "Rdv sur {EVENT} {DATE}",
    "Validation {EVENT} {DATE}",
    "Incident {EVENT} à suivre {CRITICALITY} sur {INFRA}",
    "Maj {EVENT}",
    "Annulation {EVENT}",
    "Surveiller {EVENT} {LIEU} {CRITICALITY}",
    "Surveiller {EVENT} {INFRA} {CRITICALITY}",

    # --- Formulations très naturelles / télégraphiques ---
    "ok {TACHE}",
    "vu {PERSO}",
    "app {PERSO} {CRITICALITY}",
    "pb {TECH} {CRITICALITY} {INFRA}",
    "RAS {LIEU}",
    "RAS {INFRA}",
    "rappel {HEURE} {CRITICALITY}",
    "{TACHE} {CRITICALITY}",

    # --- Personnalisé ---
    "Retrait plannifié de la liaison {LIEU} - {LIEU_BIS}",
    "Etude du N-1 de la liaison {LIEU} - {LIEU_BIS}",
    "Coupure {VOLTAGE} {LIEU}",
    "Coupure {VOLTAGE} {INFRA}",
    "{TACHE} {LIEU} {VOLTAGE}",
    "{TACHE} {INFRA} {VOLTAGE}",
    "{EVENT} {TECH} {HEURE}"

]

# --- Simple generators ---


def generer_tel():
    premier = random.choice(["06", "07"])
    reste = " ".join([f"{random.randint(0,99):02d}" for _ in range(4)])
    return f"{premier} {reste}"


def gen_hours():
    hour = random.randint(7, 19)
    minute = random.choice([0, 15, 30, 45])
    formats = [f"{hour:02d}h{minute:02d}",
               f"{hour}h", f"{hour:02d}:{minute:02d}"]
    return random.choice(formats)


def gen_date():
    today = datetime.date.today()
    delta_days = random.randint(-2, 15)
    date = today + datetime.timedelta(days=delta_days)
    jours_semaine = ["Lundi", "Mardi", "Mercredi",
                     "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    jour_nom = jours_semaine[date.weekday()]
    formats = [
        date.strftime("%d/%m"),
        date.strftime("%d/%m/%Y"),
        f"{jour_nom} {date.strftime('%d/%m')}",
        jour_nom,
    ]
    relatives = ["hier", "aujourd'hui", "demain", "lundi prochain",
                 "mardi prochain", "fin de semaine", "cette nuit", "cet aprem", "ce matin", "ce midi"]
    if random.random() < 0.5:
        return random.choice(relatives)
    return random.choice(formats)


def gen_power():
    power = random.randint(0, 1000)
    formats = [f"{power}W", f"{power}kW", f"{power}MW", f"{power}GW"]
    return random.choice(formats)


def gen_voltage():
    voltage = random.randint(10, 1000)
    formats = [f"{voltage}V", f"{voltage}kV", f"{voltage}MV", f"{voltage}GV"]
    return random.choice(formats)


def gen_perso():
    if random.random() < 0.7:
        return random.choice(actor)
    else:
        return random.choice(third_party_actor)


def gen_location():
    r = random.random()
    if r < 0.9:
        return random.choice(geographical_region)
    elif r < 0.95:
        return random.choice(center)
    else:
        return random.choice(country)


def gen_infra():
    return random.choice(infrastructure)


def gen_operation():
    return random.choice(operation)


def gen_tech():
    return random.choice(operating_mode)


def gen_event():
    r = random.random()
    if r < 0.9:
        return random.choice(event)
    else:
        return random.choice(external_event)


def gen_criticality():
    r = random.random()
    if r < 0.9:
        return ""
    else:
        return random.choice(criticality)

# --- Fonction principale ---


def tokenize(text: str):
    pattern = r"""
        \b\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?\b                                  # dates
        | \b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\b                             # heures 17:30, 17:30:45
        | \b(?:[01]?\d|2[0-3])[hH](?:[0-5]\d)?\b                                 # heures 17h30
        | \b(?:[01]?\d|2[0-3])\s?[hH]\s?(?:[0-5]\d)?\b                           # heures 17 h 30
        | (?:->|=>|→|-->|⇨|—>)                                                   # flèches
        | \b(?:\+33|0)[1-9](?:[ .-]?\d{2}){4}\b                                 # numéros de téléphone FR
        | [\w]+(?:[-'][\w]+)*                                                  # mots (Jean-Pierre, aujourd'hui, l'heure)
        | [-:.]                                                                 # tirets et :
        | [^\w\s]                                                              # ponctuation restante
    """
    token_regex = re.compile(pattern, re.UNICODE | re.VERBOSE)
    return token_regex.findall(text)


def generate_annotated_sentence():
    """
    Génère une phrase aléatoire à partir d'un template et d'entités, puis retourne les tokens et les annotations BIO pour NER.
    """
    template = random.choice(NOTES)
    entity_map = {
        "PERSO": gen_perso(),
        "HEURE": gen_hours(),
        "DATE": gen_date(),
        "LIEU": gen_location(),
        "LIEU_BIS": gen_location(),
        "TEL": generer_tel(),
        "TACHE": gen_operation(),
        "TECH": gen_tech(),
        "EVENT": gen_event(),
        "FLECHE": random.choice(ARROWS),
        "VOLTAGE": gen_voltage(),
        "CRITICALITY": gen_criticality(),
        "INFRA": gen_infra()
    }
    phrase = template.format(**entity_map)

    # Tokenisation
    tokens = tokenize(phrase)

    # Annotation BIO
    annotations = []
    print(phrase)
    for i, token in enumerate(tokens):
        label = "O"
        prev_token = tokens[i-1] if i > 0 else None
        for ent_key, ent_value in entity_map.items():
            # ex: "présentation client" → ["présentation", "client"]
            ent_tokens = tokenize(ent_value)
            if token in ent_tokens:
                idx = ent_tokens.index(token)
                if idx == 0:
                    if ent_key == "LIEU_BIS":
                        label = "B-LIEU"
                    else:
                        label = f"B-{ent_key}"
                elif prev_token == ent_tokens[idx-1]:
                    if ent_key == "LIEU_BIS":
                        label = "I-LIEU"
                    else:
                        label = f"I-{ent_key}"
        annotations.append((token, label))

    return annotations


# --- Écriture dans un fichier texte ---
path = "src/ner/db.conll"
num_examples = 10000

with open(path, "w", encoding="utf-8") as f:
    for _ in range(num_examples):
        template = random.choice(NOTES)
        annotated = generate_annotated_sentence()
        for token, label in annotated:
            f.write(f"{token}\t{label}\n")
        f.write("\n")

print(f"Dataset NER synthétique généré : {path}")
