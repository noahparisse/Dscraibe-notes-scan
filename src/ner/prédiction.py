from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

MODEL_PATH = "src/ner/model_sauvegarde"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)

# Création de la pipeline NER
nlp_ner = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"  # regroupe les tokens B/I en une entité unique
)


# Phrase à prédire
sentence = ""

# Prédiction
entities = nlp_ner(sentence)

# Merge les tokens de la même entité à la suite


def merge_tokens(tokens):
    if not tokens:
        return []

    merged = [tokens[0]]

    for tok in tokens[1:]:
        last = merged[-1]

        same_entity = tok['entity_group'] == last['entity_group']
        contiguous = tok['start'] == last['end']
        separated_by_space = tok['start'] == last['end'] + 1

        if same_entity and (contiguous or separated_by_space):
            if separated_by_space:
                last['word'] += ' '
            last['word'] += tok['word']
            last['end'] = tok['end']
            last['score'] = max(last['score'], tok['score'])
        else:
            merged.append(tok)

    return merged


# Affichage des résultats
for ent in merge_tokens(entities):
    print(f"{ent['word']}  →  {ent['entity_group']}  (score={ent['score']:.2f})")
