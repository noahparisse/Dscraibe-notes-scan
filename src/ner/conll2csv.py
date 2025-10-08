import csv


def conll_to_csv(input_conll, output_csv):
    with open(input_conll, "r", encoding="utf-8") as f:
        content = f.read().strip()

    sentences = content.split("\n\n")  # Séparer les phrases

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        writer.writerow(["tokens", "ner_tags"])

        for sentence in sentences:
            lines = sentence.split("\n")
            tokens = []
            ner_tags = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    tag = parts[-1]                  # dernier élément = tag
                    # le reste = token complet (numéro ou mot)
                    token = " ".join(parts[:-1])
                    tokens.append(token)
                    ner_tags.append(tag)

            writer.writerow([str(tokens), str(ner_tags)])


# Exemple d'utilisation
conll_to_csv("src/ner/db.conll", "src/ner/db.csv")
