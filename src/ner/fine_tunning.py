import ast
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer

# Labels


def generate_BI_labels(entity):
    return [f"B-{entity}", f"I-{entity}"]


entities = ["PERSO", "HEURE", "DATE", "LIEU", "TEL", "TACHE",
            "TECH", "EVENT", "FLECHE", "VOLTAGE", "CRITICALITY", "INFRA"]
label_list = ["O"]
for entity in entities:
    label_list.extend(generate_BI_labels(entity))

# Mapping labels --> entiers
label2id = {label: i for i, label in enumerate(label_list)}


# Charger le dataset


def parse_string_lists(example):
    """Convertit les chaînes de caractères en listes Python"""
    example["tokens"] = ast.literal_eval(example["tokens"])
    ner_tags_str = ast.literal_eval(example["ner_tags"])
    example["ner_tags"] = [label2id[tag] if isinstance(
        tag, str) else tag for tag in ner_tags_str]
    return example


dataset = load_dataset("csv", data_files={"data": "db.csv"})
dataset = dataset.map(parse_string_lists)

# Train/validation split
dataset = dataset["data"].train_test_split(
    test_size=0.2, shuffle=True, seed=42)
print(f"Training samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("camembert-base")


def align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # Ignorer les sous-tokens suivants
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = dataset.map(
    align_labels, batched=True, remove_columns=dataset["train"].column_names)

print("Tokenisation terminée !")

# Charger le modèle
model = AutoModelForTokenClassification.from_pretrained(
    "camembert-base",
    num_labels=len(label_list),
    id2label={i: label for i, label in enumerate(label_list)},
    label2id={label: i for i, label in enumerate(label_list)}
)

# Training arguments
training_args = TrainingArguments(
    output_dir="results",
    eval_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# Entraînement
trainer.train()

# Évaluation
results = trainer.evaluate()
print(results)

# Sauvegarde
model.save_pretrained("model_sauvegarde")
tokenizer.save_pretrained("model_sauvegarde")

print("Modèle sauvegardé")
