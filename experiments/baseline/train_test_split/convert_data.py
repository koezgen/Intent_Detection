import json
import spacy

# Load a SpaCy English model (adjust if needed)
# If you haven't installed it, do: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def load_rasa_data(json_path):
    """
    Loads the Rasa-like JSON data, returns a list of examples:
    [
      {
        "text": "...",
        "intent": "...",
        "entities": [
          {"start": ..., "end": ..., "value": "...", "entity": "..."},
          ...
        ]
      },
      ...
    ]
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["rasa_nlu_data"]["common_examples"]

def align_tokens_with_entities(tokens, token_offsets, entities):
    """
    Given:
      - tokens: list of token strings
      - token_offsets: list of (start_char, end_char) for each token in the original text
      - entities: list of { "start": ..., "end": ..., "entity": ... }

    Return a list of NER labels (same length as tokens).
    If a token’s character range overlaps with an entity’s character range,
    we assign that entity label; otherwise "O".
    """
    ner_tags = ["O"] * len(tokens)  # default to "O" (no entity)

    for ent in entities:
        ent_start = ent["start"]
        ent_end   = ent["end"]
        ent_label = ent["entity"]
        
        # Check overlap with each token
        for i, (t_start, t_end) in enumerate(token_offsets):
            # Overlap check: if the token range [t_start, t_end)
            # intersects with entity range [ent_start, ent_end)
            if not (t_end <= ent_start or t_start >= ent_end):
                ner_tags[i] = ent_label
    
    return ner_tags

def build_joint_dataset(json_path):
    """
    Build a single dataset with both intent and NER tags in each example.
    Returns a list of dicts with:
      {
        "tokens": [...],
        "ner_tags": [...],
        "intent": "..."
      }
    """
    examples = load_rasa_data(json_path)
    dataset = []

    for ex in examples:
        text = ex["text"]
        intent = ex["intent"]
        entity_list = ex.get("entities", [])

        # 1) Tokenize using SpaCy
        doc = nlp(text)
        tokens = [token.text for token in doc]

        # 2) Collect token offsets (start, end)
        token_offsets = [(token.idx, token.idx + len(token.text)) for token in doc]

        # 3) Align tokens with entity labels
        ner_tags = align_tokens_with_entities(tokens, token_offsets, entity_list)

        # 4) Store in a single dict
        dataset.append({
            "tokens": tokens,
            "ner_tags": ner_tags,
            "intent": intent
        })

    return dataset

def main():
    train_file = "train.json"  # Path to your Rasa-like train.json
    combined_dataset = build_joint_dataset(train_file)

    # Save the combined data as a single JSON
    with open("train_combined.json", "w", encoding="utf-8") as f:
        json.dump(combined_dataset, f, indent=2, ensure_ascii=False)

    print("Done! Combined data saved as train_combined.json")
    print("Example entry:")
    print(json.dumps(combined_dataset[0], indent=2))

if __name__ == "__main__":
    main()
