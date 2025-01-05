import os
import pickle
import json
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = "../../data/raw_data/ms-cntk-atis"
OUTPUT_DIR = "train_test_split"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_ds(file_path):
    """
    Load the ATIS dataset from a .pkl file.

    Args:
        file_path (str): Path to the .pkl file.

    Returns:
        tuple: Contains queries, slot labels, and intent labels.
    """
    with open(file_path, 'rb') as stream:
        ds, dicts = pickle.load(stream)

    print(f"Loaded dataset from: {file_path}")
    print(f"Samples: {len(ds['query'])}")
    print(f"Vocab Size: {len(dicts['token_ids'])}")
    print(f"Slot Count: {len(dicts['slot_ids'])}")
    print(f"Intent Count: {len(dicts['intent_ids'])}")

    queries = [x for x in ds['query']]
    intents = [x for x in ds['intent_labels']]
    return queries, intents, dicts


def convert_to_text_and_intent(queries, intents, dicts):
    """
    Convert tokenized queries and intent labels to human-readable format.

    Args:
        queries (list): List of tokenized queries.
        intents (list): List of intent labels.
        dicts (dict): Contains mappings for tokens and intents.

    Returns:
        list: List of dictionaries with 'text' and 'intent' keys.
    """
    i2t = {v: k for k, v in dicts['token_ids'].items()}
    i2in = {v: k for k, v in dicts['intent_ids'].items()}

    dataset = []
    for query, intent in zip(queries, intents):
        text = ' '.join([i2t[token] for token in query if token != 0])  # Ignore padding tokens
        intent_label = i2in[intent[0]]  # Intent label is a list, take the first element
        dataset.append({"text": text, "intent": intent_label})

    return dataset


def save_as_json(data, file_path):
    """
    Save the processed dataset as a JSON file.

    Args:
        data (list): Dataset to be saved.
        file_path (str): Path to save the JSON file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved processed data to: {file_path}")


def main():
    # Load train and test datasets
    train_queries, train_intents, dicts = load_ds(os.path.join(DATA_DIR, 'atis.train.pkl'))
    test_queries, test_intents, _ = load_ds(os.path.join(DATA_DIR, 'atis.test.pkl'))

    # Convert train and test datasets to text and intent format
    train_data = convert_to_text_and_intent(train_queries, train_intents, dicts)
    test_data = convert_to_text_and_intent(test_queries, test_intents, dicts)

    # Save processed datasets
    save_as_json(train_data, os.path.join(OUTPUT_DIR, 'train.json'))
    save_as_json(test_data, os.path.join(OUTPUT_DIR, 'test.json'))


if __name__ == "__main__":
    main()
