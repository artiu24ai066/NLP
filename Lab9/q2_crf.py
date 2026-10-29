from pathlib import Path
import pickle
from sklearn.metrics import accuracy_score, classification_report

LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "Hindi-POS-Data"
OUTPUT_DIR = LAB_DIR / "output"
TRAIN_FILE = DATA_DIR / "hindi-train-pos.txt"
DEV_FILE = DATA_DIR / "hindi-dev-pos.txt"
TEST_FILE = DATA_DIR / "hindi-test-pos.txt"


def load_pos_data(filepath):
    """Read word<TAB>tag files; blank lines mark sentence boundaries."""
    sentences, sentence = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"Expected word<TAB>tag in {filepath}: {raw_line.rstrip()!r}"
                )
            word, tag = parts
            sentence.append((word, tag))
    if sentence:
        sentences.append(sentence)
    if not sentences:
        raise ValueError(f"No sentences loaded from {filepath}")
    return sentences


def sentence_labels(sentence):
    return [tag for _, tag in sentence]


def flatten_gold(sentences):
    return [tag for sentence in sentences for _, tag in sentence]


def flatten_predictions(predictions):
    return [tag for sentence in predictions for tag in sentence]


def evaluate(name, sentences, predictions):
    y_true = flatten_gold(sentences)
    y_pred = flatten_predictions(predictions)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"{name}: gold has {len(y_true)} tokens, predictions have {len(y_pred)}"
        )
    print(f"\n===== {name} =====")
    print(f"Token accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))


def save_predictions(filename, predictions):
    with open(filename, "wb") as f:
        pickle.dump(predictions, f)
    print(f"Saved predictions to {filename}")


# Install once if needed:
# pip install sklearn-crfsuite
import sklearn_crfsuite


def get_features(sentence, i):
    """The exact requested feature template for CRF."""
    word = sentence[i][0]
    previous_word = sentence[i - 1][0] if i > 0 else "BOS"
    next_word = sentence[i + 1][0] if i < len(sentence) - 1 else "EOS"

    features = {
        "previous_word": previous_word,
        "current_word": word,
        "next_word": next_word,
        "length_feature": "LESS" if len(word) <= 4 else "MORE",
    }

    for n in range(1, 5):
        features[f"prefix_{n}"] = word[:n] if len(word) >= n else "NULL"
        features[f"suffix_{n}"] = word[-n:] if len(word) >= n else "NULL"

    return features


def sentence_features(sentence):
    return [get_features(sentence, i) for i in range(len(sentence))]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train = load_pos_data(TRAIN_FILE)
    dev = load_pos_data(DEV_FILE)
    test = load_pos_data(TEST_FILE)

    X_train = [sentence_features(s) for s in train]
    y_train = [sentence_labels(s) for s in train]
    X_dev = [sentence_features(s) for s in dev]
    X_test = [sentence_features(s) for s in test]

    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)

    dev_pred = crf.predict(X_dev)
    test_pred = crf.predict(X_test)

    evaluate("CRF - DEV", dev, dev_pred)
    evaluate("CRF - TEST", test, test_pred)

    save_predictions(OUTPUT_DIR / "crf_dev_predictions.pkl", dev_pred)
    save_predictions(OUTPUT_DIR / "crf_test_predictions.pkl", test_pred)
    model_path = OUTPUT_DIR / "crf_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(crf, f)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
