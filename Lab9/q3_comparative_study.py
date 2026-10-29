"""
Question 3: Comparative study of MEMM, CRF, Naive Bayes and HMM.

Run q1_memm.py and q2_crf.py first. Then create
assignment7_predictions.pkl containing a dictionary with these four keys:
    "naive_bayes_dev", "naive_bayes_test", "hmm_dev", "hmm_test"

Each value must be a list of sentences, where each sentence is a list of
predicted POS tags in the original token order. Generate these using your
Assignment 7 Naive Bayes and HMM implementations; do not use gold tags as
predictions.
"""

from pathlib import Path
import pickle
import pandas as pd
import matplotlib.pyplot as plt

LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "Hindi-POS-Data"
OUTPUT_DIR = LAB_DIR / "output"
DEV_FILE = DATA_DIR / "hindi-dev-pos.txt"
TEST_FILE = DATA_DIR / "hindi-test-pos.txt"


def load_pos_data(filepath):
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
                raise ValueError(f"Expected word<TAB>tag: {raw_line.rstrip()!r}")
            sentence.append((parts[0], parts[1]))
    if sentence:
        sentences.append(sentence)
    if not sentences:
        raise ValueError(f"No sentences loaded from {filepath}")
    return sentences


def flatten_gold(sentences):
    return [tag for s in sentences for _, tag in s]


def flatten_pred(predictions):
    return [tag for s in predictions for tag in s]


def metrics_for_split(sentences, predictions_by_model, split_name):
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, f1_score
    )

    y_true = flatten_gold(sentences)
    all_tags = sorted(set(y_true))
    summary_rows, tag_rows = [], []

    for model_name, predictions in predictions_by_model.items():
        y_pred = flatten_pred(predictions)
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"{model_name} {split_name}: {len(y_pred)} predictions for "
                f"{len(y_true)} gold tokens. Check sentence/token alignment."
            )

        summary_rows.append({
            "Split": split_name,
            "Model": model_name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Macro F1": f1_score(
                y_true, y_pred, average="macro", zero_division=0
            ),
            "Weighted F1": f1_score(
                y_true, y_pred, average="weighted", zero_division=0
            )
        })

        p, r, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=all_tags, zero_division=0
        )
        for tag, precision, recall, score, count in zip(
            all_tags, p, r, f1, support
        ):
            tag_rows.append({
                "Split": split_name,
                "Model": model_name,
                "POS Tag": tag,
                "Precision": precision,
                "Recall": recall,
                "F1-score": score,
                "Support": count
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(tag_rows)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = load_pos_data(DEV_FILE)
    test = load_pos_data(TEST_FILE)

    memm_dev = load_pickle(OUTPUT_DIR / "memm_dev_predictions.pkl")
    memm_test = load_pickle(OUTPUT_DIR / "memm_test_predictions.pkl")
    crf_dev = load_pickle(OUTPUT_DIR / "crf_dev_predictions.pkl")
    crf_test = load_pickle(OUTPUT_DIR / "crf_test_predictions.pkl")

    baseline_path = OUTPUT_DIR / "assignment7_predictions.pkl"
    if not baseline_path.exists():
        raise FileNotFoundError(
            "Create assignment7_predictions.pkl from your Assignment 7 models. "
            "Expected keys: naive_bayes_dev, naive_bayes_test, hmm_dev, hmm_test. "
            "Each value should be a list of predicted tag sequences."
        )
    baseline = load_pickle(baseline_path)

    required = {"naive_bayes_dev", "naive_bayes_test", "hmm_dev", "hmm_test"}
    missing = required - set(baseline)
    if missing:
        raise KeyError(f"Missing keys in assignment7_predictions.pkl: {missing}")

    dev_models = {
        "MEMM": memm_dev,
        "CRF": crf_dev,
        "Naive Bayes": baseline["naive_bayes_dev"],
        "HMM": baseline["hmm_dev"],
    }
    test_models = {
        "MEMM": memm_test,
        "CRF": crf_test,
        "Naive Bayes": baseline["naive_bayes_test"],
        "HMM": baseline["hmm_test"],
    }

    dev_summary, dev_tags = metrics_for_split(dev, dev_models, "Dev")
    test_summary, test_tags = metrics_for_split(test, test_models, "Test")

    summary = pd.concat([dev_summary, test_summary], ignore_index=True)
    per_tag = pd.concat([dev_tags, test_tags], ignore_index=True)

    print("\nOVERALL COMPARISON")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nPER-TAG TEST RESULTS")
    print(per_tag[per_tag["Split"] == "Test"].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))

    summary.to_csv(OUTPUT_DIR / "overall_model_comparison.csv", index=False)
    per_tag.to_csv(OUTPUT_DIR / "per_tag_comparison.csv", index=False)

    # Overall chart: accuracy and F1 scores for all four models.
    test_summary = summary[summary["Split"] == "Test"].set_index("Model")
    test_summary[["Accuracy", "Macro F1", "Weighted F1"]].plot(
        kind="bar", figsize=(10, 6), ylim=(0, 1)
    )
    plt.title("Overall POS Tagger Comparison - Test Set")
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "overall_model_comparison.png", dpi=300)
    plt.show()

    # Per-tag F1 chart on test set.
    test_tags = per_tag[per_tag["Split"] == "Test"]
    pivot = test_tags.pivot(
        index="POS Tag", columns="Model", values="F1-score"
    )
    pivot.plot(kind="bar", figsize=(max(12, len(pivot) * 0.55), 7), width=0.8)
    plt.title("Per-POS-Tag F1 Comparison - Test Set")
    plt.xlabel("POS Tag")
    plt.ylabel("F1-score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pos_tag_f1_comparison.png", dpi=300)
    plt.show()

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'overall_model_comparison.csv'}")
    print(f"- {OUTPUT_DIR / 'per_tag_comparison.csv'}")
    print(f"- {OUTPUT_DIR / 'overall_model_comparison.png'}")
    print(f"- {OUTPUT_DIR / 'pos_tag_f1_comparison.png'}")


if __name__ == "__main__":
    main()
