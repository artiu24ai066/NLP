import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


BOUNDARY = "<BOUNDARY>"
SPLITS = ("dev", "test")
MODEL_NAMES = ("MEMM", "CRF", "Naive Bayes", "HMM")


def read_data(path):
    """Read tab-separated word/tag pairs, with blank lines separating sentences."""
    sentences = []
    sentence = []

    with path.open(encoding="utf-8") as data:
        for line_number, line in enumerate(data, start=1):
            line = line.rstrip("\r\n")
            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue

            fields = line.split("\t")
            if len(fields) != 2 or not all(fields):
                raise ValueError(f"Bad word/tag pair at {path}:{line_number}")
            sentence.append(tuple(fields))

    if sentence:
        sentences.append(sentence)
    if not sentences:
        raise ValueError(f"No tagged sentences found in {path}")
    return sentences


def extract_features(words, position):
    word = words[position]
    features = {
        "previous_word": words[position - 1] if position > 0 else BOUNDARY,
        "current_word": word,
        "next_word": words[position + 1]
        if position + 1 < len(words)
        else BOUNDARY,
        "length_feature": "LESS" if len(word) <= 4 else "MORE",
    }

    for length in range(1, 5):
        features[f"prefix_{length}"] = (
            word[:length] if len(word) >= length else "NULL"
        )
        features[f"suffix_{length}"] = (
            word[-length:] if len(word) >= length else "NULL"
        )
    return features


def sentence_words(sentence):
    return [word for word, _ in sentence]


def sentence_labels(sentence):
    return [tag for _, tag in sentence]


def dataset_tags(train, splits):
    return sorted(
        {tag for sentence in train for _, tag in sentence}
        | {
            tag
            for sentences in splits.values()
            for sentence in sentences
            for _, tag in sentence
        }
    )


def calculate_scores(gold_sentences, predicted_sentences, tags):
    if len(gold_sentences) != len(predicted_sentences):
        raise ValueError("Gold and predicted sentence counts do not match")

    counts = defaultdict(Counter)
    correct = 0
    token_count = 0

    for sentence_number, (gold, predicted) in enumerate(
        zip(gold_sentences, predicted_sentences), start=1
    ):
        gold_tags = sentence_labels(gold)
        if len(gold_tags) != len(predicted):
            raise ValueError(f"Prediction length mismatch in sentence {sentence_number}")

        for actual, guessed in zip(gold_tags, predicted):
            token_count += 1
            correct += actual == guessed
            counts[actual]["support"] += 1
            if actual == guessed:
                counts[actual]["tp"] += 1
            else:
                counts[actual]["fn"] += 1
                counts[guessed]["fp"] += 1

    if not token_count:
        raise ValueError("Cannot calculate metrics for an empty dataset")

    metrics = {}
    for tag in tags:
        true_positive = counts[tag]["tp"]
        false_positive = counts[tag]["fp"]
        false_negative = counts[tag]["fn"]
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        metrics[tag] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": counts[tag]["support"],
        }

    summary = {
        "accuracy": correct / token_count,
        "macro_f1": sum(metrics[tag]["f1"] for tag in tags) / len(tags),
        "weighted_f1": sum(
            metrics[tag]["f1"] * metrics[tag]["support"] for tag in tags
        )
        / token_count,
        "tokens": token_count,
    }
    return metrics, summary


def save_predictions(path, gold_sentences, predicted_sentences):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(("word", "gold_tag", "predicted_tag"))
        for gold, predicted in zip(gold_sentences, predicted_sentences):
            for (word, gold_tag), predicted_tag in zip(gold, predicted):
                writer.writerow((word, gold_tag, predicted_tag))
            writer.writerow(())


def evaluate_model(model_name, split, gold_sentences, predicted_sentences, tags, output_dir):
    metrics, summary = calculate_scores(gold_sentences, predicted_sentences, tags)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_predictions(
        output_dir / f"{split}_{model_name.lower().replace(' ', '_')}_predictions.tsv",
        gold_sentences,
        predicted_sentences,
    )
    with (output_dir / f"{split}_{model_name.lower().replace(' ', '_')}_per_tag.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.writer(output)
        writer.writerow(("tag", "precision", "recall", "f1", "support"))
        for tag in tags:
            scores = metrics[tag]
            writer.writerow(
                (
                    tag,
                    f"{scores['precision']:.6f}",
                    f"{scores['recall']:.6f}",
                    f"{scores['f1']:.6f}",
                    scores["support"],
                )
            )

    print(
        f"{split:4} | {model_name:12} | "
        f"accuracy={summary['accuracy']:.4f} "
        f"macro_f1={summary['macro_f1']:.4f} "
        f"weighted_f1={summary['weighted_f1']:.4f}"
    )
    return metrics, summary


def save_comparison(split, tags, results, output_dir):
    metrics_path = output_dir / f"{split}_comparison_per_tag.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("model", "tag", "precision", "recall", "f1", "support"))
        for model in MODEL_NAMES:
            for tag in tags:
                scores = results[model][tag]
                writer.writerow(
                    (
                        model,
                        tag,
                        f"{scores['precision']:.6f}",
                        f"{scores['recall']:.6f}",
                        f"{scores['f1']:.6f}",
                        scores["support"],
                    )
                )

    save_comparison_chart(
        output_dir / f"{split}_comparison_per_tag_f1.svg",
        split,
        tags,
        results,
    )


def save_comparison_chart(path, split, tags, results):
    left = 250
    chart_width = 850
    row_height = 94
    top = 100
    height = top + len(tags) * row_height + 35
    colors = ("#2878B5", "#E07A24", "#3A9D5D", "#B34D9A")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="{height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:23px;'
        'font-weight:bold}.label{font-size:14px}.axis{font-size:12px}</style>',
        f'<text x="600" y="34" text-anchor="middle" class="title">'
        f'Hindi POS tag F1 comparison ({html.escape(split)})</text>',
    ]

    legend_x = left
    for model, color in zip(MODEL_NAMES, colors):
        parts.extend(
            (
                f'<rect x="{legend_x}" y="55" width="14" height="14" fill="{color}"/>',
                f'<text x="{legend_x + 20}" y="67" class="label">'
                f"{html.escape(model)}</text>",
            )
        )
        legend_x += 205

    for tick in range(0, 11, 2):
        x = left + chart_width * tick / 10
        parts.extend(
            (
                f'<line x1="{x}" y1="{top}" x2="{x}" y2="{height - 25}" stroke="#ddd"/>',
                f'<text x="{x}" y="{height - 8}" text-anchor="middle" '
                f'class="axis">{tick / 10:.1f}</text>',
            )
        )

    for tag_number, tag in enumerate(tags):
        group_top = top + tag_number * row_height
        parts.append(
            f'<text x="{left - 12}" y="{group_top + 48}" text-anchor="end" '
            f'class="label">{html.escape(tag)}</text>'
        )
        for model_number, (model, color) in enumerate(zip(MODEL_NAMES, colors)):
            f1 = results[model][tag]["f1"]
            bar_width = chart_width * f1
            y = group_top + 5 + model_number * 20
            parts.append(
                f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="15" '
                f'rx="2" fill="{color}"/>'
            )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
