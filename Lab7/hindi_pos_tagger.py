import argparse
import csv
import html
import math
from collections import Counter, defaultdict
from pathlib import Path


BOUNDARY = "<BOUNDARY>"
MODELS = (
    "Naive Bayes",
    "Forward Viterbi",
    "Backward Viterbi",
    "Forward-backward Viterbi",
)
MODEL_FILENAMES = {
    "Naive Bayes": "naive_bayes",
    "Forward Viterbi": "forward_viterbi",
    "Backward Viterbi": "backward_viterbi",
    "Forward-backward Viterbi": "forward_backward_viterbi",
}


def read_data(path):
    """Read a tab-separated word/tag corpus; blank lines separate sentences."""
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
    return sentences


def get_features(words, position, previous_tag):
    """Return the five requested Naive Bayes features for one word."""
    word = words[position]
    previous_word = words[position - 1] if position > 0 else BOUNDARY
    next_word = words[position + 1] if position + 1 < len(words) else BOUNDARY
    length = "MORE" if len(word) > 4 else "LESS"
    return length, word, previous_word, next_word, previous_tag


class NaiveBayesTagger:
    def __init__(self, smoothing=0.1):
        self.smoothing = smoothing
        self.tags = []
        self.tag_counts = Counter()
        self.feature_counts = [defaultdict(Counter) for _ in range(5)]
        self.feature_sizes = [0] * 5
        self.token_count = 0

    def fit(self, sentences):
        feature_values = [set() for _ in range(5)]

        for sentence in sentences:
            words = [word for word, _ in sentence]
            previous_tag = BOUNDARY

            for position, (word, tag) in enumerate(sentence):
                self.tag_counts[tag] += 1
                self.token_count += 1
                features = get_features(words, position, previous_tag)
                for feature_number, value in enumerate(features):
                    self.feature_counts[feature_number][tag][value] += 1
                    feature_values[feature_number].add(value)
                previous_tag = tag

        self.tags = sorted(self.tag_counts)
        # One extra value in each vocabulary accounts for unseen features.
        self.feature_sizes = [len(values) + 1 for values in feature_values]

    def predict(self, words):
        predictions = []

        for position in range(len(words)):
            previous_tag = predictions[-1] if predictions else BOUNDARY
            features = get_features(words, position, previous_tag)
            scores = {}

            for tag in self.tags:
                prior = (self.tag_counts[tag] + self.smoothing) / (
                    self.token_count + self.smoothing * len(self.tags)
                )
                score = math.log(prior)

                for feature_number, value in enumerate(features):
                    count = self.feature_counts[feature_number][tag][value]
                    denominator = self.tag_counts[tag] + (
                        self.smoothing * self.feature_sizes[feature_number]
                    )
                    score += math.log((count + self.smoothing) / denominator)

                scores[tag] = score

            predictions.append(max(scores, key=scores.get))

        return predictions


class HMMTagger:
    def __init__(self, smoothing=0.1):
        self.smoothing = smoothing
        self.tags = []
        self.tag_counts = Counter()
        self.word_counts = defaultdict(Counter)
        self.transitions = Counter()
        self.outgoing_counts = Counter()
        self.vocabulary = set()
        self.start = []
        self.end = []
        self.transition_probs = []
        self.emission_denominators = []

    def fit(self, sentences):
        for sentence in sentences:
            previous_tag = BOUNDARY

            for word, tag in sentence:
                self.transitions[previous_tag, tag] += 1
                self.outgoing_counts[previous_tag] += 1
                self.tag_counts[tag] += 1
                self.word_counts[tag][word] += 1
                self.vocabulary.add(word)
                previous_tag = tag

            self.transitions[previous_tag, BOUNDARY] += 1
            self.outgoing_counts[previous_tag] += 1

        self.tags = sorted(self.tag_counts)
        tag_count = len(self.tags)
        transition_choices = tag_count + 1  # tags plus end-of-sentence

        def transition_log_probability(previous, current):
            numerator = self.transitions[previous, current] + self.smoothing
            denominator = self.outgoing_counts[previous] + (
                self.smoothing * transition_choices
            )
            return math.log(numerator / denominator)

        self.start = [
            transition_log_probability(BOUNDARY, tag) for tag in self.tags
        ]
        self.end = [
            transition_log_probability(tag, BOUNDARY) for tag in self.tags
        ]
        self.transition_probs = [
            [transition_log_probability(previous, current) for current in self.tags]
            for previous in self.tags
        ]

        # Include an unknown-word outcome in each tag's emission distribution.
        emission_choices = len(self.vocabulary) + 1
        self.emission_denominators = [
            math.log(
                self.tag_counts[tag] + self.smoothing * emission_choices
            )
            for tag in self.tags
        ]

    def emission(self, word, tag_number):
        tag = self.tags[tag_number]
        count = self.word_counts[tag][word]
        return math.log(count + self.smoothing) - self.emission_denominators[tag_number]

    def predict_forward(self, words):
        """Standard left-to-right Viterbi decoding."""
        if not words:
            return []

        tag_count = len(self.tags)
        scores = [
            self.start[tag_number] + self.emission(words[0], tag_number)
            for tag_number in range(tag_count)
        ]
        backpointers = [[-1] * tag_count]

        for position in range(1, len(words)):
            next_scores = []
            previous_tags = []

            for current in range(tag_count):
                candidates = [
                    scores[previous] + self.transition_probs[previous][current]
                    for previous in range(tag_count)
                ]
                best_previous = max(range(tag_count), key=candidates.__getitem__)
                previous_tags.append(best_previous)
                next_scores.append(
                    candidates[best_previous] + self.emission(words[position], current)
                )

            scores = next_scores
            backpointers.append(previous_tags)

        last_tag = max(
            range(tag_count), key=lambda tag: scores[tag] + self.end[tag]
        )
        path = [last_tag]
        for position in range(len(words) - 1, 0, -1):
            path.append(backpointers[position][path[-1]])
        path.reverse()
        return [self.tags[tag_number] for tag_number in path]

    def predict_backward(self, words):
        """Viterbi decoding using the recurrence from right to left."""
        if not words:
            return []

        tag_count = len(self.tags)
        next_scores = [
            self.emission(words[-1], tag) + self.end[tag]
            for tag in range(tag_count)
        ]
        next_pointers = [[-1] * tag_count for _ in words]

        for position in range(len(words) - 2, -1, -1):
            current_scores = []

            for current in range(tag_count):
                candidates = [
                    self.transition_probs[current][following] + next_scores[following]
                    for following in range(tag_count)
                ]
                best_next = max(range(tag_count), key=candidates.__getitem__)
                next_pointers[position][current] = best_next
                current_scores.append(
                    candidates[best_next] + self.emission(words[position], current)
                )

            next_scores = current_scores

        first_tag = max(
            range(tag_count),
            key=lambda tag: self.start[tag] + next_scores[tag],
        )
        path = [first_tag]
        for position in range(len(words) - 1):
            path.append(next_pointers[position][path[-1]])
        return [self.tags[tag_number] for tag_number in path]

    def predict_forward_backward(self, words):
        """Choose each tag from its combined forward and backward score."""
        if not words:
            return []

        tag_count = len(self.tags)
        forward = [[0.0] * tag_count for _ in words]
        backward = [[0.0] * tag_count for _ in words]

        for tag in range(tag_count):
            forward[0][tag] = self.start[tag] + self.emission(words[0], tag)
            backward[-1][tag] = self.end[tag]

        for position in range(1, len(words)):
            for current in range(tag_count):
                candidates = [
                    forward[position - 1][previous]
                    + self.transition_probs[previous][current]
                    for previous in range(tag_count)
                ]
                forward[position][current] = (
                    max(candidates) + self.emission(words[position], current)
                )

        for position in range(len(words) - 2, -1, -1):
            for current in range(tag_count):
                candidates = [
                    self.transition_probs[current][following]
                    + self.emission(words[position + 1], following)
                    + backward[position + 1][following]
                    for following in range(tag_count)
                ]
                backward[position][current] = max(candidates)

        path = []
        for position in range(len(words)):
            best_tag = max(
                range(tag_count),
                key=lambda tag: forward[position][tag] + backward[position][tag],
            )
            path.append(self.tags[best_tag])
        return path


def calculate_scores(gold_sentences, predicted_sentences):
    """Return per-tag precision, recall, F1, plus overall summary scores."""
    counts = defaultdict(Counter)
    correct = 0
    token_count = 0

    for sentence_number, (gold, predicted) in enumerate(
        zip(gold_sentences, predicted_sentences), start=1
    ):
        if len(gold) != len(predicted):
            raise ValueError(f"Prediction length mismatch in sentence {sentence_number}")

        for (_, actual), guessed in zip(gold, predicted):
            token_count += 1
            correct += actual == guessed
            counts[actual]["support"] += 1
            if actual == guessed:
                counts[actual]["tp"] += 1
            else:
                counts[actual]["fn"] += 1
                counts[guessed]["fp"] += 1

    tags = sorted(counts)
    metrics = {}
    for tag in tags:
        true_positive = counts[tag]["tp"]
        false_positive = counts[tag]["fp"]
        false_negative = counts[tag]["fn"]
        precision = true_positive / (true_positive + false_positive) if (
            true_positive + false_positive
        ) else 0.0
        recall = true_positive / (true_positive + false_negative) if (
            true_positive + false_negative
        ) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (
            precision + recall
        ) else 0.0
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
        ) / token_count,
        "tokens": token_count,
    }
    return tags, metrics, summary


def save_metrics(path, results):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("model", "tag", "precision", "recall", "f1", "support"))
        for model in MODELS:
            for tag, scores in results[model].items():
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


def save_predictions(path, gold_sentences, predicted_sentences):
    if len(gold_sentences) != len(predicted_sentences):
        raise ValueError("Gold and predicted sentence counts do not match")

    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(("word", "gold_tag", "predicted_tag"))

        for sentence_number, (gold, predicted) in enumerate(
            zip(gold_sentences, predicted_sentences), start=1
        ):
            if len(gold) != len(predicted):
                raise ValueError(
                    f"Prediction length mismatch in sentence {sentence_number}"
                )
            for (word, gold_tag), predicted_tag in zip(gold, predicted):
                writer.writerow((word, gold_tag, predicted_tag))
            writer.writerow(())


def save_chart(path, tags, results, split):
    """Create a simple grouped bar chart as an SVG file."""
    left = 240
    chart_width = 780
    row_height = 100
    top = 100
    height = top + len(tags) * row_height + 35
    colors = ("#2878B5", "#E07A24", "#3A9D5D", "#B34D9A")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.title{font-size:23px;'
        'font-weight:bold}.label{font-size:14px}.axis{font-size:12px}</style>',
        f'<text x="560" y="34" text-anchor="middle" class="title">'
        f'Hindi POS tag F1 scores ({html.escape(split)})</text>',
    ]

    legend_x = left
    for model, color in zip(MODELS, colors):
        parts.extend(
            (
                f'<rect x="{legend_x}" y="55" width="14" height="14" fill="{color}"/>',
                f'<text x="{legend_x + 20}" y="67" class="label">'
                f'{html.escape(model)}</text>',
            )
        )
        legend_x += 190

    for tick in range(0, 11, 2):
        x = left + chart_width * tick / 10
        parts.append(
            f'<line x1="{x}" y1="{top}" x2="{x}" y2="{height - 25}" stroke="#ddd"/>'
        )
        parts.append(
            f'<text x="{x}" y="{height - 8}" text-anchor="middle" '
            f'class="axis">{tick / 10:.1f}</text>'
        )

    for tag_number, tag in enumerate(tags):
        group_top = top + tag_number * row_height
        parts.append(
            f'<text x="{left - 12}" y="{group_top + 48}" text-anchor="end" '
            f'class="label">{html.escape(tag)}</text>'
        )
        for model_number, (model, color) in enumerate(zip(MODELS, colors)):
            f1 = results[model][tag]["f1"]
            bar_width = chart_width * f1
            y = group_top + 5 + model_number * 21
            parts.append(
                f'<rect x="{left}" y="{y}" width="{bar_width:.2f}" height="15" '
                f'rx="2" fill="{color}"/>'
            )

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def save_report(path, summaries):
    lines = [
        "# Hindi POS tagger results",
        "",
        "| Split | Model | Accuracy | Macro F1 | Weighted F1 | Tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for split, models in summaries.items():
        for model in MODELS:
            scores = models[model]
            lines.append(
                f"| {split} | {model} | {scores['accuracy']:.4f} | "
                f"{scores['macro_f1']:.4f} | {scores['weighted_f1']:.4f} | "
                f"{scores['tokens']} |"
            )

    lines.extend(
        (
            "",
            "Word-level predictions are saved as TSV files, one file per model "
            "and split, with each word, gold tag, and predicted tag on a row. "
            "Each split's CSV has precision, recall, F1, and support for every tag "
            "and model. The SVG chart compares per-tag F1 scores.",
            "",
            "Naive Bayes uses the five requested features. It uses the preceding "
            "gold tag while training and its own preceding prediction while testing. "
            "HMM probabilities use add-0.1 smoothing. Forward and backward Viterbi "
            "find the best sequence from opposite directions. Forward-backward "
            "combines the best left and right sequence scores at each word, then "
            "chooses the best tag for that word.",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    lab_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=lab_dir / "Hindi-POS-Data")
    parser.add_argument("--output-dir", type=Path, default=lab_dir / "results")
    args = parser.parse_args()

    train = read_data(args.data_dir / "hindi-train-pos.txt")
    splits = {
        "dev": read_data(args.data_dir / "hindi-dev-pos.txt"),
        "test": read_data(args.data_dir / "hindi-test-pos.txt"),
    }
    naive_bayes = NaiveBayesTagger()
    naive_bayes.fit(train)
    hmm = HMMTagger()
    hmm.fit(train)
    predictors = {
        "Naive Bayes": naive_bayes.predict,
        "Forward Viterbi": hmm.predict_forward,
        "Backward Viterbi": hmm.predict_backward,
        "Forward-backward Viterbi": hmm.predict_forward_backward,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}

    for split, sentences in splits.items():
        split_metrics = {}
        summaries[split] = {}

        for model, predict in predictors.items():
            predictions = [
                predict([word for word, _ in sentence]) for sentence in sentences
            ]
            save_predictions(
                args.output_dir
                / f"{split}_{MODEL_FILENAMES[model]}_predictions.tsv",
                sentences,
                predictions,
            )
            _, scores, summary = calculate_scores(sentences, predictions)
            split_metrics[model] = scores
            summaries[split][model] = summary
            print(
                f"{split:4} | {model:25} | "
                f"accuracy={summary['accuracy']:.4f} "
                f"macro_f1={summary['macro_f1']:.4f} "
                f"weighted_f1={summary['weighted_f1']:.4f}"
            )

        tags = sorted(split_metrics[MODELS[0]])
        save_metrics(args.output_dir / f"{split}_per_tag.csv", split_metrics)
        save_chart(
            args.output_dir / f"{split}_per_tag_f1.svg",
            tags,
            split_metrics,
            split,
        )

    save_report(args.output_dir / "REPORT.md", summaries)
    print(f"Saved results in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()