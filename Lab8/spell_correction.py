import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "Spell-Correction"
SMOOTHING = 0.1


def read_unigram_counts(path):
    counts = Counter()
    with path.open(encoding="utf-8") as data:
        for line_number, line in enumerate(data, start=1):
            word, separator, count = line.rstrip("\r\n").partition("\t")
            if not separator:
                raise ValueError(f"Expected a tab-separated word and count at {path}:{line_number}")
            counts[word.casefold()] += int(count)
    return counts


def read_edit_counts(path):
    counts = Counter()
    with path.open(encoding="latin-1") as data:
        for line_number, line in enumerate(data, start=1):
            pair, separator, count = line.rstrip("\r\n").partition("\t")
            if not separator or "|" not in pair:
                raise ValueError(f"Expected an edit pair and count at {path}:{line_number}")
            source, target = pair.split("|", 1)
            counts[(source.casefold(), target.casefold())] += int(count)
    return counts


def read_spell_errors(path):
    examples = []
    with path.open(encoding="utf-8") as data:
        for line_number, line in enumerate(data, start=1):
            correct, separator, misspellings = line.rstrip("\r\n").partition(":")
            if not separator or not correct.strip():
                raise ValueError(f"Expected a correct word and misspellings at {path}:{line_number}")

            for item in misspellings.split(","):
                item = item.strip()
                if not item:
                    continue
                if "*" in item:
                    misspelling, weight_text = item.rsplit("*", 1)
                    try:
                        weight = int(weight_text)
                    except ValueError as error:
                        raise ValueError(
                            f"Invalid typo weight at {path}:{line_number}: {item}"
                        ) from error
                else:
                    misspelling, weight = item, 1

                misspelling = misspelling.strip().casefold()
                if not misspelling or weight < 1:
                    raise ValueError(f"Invalid typo example at {path}:{line_number}: {item}")
                examples.append((correct.strip().casefold(), misspelling, weight))
    return examples


def edit_operations(correct, misspelling):
    """Return one-edit (correct fragment, typo fragment) alignments."""
    operations = set()

    if len(correct) == len(misspelling):
        differences = [
            index
            for index, (correct_char, typo_char) in enumerate(zip(correct, misspelling))
            if correct_char != typo_char
        ]
        if len(differences) == 1:
            index = differences[0]
            operations.add((correct[index], misspelling[index]))
        elif len(differences) == 2:
            index = differences[0]
            if (
                differences[1] == index + 1
                and correct[index] == misspelling[index + 1]
                and correct[index + 1] == misspelling[index]
            ):
                operations.add((correct[index:index + 2], misspelling[index:index + 2]))
    elif len(misspelling) == len(correct) + 1:
        for index, inserted_char in enumerate(misspelling):
            if correct == misspelling[:index] + misspelling[index + 1:]:
                context = correct[index - 1] if index else ""
                operations.add((context, context + inserted_char))
    elif len(correct) == len(misspelling) + 1:
        for index, deleted_char in enumerate(correct):
            if misspelling == correct[:index] + correct[index + 1:]:
                context = correct[index - 1] if index else ""
                operations.add((context + deleted_char, context))

    return operations


class NoisyChannelSpellCorrector:
    def __init__(self, unigram_counts, edit_counts, smoothing=SMOOTHING):
        self.unigram_counts = unigram_counts
        self.edit_counts = edit_counts
        self.smoothing = smoothing
        self.total_unigrams = sum(unigram_counts.values())
        self.edit_totals = Counter()
        self.edit_outcomes = defaultdict(set)
        self.deletion_index = defaultdict(list)

        for (source, target), count in edit_counts.items():
            self.edit_totals[source] += count
            self.edit_outcomes[source].add(target)

        for word in unigram_counts:
            for index in range(len(word)):
                self.deletion_index[word[:index] + word[index + 1:]].append(word)

    def candidates(self, misspelling):
        misspelling = misspelling.casefold()
        candidates = set()
        for index in range(len(misspelling)):
            deleted = misspelling[:index] + misspelling[index + 1:]
            if deleted in self.unigram_counts:
                candidates.add(deleted)
            candidates.update(self.deletion_index.get(deleted, ()))
        candidates.update(self.deletion_index.get(misspelling, ()))
        candidates.discard(misspelling)
        return {
            candidate
            for candidate in candidates
            if edit_operations(candidate, misspelling)
        }

    def _channel_probability(self, operation):
        source, target = operation
        outcomes = len(self.edit_outcomes[source]) + 1
        denominator = self.edit_totals[source] + self.smoothing * outcomes
        return (self.edit_counts[operation] + self.smoothing) / denominator

    def score(self, correct, misspelling):
        operations = edit_operations(correct, misspelling)
        if not operations:
            return float("-inf")

        channel_probability = sum(
            self._channel_probability(operation) for operation in operations
        )
        prior_probability = self.unigram_counts[correct] / self.total_unigrams
        return math.log(prior_probability) + math.log(channel_probability)

    def correct(self, misspelling):
        misspelling = misspelling.casefold()
        candidates = self.candidates(misspelling)
        return self._best_candidate(misspelling, candidates)

    def _best_candidate(self, misspelling, candidates):
        if not candidates:
            return misspelling
        return min(
            candidates,
            key=lambda candidate: (
                -self.score(candidate, misspelling),
                -self.unigram_counts[candidate],
                candidate,
            ),
        )


def evaluate(corrector, examples):
    prediction_cache = {}
    total_weight = 0
    correct_weight = 0
    covered_weight = 0

    for correct, misspelling, weight in examples:
        if misspelling not in prediction_cache:
            candidates = corrector.candidates(misspelling)
            prediction = corrector._best_candidate(misspelling, candidates)
            prediction_cache[misspelling] = (prediction, candidates)

        prediction, candidates = prediction_cache[misspelling]
        total_weight += weight
        if correct in candidates:
            covered_weight += weight
        if prediction == correct:
            correct_weight += weight

    return {
        "unique_misspellings": len(prediction_cache),
        "total_weight": total_weight,
        "correct_weight": correct_weight,
        "covered_weight": covered_weight,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate or use the unigram noisy-channel spell corrector."
    )
    parser.add_argument(
        "words",
        nargs="*",
        help="optional misspelled words to correct instead of evaluating the test data",
    )
    args = parser.parse_args()

    corrector = NoisyChannelSpellCorrector(
        read_unigram_counts(DATA_DIR / "count_unigram.txt"),
        read_edit_counts(DATA_DIR / "count_1edit.txt"),
    )

    if args.words:
        for word in args.words:
            print(f"{word}\t{corrector.correct(word)}")
        return

    results = evaluate(corrector, read_spell_errors(DATA_DIR / "spell-errors.txt"))
    total = results["total_weight"]
    covered = results["covered_weight"]
    accuracy = results["correct_weight"] / total if total else 0.0
    coverage = covered / total if total else 0.0
    covered_accuracy = results["correct_weight"] / covered if covered else 0.0

    print(f"Unique misspellings: {results['unique_misspellings']}")
    print(f"Weighted test examples: {total}")
    print(f"Gold candidate coverage: {covered}/{total} ({coverage:.2%})")
    print(f"Correct predictions: {results['correct_weight']}/{total}")
    print(f"Overall accuracy: {accuracy:.2%}")
    print(f"Accuracy when the gold word is a candidate: {covered_accuracy:.2%}")


if __name__ == "__main__":
    main()
