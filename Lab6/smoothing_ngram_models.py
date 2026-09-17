from collections import Counter, defaultdict
import csv
import math
import random
from functools import lru_cache
from pathlib import Path


# CONFIGURATION
FILE_NAME = (
    Path(__file__).resolve().parent.parent
    / "Lab1" / "output" / "indiccorp" / "indiccorp_hi_tokenized.parquet"
)

TOTAL_SENTENCES = 100000
TRAIN_SIZE = 98000
DEV_SIZE = 1000
TEST_SIZE = 1000
SEED = 42
DISCOUNT = 0.75
BACKOFF_ALPHA = 0.4
OUTPUT_FILE = Path(__file__).resolve().parent / "smoothing_results.csv"


# 1. READ TOKENIZED SENTENCES

def load_sentences(filename, limit=100000):
    sentences = []

    if str(filename).lower().endswith(".parquet"):
        import pyarrow.parquet as pq

        table = pq.read_table(filename, columns=["tokenized_sentence"])

        for sentence in table.column("tokenized_sentence").to_pylist():
            if sentence and sentence.strip():
                sentences.append(sentence.strip().split())

            if len(sentences) == limit:
                break

        return sentences

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                sentences.append(line.split())

            if len(sentences) == limit:
                break

    return sentences


# 2. REPLACE WORDS NOT IN TRAINING VOCABULARY WITH <UNK>

def replace_unknown_words(sentences, vocabulary):
    return [
        [word if word in vocabulary else "<UNK>" for word in sentence]
        for sentence in sentences
    ]


# 3. BUILD ALL FOUR N-GRAM MODELS

def build_ngram_models(sentences, max_order=4):
    models = {}

    for n in range(1, max_order + 1):
        ngram_counts = Counter()
        context_counts = Counter()
        ngrams_by_context = defaultdict(list)
        vocabulary = set()

        for sentence in sentences:
            if n == 1:
                tokens = sentence + ["</s>"]
            else:
                tokens = ["<s>"] * (n - 1) + sentence + ["</s>"]

            vocabulary.update(tokens)

            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i + n])
                ngram_counts[ngram] += 1

                if n > 1:
                    context_counts[ngram[:-1]] += 1
                    ngrams_by_context[ngram[:-1]].append(ngram)

        models[n] = {
            "ngram_counts": ngram_counts,
            "context_counts": context_counts,
            "ngrams_by_context": ngrams_by_context,
            "vocabulary": vocabulary,
            "total_tokens": sum(ngram_counts.values()),
        }

    return models


# 4. COMMON PROBABILITY HELPERS

def mle_probability(ngram, models):
    n = len(ngram)
    model = models[n]
    count = model["ngram_counts"].get(ngram, 0)

    if n == 1:
        return count / model["total_tokens"] if model["total_tokens"] else 0.0

    context_count = model["context_counts"].get(ngram[:-1], 0)
    return count / context_count if context_count else 0.0


def unigram_probability(word, models):
    model = models[1]
    count = model["ngram_counts"].get((word,), 0)
    vocabulary_size = len(model["vocabulary"])
    return (count + 1) / (model["total_tokens"] + vocabulary_size)


def safe_probability(probability):
    return max(probability, 1e-12)


# 5. INTERPOLATED SMOOTHING

INTERPOLATION_WEIGHTS = {
    2: (0.4, 0.6),
    3: (0.2, 0.3, 0.5),
    4: (0.1, 0.2, 0.3, 0.4),
}


def interpolated_probability(ngram, models):
    n = len(ngram)

    if n == 1:
        return unigram_probability(ngram[0], models)

    weights = INTERPOLATION_WEIGHTS[n]
    probability = weights[-1] * mle_probability(ngram, models)

    for lower_order in range(1, n):
        probability += weights[lower_order - 1] * interpolated_probability(
            ngram[-lower_order:], models
        )

    return probability


# 6. GOOD TURING SMOOTHING

def build_count_of_counts(models):
    return {
        n: Counter(model["ngram_counts"].values())
        for n, model in models.items()
    }


def good_turing_adjusted_count(n, count, count_of_counts, vocabulary_size):
    frequencies = count_of_counts[n]

    if count == 0:
        n_zero = max(1, vocabulary_size ** n - sum(frequencies.values()))
        return frequencies.get(1, 0) / n_zero if frequencies.get(1, 0) else 0.01

    if frequencies.get(count, 0) and frequencies.get(count + 1, 0):
        return (count + 1) * frequencies[count + 1] / frequencies[count]

    return float(count)


def good_turing_probability(ngram, models, count_of_counts):
    n = len(ngram)
    model = models[n]
    vocabulary_size = len(model["vocabulary"])
    count = model["ngram_counts"].get(ngram, 0)
    adjusted = good_turing_adjusted_count(n, count, count_of_counts, vocabulary_size)

    if n == 1:
        return adjusted / model["total_tokens"]

    context_count = model["context_counts"].get(ngram[:-1], 0)
    if context_count == 0:
        return good_turing_probability(ngram[1:], models, count_of_counts)

    return adjusted / context_count


# 7. KATZ BACKOFF SMOOTHING

def katz_probability(ngram, models, count_of_counts, katz_cache=None):
    if katz_cache is None:
        katz_cache = {}
    n = len(ngram)

    if n == 1:
        return unigram_probability(ngram[0], models)

    model = models[n]
    context = ngram[:-1]
    context_count = model["context_counts"].get(context, 0)

    if context_count == 0:
        return katz_probability(ngram[1:], models, count_of_counts, katz_cache)

    count = model["ngram_counts"].get(ngram, 0)
    if count:
        adjusted = good_turing_adjusted_count(
            n, count, count_of_counts, len(model["vocabulary"])
        )
        return min(1.0, adjusted / count) * count / context_count

    context_key = (n, context)
    if context_key not in katz_cache:
        observed = model["ngrams_by_context"].get(context, [])
        seen_mass = 0.0
        lower_seen_mass = 0.0

        for item in observed:
            item_count = model["ngram_counts"][item]
            adjusted = good_turing_adjusted_count(
                n, item_count, count_of_counts, len(model["vocabulary"])
            )
            seen_mass += min(1.0, adjusted / item_count) * item_count / context_count
            lower_seen_mass += katz_probability(
                item[1:], models, count_of_counts, katz_cache
            )

        katz_cache[context_key] = (
            1.0 - seen_mass,
            max(1e-12, 1.0 - lower_seen_mass),
        )

    remaining_mass, lower_remaining_mass = katz_cache[context_key]
    alpha = remaining_mass / lower_remaining_mass
    return alpha * katz_probability(ngram[1:], models, count_of_counts, katz_cache)


# 8. STUPID BACKOFF SMOOTHING

def stupid_backoff_probability(ngram, models):
    n = len(ngram)
    if n == 1:
        return safe_probability(mle_probability(ngram, models))

    probability = mle_probability(ngram, models)
    if probability:
        return probability

    return BACKOFF_ALPHA * stupid_backoff_probability(ngram[1:], models)


# 9. KNESER-NEY SMOOTHING

def build_kneser_ney_statistics(models):
    bigrams = models[2]["ngram_counts"]
    predecessor_sets = defaultdict(set)
    successor_sets = defaultdict(set)
    follower_counts = {
        n: Counter(ngram[:-1] for ngram in model["ngram_counts"])
        for n, model in models.items() if n > 1
    }

    for previous, word in bigrams:
        predecessor_sets[word].add(previous)
        successor_sets[previous].add(word)

    return predecessor_sets, successor_sets, len(bigrams), follower_counts


def kneser_ney_probability(ngram, models, statistics):
    predecessor_sets, successor_sets, total_bigrams, follower_counts = statistics
    n = len(ngram)

    if n == 1:
        return max(1, len(predecessor_sets[ngram[0]])) / max(1, total_bigrams)

    model = models[n]
    context = ngram[:-1]
    context_count = model["context_counts"].get(context, 0)
    if context_count == 0:
        return kneser_ney_probability(ngram[1:], models, statistics)

    count = model["ngram_counts"].get(ngram, 0)
    distinct_followers = follower_counts[n].get(context, 0)
    backoff_weight = DISCOUNT * distinct_followers / context_count

    return (
        max(count - DISCOUNT, 0) / context_count
        + backoff_weight * kneser_ney_probability(ngram[1:], models, statistics)
    )


# 10. SENTENCE PROBABILITY AND PERPLEXITY

def get_sentence_tokens(sentence, n):
    if n == 1:
        return sentence + ["</s>"]
    return ["<s>"] * (n - 1) + sentence + ["</s>"]


def smoothing_probability(ngram, method, models, count_of_counts, kneser_ney_statistics):
    if method == "Interpolated":
        probability = interpolated_probability(ngram, models)
    elif method == "Good Turing":
        probability = good_turing_probability(ngram, models, count_of_counts)
    elif method == "Katz Backoff":
        probability = katz_probability(ngram, models, count_of_counts, {})
    elif method == "Stupid Backoff":
        probability = stupid_backoff_probability(ngram, models)
    else:
        probability = kneser_ney_probability(ngram, models, kneser_ney_statistics)

    return safe_probability(probability)


def calculate_perplexity(sentences, n, method, models, count_of_counts, kneser_ney_statistics):
    total_log_probability = 0.0
    total_ngrams = 0
    katz_cache = {}

    for sentence in sentences:
        tokens = get_sentence_tokens(sentence, n)

        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            if method == "Katz Backoff":
                probability = safe_probability(
                    katz_probability(ngram, models, count_of_counts, katz_cache)
                )
            else:
                probability = smoothing_probability(
                    ngram, method, models, count_of_counts, kneser_ney_statistics
                )
            total_log_probability += math.log(probability)

        total_ngrams += len(tokens) - n + 1

    return math.exp(-total_log_probability / total_ngrams)


# 11. EVALUATE ALL MODELS

def evaluate_models(models, development_data, test_data):
    count_of_counts = build_count_of_counts(models)
    kneser_ney_statistics = build_kneser_ney_statistics(models)
    methods = [
        "Interpolated",
        "Good Turing",
        "Katz Backoff",
        "Stupid Backoff",
        "Kneser-Ney",
    ]
    model_names = {1: "Unigram", 2: "Bigram", 3: "Trigram", 4: "Quadgram"}
    results = []

    for method in methods:
        for n in range(1, 5):
            if method == "Interpolated" and n == 1:
                continue

            results.append({
                "Smoothing": method,
                "Model": model_names[n],
                "Development": calculate_perplexity(
                    development_data, n, method, models,
                    count_of_counts, kneser_ney_statistics
                ),
                "Test": calculate_perplexity(
                    test_data, n, method, models,
                    count_of_counts, kneser_ney_statistics
                ),
            })

    return results


def print_results(results):
    print("\n" + "=" * 80)
    print("PERPLEXITY RESULTS")
    print("=" * 80)
    print("{:<20} {:<12} {:>18} {:>18}".format(
        "Smoothing", "Model", "Development", "Test"
    ))
    print("-" * 80)

    for result in results:
        print("{:<20} {:<12} {:>18.4f} {:>18.4f}".format(
            result["Smoothing"], result["Model"],
            result["Development"], result["Test"]
        ))


def save_results(results, filename):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


# 12. MAIN PROGRAM

def main():
    print("=" * 80)
    print("HINDI N-GRAM LANGUAGE MODEL")
    print("SMOOTHING TECHNIQUES")
    print("=" * 80)

    print("\nReading dataset...")
    sentences = load_sentences(FILE_NAME, TOTAL_SENTENCES)
    print("Sentences loaded:", len(sentences))

    if len(sentences) < TOTAL_SENTENCES:
        print("ERROR: Dataset contains fewer than 100,000 sentences.")
        return

    random.Random(SEED).shuffle(sentences)
    training_data = sentences[:TRAIN_SIZE]
    development_data = sentences[TRAIN_SIZE:TRAIN_SIZE + DEV_SIZE]
    test_data = sentences[TRAIN_SIZE + DEV_SIZE:TOTAL_SENTENCES]

    training_vocabulary = {word for sentence in training_data for word in sentence}
    training_vocabulary.add("<UNK>")
    training_data = replace_unknown_words(training_data, training_vocabulary)
    development_data = replace_unknown_words(development_data, training_vocabulary)
    test_data = replace_unknown_words(test_data, training_vocabulary)

    print("\nDATASET SPLIT")
    print("Training sentences    :", len(training_data))
    print("Development sentences :", len(development_data))
    print("Test sentences        :", len(test_data))

    models = build_ngram_models(training_data)
    results = evaluate_models(models, development_data, test_data)
    print_results(results)
    save_results(results, OUTPUT_FILE)
    print("\nResults saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()