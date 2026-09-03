from collections import Counter
import math
import random
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

FILE_NAME = (
    Path(__file__).resolve().parent.parent
    / "Lab1" / "output" / "indiccorp" / "indiccorp_hi_tokenized.parquet"
)

TOTAL_SENTENCES = 100000
TRAIN_SIZE = 98000
DEV_SIZE = 1000
TEST_SIZE = 1000

random.seed(42)


# ============================================================
# 1. READ TOKENIZED SENTENCES
# ============================================================

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

            if not line:
                continue

            # Each line is one tokenized sentence
            tokens = line.split()

            if len(tokens) > 0:
                sentences.append(tokens)

            if len(sentences) == limit:
                break

    return sentences


# ============================================================
# 2. REPLACE WORDS NOT IN TRAINING VOCABULARY WITH <UNK>
# ============================================================

def replace_unknown_words(sentences, vocabulary):

    result = []

    for sentence in sentences:

        new_sentence = []

        for word in sentence:

            if word in vocabulary:
                new_sentence.append(word)
            else:
                new_sentence.append("<UNK>")

        result.append(new_sentence)

    return result


# ============================================================
# 3. BUILD N-GRAM MODEL
# ============================================================

def build_ngram_model(sentences, n):

    ngram_counts = Counter()
    context_counts = Counter()

    vocabulary = set()

    for sentence in sentences:

        # Add boundary tokens
        if n == 1:
            tokens = sentence + ["</s>"]
        else:
            tokens = ["<s>"] * (n - 1) + sentence + ["</s>"]

        for word in tokens:
            vocabulary.add(word)

        # Generate n-grams
        for i in range(len(tokens) - n + 1):

            ngram = tuple(tokens[i:i + n])

            ngram_counts[ngram] += 1

            # Context = first n-1 words
            if n > 1:

                context = tuple(tokens[i:i + n - 1])

                context_counts[context] += 1

    return ngram_counts, context_counts, vocabulary


# ============================================================
# 4. LAPLACE / ADD-ONE SMOOTHING
# ============================================================

def laplace_probability(
        ngram,
        ngram_counts,
        context_counts,
        vocabulary,
        total_tokens=None):

    V = len(vocabulary)

    count_ngram = ngram_counts.get(ngram, 0)

    # --------------------------------------------------------
    # UNIGRAM
    # P(w) = (C(w) + 1) / (N + V)
    # --------------------------------------------------------

    if len(ngram) == 1:

        probability = (
            count_ngram + 1
        ) / (
            total_tokens + V
        )

    # --------------------------------------------------------
    # BIGRAM / TRIGRAM / QUADGRAM
    #
    # P(w_n | previous words)
    #
    # = (C(ngram) + 1)
    #   /
    #   (C(context) + V)
    # --------------------------------------------------------

    else:

        context = ngram[:-1]

        count_context = context_counts.get(context, 0)

        probability = (
            count_ngram + 1
        ) / (
            count_context + V
        )

    return probability


# ============================================================
# 5. CALCULATE SENTENCE LOG PROBABILITY
# ============================================================

def sentence_log_probability(
        sentence,
        n,
        ngram_counts,
        context_counts,
        vocabulary,
        total_tokens):

    # Add boundary tokens

    if n == 1:
        tokens = sentence + ["</s>"]
    else:
        tokens = ["<s>"] * (n - 1) + sentence + ["</s>"]

    log_probability = 0.0

    for i in range(len(tokens) - n + 1):

        ngram = tuple(tokens[i:i + n])

        probability = laplace_probability(
            ngram,
            ngram_counts,
            context_counts,
            vocabulary,
            total_tokens
        )

        log_probability += math.log(probability)

    return log_probability


# ============================================================
# 6. CALCULATE PERPLEXITY
# ============================================================

def calculate_perplexity(
        sentences,
        n,
        ngram_counts,
        context_counts,
        vocabulary,
        total_tokens):

    total_log_probability = 0.0
    total_ngrams = 0

    for sentence in sentences:

        if n == 1:
            tokens = sentence + ["</s>"]
        else:
            tokens = ["<s>"] * (n - 1) + sentence + ["</s>"]

        log_probability = sentence_log_probability(
            sentence,
            n,
            ngram_counts,
            context_counts,
            vocabulary,
            total_tokens
        )

        total_log_probability += log_probability

        total_ngrams += len(tokens) - n + 1

    perplexity = math.exp(
        -total_log_probability / total_ngrams
    )

    return perplexity


# ============================================================
# 7. TRAIN ONE MODEL
# ============================================================

def train_model(training_data, n):

    print("\n" + "=" * 60)

    if n == 1:
        print("TRAINING UNIGRAM MODEL")
    elif n == 2:
        print("TRAINING BIGRAM MODEL")
    elif n == 3:
        print("TRAINING TRIGRAM MODEL")
    elif n == 4:
        print("TRAINING QUADGRAM MODEL")

    print("=" * 60)

    ngram_counts, context_counts, vocabulary = \
        build_ngram_model(training_data, n)

    # Total number of tokens for unigram denominator
    total_tokens = sum(
        ngram_counts.values()
    )

    print("Vocabulary size :", len(vocabulary))
    print("N-gram count    :", len(ngram_counts))

    return {
        "ngram_counts": ngram_counts,
        "context_counts": context_counts,
        "vocabulary": vocabulary,
        "total_tokens": total_tokens
    }


# ============================================================
# 8. DISPLAY SAMPLE PROBABILITIES
# ============================================================

def show_sample_probabilities(models):

    print("\n")
    print("=" * 70)
    print("SAMPLE LAPLACE PROBABILITIES")
    print("=" * 70)

    # --------------------------------------------------------
    # UNIGRAM
    # --------------------------------------------------------

    model = models[1]

    # Change this word if you want to test a Hindi word
    word = next(
        word for word in model["vocabulary"]
        if word not in {"<s>", "</s>", "<UNK>"}
    )

    unigram = (word,)

    probability = laplace_probability(
        unigram,
        model["ngram_counts"],
        model["context_counts"],
        model["vocabulary"],
        model["total_tokens"]
    )

    print("\nUnigram:")
    print("Word:", word)
    print("P(" + word + ") =", probability)

    # --------------------------------------------------------
    # BIGRAM
    # --------------------------------------------------------

    model = models[2]

    # Find an actual bigram from training data
    bigram = next(iter(model["ngram_counts"]))

    probability = laplace_probability(
        bigram,
        model["ngram_counts"],
        model["context_counts"],
        model["vocabulary"],
        model["total_tokens"]
    )

    print("\nBigram:")
    print("Bigram:", bigram)
    print("Probability =", probability)

    # --------------------------------------------------------
    # TRIGRAM
    # --------------------------------------------------------

    model = models[3]

    trigram = next(iter(model["ngram_counts"]))

    probability = laplace_probability(
        trigram,
        model["ngram_counts"],
        model["context_counts"],
        model["vocabulary"],
        model["total_tokens"]
    )

    print("\nTrigram:")
    print("Trigram:", trigram)
    print("Probability =", probability)

    # --------------------------------------------------------
    # QUADGRAM
    # --------------------------------------------------------

    model = models[4]

    quadgram = next(iter(model["ngram_counts"]))

    probability = laplace_probability(
        quadgram,
        model["ngram_counts"],
        model["context_counts"],
        model["vocabulary"],
        model["total_tokens"]
    )

    print("\nQuadgram:")
    print("Quadgram:", quadgram)
    print("Probability =", probability)


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 70)
    print("HINDI N-GRAM LANGUAGE MODEL")
    print("ADD-ONE / LAPLACE SMOOTHING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load 100,000 sentences
    # --------------------------------------------------------

    print("\nReading dataset...")

    sentences = load_sentences(
        FILE_NAME,
        TOTAL_SENTENCES
    )

    print("Sentences loaded:", len(sentences))

    if len(sentences) < TOTAL_SENTENCES:

        print(
            "\nERROR: Dataset contains fewer than 100,000 sentences."
        )

        return

    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.shuffle(sentences)

    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------

    training_data = sentences[
        :TRAIN_SIZE
    ]

    development_data = sentences[
        TRAIN_SIZE:TRAIN_SIZE + DEV_SIZE
    ]

    test_data = sentences[
        TRAIN_SIZE + DEV_SIZE:
        TOTAL_SENTENCES
    ]

    print("\n" + "=" * 70)
    print("DATASET SPLIT")
    print("=" * 70)

    print("Training sentences    :", len(training_data))
    print("Development sentences :", len(development_data))
    print("Test sentences        :", len(test_data))

    # --------------------------------------------------------
    # Create training vocabulary
    # --------------------------------------------------------

    training_vocabulary = set()

    for sentence in training_data:

        for word in sentence:

            training_vocabulary.add(word)

    # Add UNK token
    training_vocabulary.add("<UNK>")

    # --------------------------------------------------------
    # Replace unknown words in development/test
    # --------------------------------------------------------

    development_data = replace_unknown_words(
        development_data,
        training_vocabulary
    )

    test_data = replace_unknown_words(
        test_data,
        training_vocabulary
    )

    # --------------------------------------------------------
    # Replace unknown words in training data
    # --------------------------------------------------------

    training_data = replace_unknown_words(
        training_data,
        training_vocabulary
    )

    # --------------------------------------------------------
    # Train all four models
    # --------------------------------------------------------

    models = {}

    for n in range(1, 5):

        models[n] = train_model(
            training_data,
            n
        )

    # --------------------------------------------------------
    # Show sample probabilities
    # --------------------------------------------------------

    show_sample_probabilities(models)

    # --------------------------------------------------------
    # Calculate perplexity
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PERPLEXITY RESULTS")
    print("=" * 70)

    print(
        "\n{:<15} {:>20} {:>20}".format(
            "Model",
            "Development",
            "Test"
        )
    )

    print("-" * 60)

    for n in range(1, 5):

        model = models[n]

        dev_perplexity = calculate_perplexity(
            development_data,
            n,
            model["ngram_counts"],
            model["context_counts"],
            model["vocabulary"],
            model["total_tokens"]
        )

        test_perplexity = calculate_perplexity(
            test_data,
            n,
            model["ngram_counts"],
            model["context_counts"],
            model["vocabulary"],
            model["total_tokens"]
        )

        if n == 1:
            model_name = "Unigram"
        elif n == 2:
            model_name = "Bigram"
        elif n == 3:
            model_name = "Trigram"
        else:
            model_name = "Quadgram"

        print(
            "{:<15} {:>20.4f} {:>20.4f}".format(
                model_name,
                dev_perplexity,
                test_perplexity
            )
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
