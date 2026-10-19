from collections import Counter


def main():
    sentences = [
        "I like NLP",
        "I like AI",
        "You like NLP",
        "You love AI",
        "We love NLP",
    ]
    discount = 0.5
    bigram_counts = Counter()

    for sentence in sentences:
        words = sentence.split()
        bigram_counts.update(zip(words, words[1:]))

    context_counts = Counter()
    continuation_counts = Counter()
    for (history, word), count in bigram_counts.items():
        context_counts[history] += count
        continuation_counts[word] += 1

    distinct_bigram_types = len(bigram_counts)
    print("Interpolated Kneser-Ney smoothing with d=0.5.")
    print(f"Distinct bigram types: {distinct_bigram_types}.")
    for word in ("AI", "NLP"):
        probability = continuation_counts[word] / distinct_bigram_types
        print(
            f"P_cont({word}) = {continuation_counts[word]}/"
            f"{distinct_bigram_types} = {probability:.6f}"
        )

    for history, word in (("love", "AI"), ("like", "NLP")):
        count = bigram_counts[(history, word)]
        history_count = context_counts[history]
        distinct_successors = sum(
            previous == history for previous, _ in bigram_counts
        )
        continuation_probability = (
            continuation_counts[word] / distinct_bigram_types
        )
        discounted_bigram = max(count - discount, 0) / history_count
        backoff_weight = discount * distinct_successors / history_count
        probability = discounted_bigram + backoff_weight * continuation_probability
        print(
            f"\nP_KN({word} | {history}) = max({count}-{discount}, 0)/"
            f"{history_count} + ({discount}*{distinct_successors}/"
            f"{history_count})*{continuation_counts[word]}/"
            f"{distinct_bigram_types}"
        )
        print(
            f"= {discounted_bigram:.6f} + {backoff_weight:.6f}*"
            f"{continuation_probability:.6f} = {probability:.6f}"
        )


if __name__ == "__main__":
    main()
