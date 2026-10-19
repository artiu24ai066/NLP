import math
from collections import Counter
from itertools import combinations


def main():
    sentences = {
        "S1": ["cat", "dog", "animal"],
        "S2": ["cat", "cat", "pet"],
        "S3": ["dog", "pet", "animal"],
        "S4": ["cat", "dog", "pet"],
    }
    vocabulary = sorted({word for sentence in sentences.values() for word in sentence})
    document_frequency = {
        word: sum(word in sentence for sentence in sentences.values())
        for word in vocabulary
    }
    idf = {
        word: math.log(len(sentences) / document_frequency[word])
        for word in vocabulary
    }
    counts = {name: Counter(sentence) for name, sentence in sentences.items()}
    vectors = {}

    print("Assumption: the 12 tokens form four consecutive 3-token sentences.")
    print("TF is the raw within-sentence count; IDF(w) = ln(N / df(w)).")
    print("Word      df       IDF")
    for word in vocabulary:
        print(f"{word:8}  {document_frequency[word]:>2}   {idf[word]:.6f}")

    print(f"\nVocabulary order: {vocabulary}")
    for name, sentence_counts in counts.items():
        vector = [sentence_counts[word] * idf[word] for word in vocabulary]
        vectors[name] = vector
        print(f"{name} TF-IDF vector: {[round(value, 6) for value in vector]}")

    similarities = []
    print("\nCosine similarities:")
    for left, right in combinations(sentences, 2):
        first, second = vectors[left], vectors[right]
        dot = sum(a * b for a, b in zip(first, second))
        norm = math.sqrt(sum(a * a for a in first) * sum(b * b for b in second))
        similarity = dot / norm if norm else 0.0
        similarities.append((similarity, left, right))
        print(f"cosine({left}, {right}) = {dot:.6f}/{norm:.6f} = {similarity:.6f}")

    similarity, left, right = max(similarities)
    print(f"\nClosest sentences: {left} and {right} (cosine={similarity:.6f})")


if __name__ == "__main__":
    main()
