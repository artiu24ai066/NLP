from collections import Counter


def train_wordpiece(word_frequencies, merge_count):
    words = {word: list(word) for word in word_frequencies}
    merges = []
    details = []

    for _ in range(merge_count):
        token_frequencies = Counter()
        pair_frequencies = Counter()
        pair_order = {}
        order = 0

        for word, pieces in words.items():
            frequency = word_frequencies[word]
            for piece in pieces:
                token_frequencies[piece] += frequency
            for pair in zip(pieces, pieces[1:]):
                pair_frequencies[pair] += frequency
                if pair not in pair_order:
                    pair_order[pair] = order
                    order += 1

        if not pair_frequencies:
            break

        def score(pair):
            first, second = pair
            return pair_frequencies[pair] / (
                token_frequencies[first] * token_frequencies[second]
            )

        pair = max(
            pair_frequencies,
            key=lambda candidate: (score(candidate), -pair_order[candidate]),
        )
        merged_piece = "".join(pair)
        details.append((pair, pair_frequencies[pair], score(pair)))
        merges.append(pair)

        for word, pieces in words.items():
            updated = []
            index = 0
            while index < len(pieces):
                if (
                    index + 1 < len(pieces)
                    and (pieces[index], pieces[index + 1]) == pair
                ):
                    updated.append(merged_piece)
                    index += 2
                else:
                    updated.append(pieces[index])
                    index += 1
            words[word] = updated

    return merges, details


def tokenize(word, merges):
    pieces = list(word)
    for first, second in merges:
        merged = []
        index = 0
        while index < len(pieces):
            if (
                index + 1 < len(pieces)
                and (pieces[index], pieces[index + 1]) == (first, second)
            ):
                merged.append(first + second)
                index += 2
            else:
                merged.append(pieces[index])
                index += 1
        pieces = merged
    return [piece if index == 0 else "##" + piece for index, piece in enumerate(pieces)]


def main():
    corpus = [
        "low", "lower", "low", "lower", "lowest",
        "low", "lowest", "low", "lowest", "low",
        "new", "new", "new", "newer", "newer", "new",
    ]
    merges, details = train_wordpiece(Counter(corpus), merge_count=2)
    print("WordPiece score: pair_frequency / (first_piece_frequency * second_piece_frequency)")
    print(f"Word frequencies: {dict(Counter(corpus))}")
    for number, (pair, frequency, score) in enumerate(details, start=1):
        print(
            f"Merge {number}: {pair[0]} + {pair[1]} -> {''.join(pair)} "
            f"(count={frequency}, score={score:.6f})"
        )
    print(f"Tokenization of newest: {' '.join(tokenize('newest', merges))}")
    print("Continuation pieces are prefixed by ##.")


if __name__ == "__main__":
    main()
