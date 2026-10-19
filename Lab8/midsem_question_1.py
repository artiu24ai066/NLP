def plural_form(stem):
    folded = stem.casefold()
    if folded.endswith("z") and not folded.endswith("zz"):
        return stem + "zes"
    if folded.endswith(("s", "x", "z", "ch", "sh")):
        return stem + "es"
    return stem + "s"


def analyze_noun(surface):
    analyses = [f"{surface}+N+SG"]
    possible_stems = set()
    if surface.casefold().endswith("s"):
        possible_stems.add(surface[:-1])
    if surface.casefold().endswith("es"):
        possible_stems.add(surface[:-2])

    for stem in sorted(possible_stems):
        if stem and plural_form(stem).casefold() == surface.casefold():
            analyses.append(f"{stem}+N+PL")
    return analyses


def main():
    print("English noun-number finite-state transducer")
    print("States: q0 (start), qR, qS, qX, qZ, qZZ, qC, qCH, qSH, qF (final)")
    print("Stem characters are copied while the state tracks the ending.")
    print("Transition table (other means any non-special stem character):")
    transitions = [
        ("Any stem state", "s", "qS", "s"),
        ("Any stem state", "x", "qX", "x"),
        ("Any stem state", "z", "qZ", "z"),
        ("qZ", "z", "qZZ", "z"),
        ("qZZ", "z", "qZZ", "z"),
        ("Any stem state", "c", "qC", "c"),
        ("qS", "h", "qSH", "h"),
        ("qC", "h", "qCH", "h"),
        ("Any other stem transition", "other", "qR", "copy"),
        ("Any stem-ending state", "+N+SG", "qF", "epsilon"),
        ("qR or qC", "+N+PL", "qF", "s"),
        ("qS, qX, qCH, or qSH", "+N+PL", "qF", "es"),
        ("qZ", "+N+PL", "qF", "zes (double z + es)"),
        ("qZZ", "+N+PL", "qF", "es"),
    ]
    for state, symbol, next_state, output in transitions:
        print(f"  {state:34} {symbol:12} -> {next_state:4} / {output}")

    print("\nGenerated forms:")
    for stem in ("cat", "bus", "box", "quiz", "church", "dish", "buzz"):
        print(f"  {stem}+N+SG -> {stem}; {stem}+N+PL -> {plural_form(stem)}")
    print("\nAnalyses:")
    for surface in ("cats", "buses", "boxes", "quizzes", "churches", "dishes"):
        print(f"  {surface}: {', '.join(analyze_noun(surface))}")


if __name__ == "__main__":
    main()
