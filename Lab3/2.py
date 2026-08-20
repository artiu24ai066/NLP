from pathlib import Path
import sys


# FST alphabets. The stem transition copies lowercase input symbols to output.
INPUT_ALPHABET = set("abcdefghijklmnopqrstuvwxyz") | {"<EOS>"}
OUTPUT_ALPHABET = INPUT_ALPHABET | {"+N", "+SG", "+PL", "epsilon"}

# State transitions are represented as (input, output, next state).
STEM_TRANSITIONS = {
	"q_start": ("lowercase", "copy", "q_stem"),
	"q_stem": ("lowercase", "copy", "q_stem"),
	"q_singular_end": ("<EOS>", "+N+SG", "q_accept"),
	"q_plural_end": ("<EOS>", "+N+PL", "q_accept"),
}

# The suffix table is the plural part of the transducer.
PLURAL_TRANSITIONS = (
	{"state": "q_e_insertion", "endings": ("s", "z", "x", "ch", "sh"),
	 "surface_suffix": "es", "stem_change": "none"},
	{"state": "q_y_replacement", "endings": ("y",),
	 "surface_suffix": "ies", "stem_change": "remove_final_y"},
	{"state": "q_s_addition", "endings": ("default",),
	 "surface_suffix": "s", "stem_change": "none"},
)


def read_noun_lexicon():
	corpus_path = Path(__file__).with_name("brown_nouns.txt")
	return {
		line.strip().lower()
		for line in corpus_path.read_text(encoding="utf-8").splitlines()
		if line.strip().isalpha() and line.strip().islower()
	}


def transition_for_root(root):
	"""Return the plural transition selected by the FST table."""
	for transition in PLURAL_TRANSITIONS:
		if transition["state"] == "q_e_insertion" and root.endswith(transition["endings"]):
			return transition
		if transition["state"] == "q_y_replacement" and root.endswith("y"):
			return transition
		if transition["state"] == "q_s_addition":
			return transition
	return None


def plural_surface(root):
	transition = transition_for_root(root)
	if transition["stem_change"] == "remove_final_y":
		return root[:-1] + transition["surface_suffix"]
	return root + transition["surface_suffix"]


def transduce(word, noun_lexicon):
	if not word or any(character not in INPUT_ALPHABET for character in word):
		return "Invalid Word"

	# A matching generated plural takes precedence over a singular reading.
	for root in noun_lexicon:
		if plural_surface(root) == word:
			return f"{root}+N+PL"

	if word in noun_lexicon:
		return f"{word}+N+SG"

	return "Invalid Word"


if __name__ == "__main__":
	input_word = sys.stdin.readline().rstrip("\r\n")
	print(transduce(input_word, read_noun_lexicon()))
