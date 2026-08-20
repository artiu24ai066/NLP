def is_valid_word(word):
	"""Return True when word is accepted by the simplified-English DFA."""
	start_state = 0
	accepting_state = 1
	state = start_state

	for position, character in enumerate(word):
		if "a" <= character <= "z":
			if position == 0:
				state = accepting_state
		else:
			return False

	return state == accepting_state


word = input().strip("\n")
print("Accepted" if is_valid_word(word) else "Not Accepted")
