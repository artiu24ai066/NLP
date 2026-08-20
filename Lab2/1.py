import json
import math

with open("text_segmentation_dataset.json", "r") as file:
    data = json.load(file)

word_counts = data["word_counts"]
test_cases = data["test_cases"]
total_words = data["metadata"]["total_corpus_words"]


# 1. GREEDY LONGEST WORD MATCHING
def greedy_segmentation(text, vocabulary):
    result = []
    i = 0

    # Maximum word length in vocabulary
    max_len = max(len(word) for word in vocabulary)

    while i < len(text):
        best_word = None

        # Try longest word first
        for length in range(min(max_len, len(text) - i), 0, -1):
            candidate = text[i:i + length]
            if candidate in vocabulary:
                best_word = candidate
                break

        # If no word found, take one character
        if best_word is None:
            best_word = text[i]

        result.append(best_word)
        i += len(best_word)
    return result



# 2. DYNAMIC PROGRAMMING
# Maximum log probability segmentation
def dp_segmentation(text, word_counts, total_words):
    n = len(text)

    # dp[i] = maximum log probability for first i characters
    dp = [-float("inf")] * (n + 1)

    # Store the word used to reach each position
    previous_word = [None] * (n + 1)
    dp[0] = 0

    # Maximum word length
    max_len = max(len(word) for word in word_counts)

    for i in range(1, n + 1):

        # Try every possible word ending at position i
        for length in range(1, min(max_len, i) + 1):
            start = i - length
            word = text[start:i]

            if word in word_counts:
                probability = word_counts[word] / total_words
                log_probability = math.log(probability)
                candidate_score = dp[start] + log_probability

                if candidate_score > dp[i]:
                    dp[i] = candidate_score
                    previous_word[i] = word

    # Reconstruct segmentation
    result = []
    position = n

    while position > 0:
        word = previous_word[position]

        # If no valid segmentation exists
        if word is None:
            result.append(text[position - 1])
            position -= 1
        else:
            result.append(word)
            position -= len(word)

    result.reverse()
    return result


# 3. ACCURACY
def calculate_accuracy(predicted, actual):
    correct = 0

    # Compare corresponding words
    min_length = min(len(predicted), len(actual))

    for i in range(min_length):
        if predicted[i] == actual[i]:
            correct += 1

    if len(actual) == 0:
        return 0
    
    return correct / len(actual)


# 4. EDIT DISTANCE
def edit_distance(predicted, actual):

    # Convert list of words into strings
    predicted = " ".join(predicted)
    actual = " ".join(actual)

    m = len(predicted)
    n = len(actual)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    # DP
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if predicted[i - 1] == actual[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                insertion = dp[i][j - 1]
                deletion = dp[i - 1][j]
                substitution = dp[i - 1][j - 1]
                dp[i][j] = 1 + min(
                    insertion,
                    deletion,
                    substitution
                )
    return dp[m][n]


# MAIN PROGRAM
vocabulary = set(word_counts.keys())
greedy_total_accuracy = 0
dp_total_accuracy = 0
greedy_total_edit_distance = 0
dp_total_edit_distance = 0

for case in test_cases:
    text = case["input"]
    ground_truth = case["ground_truth"].split()

    # -----------------------------
    # Greedy
    # -----------------------------

    greedy_result = greedy_segmentation(
        text,
        vocabulary
    )

    greedy_accuracy = calculate_accuracy(
        greedy_result,
        ground_truth
    )

    greedy_edit = edit_distance(
        greedy_result,
        ground_truth
    )


    # -----------------------------
    # Dynamic Programming
    # -----------------------------

    dp_result = dp_segmentation(
        text,
        word_counts,
        total_words
    )

    dp_accuracy = calculate_accuracy(
        dp_result,
        ground_truth
    )

    dp_edit = edit_distance(
        dp_result,
        ground_truth
    )


    # Add to totals
    greedy_total_accuracy += greedy_accuracy
    dp_total_accuracy += dp_accuracy

    greedy_total_edit_distance += greedy_edit
    dp_total_edit_distance += dp_edit


# ---------------------------------------------------------
# FINAL RESULTS
# ---------------------------------------------------------

number_of_cases = len(test_cases)

greedy_avg_accuracy = greedy_total_accuracy / number_of_cases
dp_avg_accuracy = dp_total_accuracy / number_of_cases

greedy_avg_edit = greedy_total_edit_distance / number_of_cases
dp_avg_edit = dp_total_edit_distance / number_of_cases


print("TEXT SEGMENTATION RESULTS")

print("\nGreedy Based Approach (Taking the longest word first)")
print("------------------------------------------")
print("Accuracy      :", round(greedy_avg_accuracy, 4))
print("Edit Distance :", round(greedy_avg_edit, 4))


print("\nDynamic Programming Approach (Maximum Log Probability Segmentation)")
print("------------------------------------------")
print("Accuracy      :", round(dp_avg_accuracy, 4))
print("Edit Distance :", round(dp_avg_edit, 4))

print("\n")

print("COMPARISON")

if dp_avg_accuracy > greedy_avg_accuracy:
    print("DP has better Accuracy.")

elif dp_avg_accuracy < greedy_avg_accuracy:
    print("Greedy has better Accuracy.")

else:
    print("Both have the same Accuracy.")


if dp_avg_edit < greedy_avg_edit:
    print("DP has lower Edit Distance.")

elif dp_avg_edit > greedy_avg_edit:
    print("Greedy has lower Edit Distance.")

else:
    print("Both have the same Edit Distance.")


print("\n")
print("Testing with user input")

user_text = input("Enter a string without spaces: ")

# Greedy segmentation
user_greedy = greedy_segmentation(
    user_text,
    vocabulary
)

# DP segmentation
user_dp = dp_segmentation(
    user_text,
    word_counts,
    total_words
)

print("\nOriginal String:")
print(user_text)

print("\nGreedy Segmentation:")
print(" ".join(user_greedy))

print("\nDynamic Programming Segmentation:")
print(" ".join(user_dp))
