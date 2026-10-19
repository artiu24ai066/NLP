import math


def main():
    features = ("free", "win", "offer")
    examples = [
        ((1, 1, 1), "Spam"),
        ((1, 1, 0), "Spam"),
        ((1, 0, 1), "Spam"),
        ((0, 1, 1), "Spam"),
        ((0, 0, 1), "NotSpam"),
        ((0, 1, 0), "NotSpam"),
        ((1, 0, 0), "NotSpam"),
    ]
    email = (0, 0, 0)
    class_names = ("Spam", "NotSpam")
    class_examples = {
        label: [values for values, actual in examples if actual == label]
        for label in class_names
    }
    joint_scores = {}

    print("Bernoulli Naive Bayes; use the unsmoothed maximum-likelihood estimates.")
    print(f"Class priors: P(Spam)=4/7, P(NotSpam)=3/7; E8={email}.")
    for label in class_names:
        rows = class_examples[label]
        probabilities = []
        for index, feature in enumerate(features):
            present = sum(row[index] for row in rows)
            probability = (len(rows) - present) / len(rows)
            probabilities.append(probability)
            print(f"P({feature}=0 | {label}) = {probability:.6f}")

        prior = len(rows) / len(examples)
        likelihood = math.prod(probabilities)
        joint_scores[label] = prior * likelihood
        print(
            f"P({label}) * P(E8 | {label}) = {prior:.6f} * "
            f"{likelihood:.6f} = {joint_scores[label]:.6f}"
        )

    normalizer = sum(joint_scores.values())
    for label in class_names:
        posterior = joint_scores[label] / normalizer
        print(f"P({label} | E8) = {posterior:.6f}")
    print(f"Prediction: {max(class_names, key=joint_scores.get)}")


if __name__ == "__main__":
    main()
