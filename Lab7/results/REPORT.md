# Hindi POS tagger results

| Split | Model | Accuracy | Macro F1 | Weighted F1 | Tokens |
|---|---|---:|---:|---:|---:|
| dev | Naive Bayes | 0.9247 | 0.8331 | 0.9249 | 40762 |
| dev | Forward Viterbi | 0.9305 | 0.9071 | 0.9302 | 40762 |
| dev | Backward Viterbi | 0.9305 | 0.9071 | 0.9302 | 40762 |
| dev | Forward-backward Viterbi | 0.9305 | 0.9071 | 0.9302 | 40762 |
| test | Naive Bayes | 0.9257 | 0.8444 | 0.9259 | 41415 |
| test | Forward Viterbi | 0.9285 | 0.8675 | 0.9283 | 41415 |
| test | Backward Viterbi | 0.9285 | 0.8675 | 0.9283 | 41415 |
| test | Forward-backward Viterbi | 0.9285 | 0.8675 | 0.9283 | 41415 |

Word-level predictions are saved as TSV files, one file per model and split, with each word, gold tag, and predicted tag on a row. Each split's CSV has precision, recall, F1, and support for every tag and model. The SVG chart compares per-tag F1 scores.

Naive Bayes uses the five requested features. It uses the preceding gold tag while training and its own preceding prediction while testing. HMM probabilities use add-0.1 smoothing. Forward and backward Viterbi find the best sequence from opposite directions. Forward-backward combines the best left and right sequence scores at each word, then chooses the best tag for that word.
