# Lab 7: Hindi POS Tagging

This lab implements the requested Naive Bayes classifier and three HMM decoding
variants using the provided word/tag corpus.

## Run

From the repository root, run:

```powershell
.\venv\Scripts\python.exe .\Lab7\hindi_pos_tagger.py
```

The corpus is UTF-8, tab-separated (`word<TAB>tag`), with blank lines marking
sentence boundaries. The default command trains on `Hindi-POS-Data/hindi-train-pos.txt`,
evaluates on the supplied dev and test files, and recreates the `results` directory
contents. To use alternate paths:

```powershell
.\venv\Scripts\python.exe .\Lab7\hindi_pos_tagger.py --data-dir .\Lab7\Hindi-POS-Data --output-dir .\Lab7\results
```

The implementation uses Python's standard library; no packages need to be installed.

## Models

- **Naive Bayes:** Factorized categorical features are word length (`MORE` iff
  length > 4, otherwise `LESS`), actual word, previous word, next word, and
  previous tag. Add-0.1 smoothing is used. At evaluation time the previous-tag
  feature is the preceding model prediction, so tagging is greedy and does not
  use the gold tags from dev/test.
- **Forward Viterbi:** Standard left-to-right max-product decoding with BOS/EOS
  transitions.
- **Backward Viterbi:** The same sequence objective, evaluated right-to-left
  from EOS.
- **Forward-backward Viterbi:** Combines max-product forward and backward scores
  for each tag at each position, then independently selects the best tag for
  each token. Unlike the first two HMM decoders, this tokenwise path is not
  guaranteed to be the single globally best sequence.

HMM transition and emission distributions are estimated from the training split
with add-0.1 smoothing. Words not seen during training use the smoothed unknown
emission probability.

## Outputs

`results/REPORT.md` compares accuracy, macro F1, and weighted F1 for dev and
test. The split-specific CSV files contain precision, recall, F1, and gold
support for every tag and model. Split-specific SVGs plot per-tag F1 for all
four models.
