# Hindi POS Tagging

This lab uses the Assignment 7 Hindi POS train/dev/test splits in `Hindi-POS-Data`.
The Lab9 copies are identical to the Lab7 files.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

From the `Lab9` directory, run the three scripts in order:

```powershell
python 1.py
python 2.py
python 3.py
```

The comparison script reuses the saved MEMM and CRF predictions instead of
retraining them.
The first two scripts can also be run on their own when only one model is needed:

```powershell
python 1.py
python 2.py
```

`1.py` trains the MEMM using NLTK's `MaxentClassifier` (the NLTK maximum-entropy
approach used by [nltk-maxent-pos-tagger](https://github.com/arne-cl/nltk-maxent-pos-tagger)).
It uses NLTK's GIS optimizer because the original project's MEGAM optimizer is
an external legacy dependency, and applies the reference project's rare-feature
cutoff of 5.
`2.py` trains a CRF using the
[sklearn-crfsuite tutorial](https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html)
approach and LBFGS.

`3.py` compares the saved MEMM and CRF predictions with Lab7 Naive Bayes and
HMM on the same dev/test splits. It writes overall summaries, per-tag CSV
metrics, word-level predictions, and grouped per-tag F1 SVG charts to `results`.
The default training limits are 5 GIS iterations for MEMM and 30 LBFGS
iterations for CRF. They can be adjusted with `--max-iterations` on `1.py`
and `2.py`.

All models use the same required lexical features: prefixes and suffixes of
lengths 1–4 (using `NULL` when the word is too short), previous/current/next
word, and the `LESS`/`MORE` length feature. The MEMM additionally conditions on
the preceding tag, as required for a Markov model.
