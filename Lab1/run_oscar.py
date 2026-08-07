"""
Task 2 – Process Hindi subset of OSCAR-2301 monolingual corpus.

Dataset: https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
Language: Hindi (config code 'hi')

Note: OSCAR is a gated dataset. Accept the terms on HuggingFace and set HF_TOKEN.

Set HF_TOKEN before running:
    set HF_TOKEN=your_huggingface_token   (Windows)
    export HF_TOKEN=your_huggingface_token  (Linux/Mac)
"""

from pathlib import Path

from corpus_processor import process_corpus

OUTPUT_DIR = Path(__file__).parent / "output" / "oscar"

# Set to an integer (e.g. 1000) for a quick test run; None processes full corpus
MAX_DOCUMENTS = 10000  # 10 lakh documents for reasonable sample

if __name__ == "__main__":
    process_corpus(
        dataset_name="oscar_te",
        output_dir=OUTPUT_DIR,
        text_field="content",
        load_kwargs={
            "path": "oscar-corpus/OSCAR-2301",
            "name": "hi",  # language config (file: data/hi.txt)
            "streaming": True,
            "split": "train",
        },
        max_documents=MAX_DOCUMENTS,
    )
