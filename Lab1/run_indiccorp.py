"""
Task 1 – Process Hindi subset of IndicCorpV2.

Dataset: https://huggingface.co/datasets/ai4bharat/IndicCorpV2
Language: Hindi (hin_Deva)

Set HF_TOKEN before running:
    set HF_TOKEN=your_huggingface_token   (Windows)
    export HF_TOKEN=your_huggingface_token  (Linux/Mac)
"""

from pathlib import Path

from corpus_processor import process_corpus

OUTPUT_DIR = Path(__file__).parent / "output" / "indiccorp"

# Set to an integer (e.g. 1000) for a quick test run; None processes full corpus
MAX_DOCUMENTS = 100000  # Process what you already did to get proper stats

if __name__ == "__main__":
    process_corpus(
        dataset_name="indiccorp_hi",
        output_dir=OUTPUT_DIR,
        text_field="text",
        load_kwargs={
            "path": "ai4bharat/IndicCorpV2",
            "name": "indiccorp_v2",
            "split": "hin_Deva",  # language split (file: data/hi.txt)
            "streaming": True,
        },
        max_documents=MAX_DOCUMENTS,
    )
