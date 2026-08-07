"""
Download Hindi corpora, tokenize with regex tokenizers, save as compressed parquet, and compute corpus-level statistics.
"""

import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from tokenizer import tokenize_paragraph


class CorpusStats:
    """Running counters for corpus statistics (updated incrementally)."""

    def __init__(self) -> None:
        self.num_sentences = 0
        self.num_words = 0
        self.num_characters = 0  # characters in original (pre-tokenization) text
        self.total_word_chars = 0  # sum of len(token) for every word token
        self.token_types: set[str] = set()

    def update(self, raw_text: str, tokenized_sentences: list[list[str]]) -> None:
        self.num_characters += len(raw_text)
        for sent_tokens in tokenized_sentences:
            if not sent_tokens:
                continue
            self.num_sentences += 1
            self.num_words += len(sent_tokens)
            for tok in sent_tokens:
                self.total_word_chars += len(tok)
                self.token_types.add(tok)

    def report(self) -> dict:
        avg_sent_len = (
            self.num_words / self.num_sentences if self.num_sentences else 0.0
        )
        avg_word_len = (
            self.total_word_chars / self.num_words if self.num_words else 0.0
        )
        # TTR = unique token types / total tokens  (assignment definition)
        type_token_ratio = (
            len(self.token_types) / self.num_words if self.num_words else 0.0
        )

        return {
            "total_sentences": self.num_sentences,
            "total_words": self.num_words,
            "total_characters": self.num_characters,
            "average_sentence_length": round(avg_sent_len, 4),
            "average_word_length": round(avg_word_len, 4),
            "unique_token_types": len(self.token_types),
            "type_token_ratio": round(type_token_ratio, 6),
        }


def _extract_text(record: dict, text_field: str) -> str:
    """Return plain text from a dataset row."""
    # Try the configured field, then common alternatives
    for field in (text_field, "text", "content", "sentence"):
        if field in record and record[field]:
            return str(record[field]).strip()
    return ""


def process_corpus(
    dataset_name: str,
    output_dir: Path,
    text_field: str = "text",
    load_kwargs: dict | None = None,
    max_documents: int | None = None,
    batch_size: int = 5000,
) -> dict:
    """
    Stream a HuggingFace dataset, tokenize each document, and save results.

    Parameters
    ----------
    dataset_name : str
        Label used in output filenames (e.g. 'indiccorp_hi', 'oscar_hi').
    output_dir : Path
        Directory for parquet output and statistics JSON.
    text_field : str
        Column name containing raw text in the dataset.
    load_kwargs : dict
        Arguments forwarded to ``datasets.load_dataset``.
    max_documents : int | None
        Stop after this many documents (useful for testing; None = all data).
    batch_size : int
        Number of tokenized sentences buffered before each parquet write.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{dataset_name}_tokenized.parquet"
    stats_path = output_dir / f"{dataset_name}_statistics.json"

    load_kwargs = load_kwargs or {}
    load_kwargs.setdefault("streaming", True)
    # Do not default split to "train" — IndicCorpV2 uses language names as splits

    # Authenticate via HF_TOKEN environment variable if set
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        load_kwargs["token"] = hf_token

    print(f"Loading dataset with args: {load_kwargs}")
    dataset = load_dataset(**load_kwargs)

    stats = CorpusStats()
    sentence_buffer: list[str] = []
    schema = pa.schema([("tokenized_sentence", pa.string())])

    writer = pq.ParquetWriter(
        parquet_path,
        schema=schema,
        compression="snappy",  # compressed parquet as required
    )

    doc_count = 0
    try:
        for record in dataset:
            raw = _extract_text(record, text_field)
            if not raw:
                continue

            tokenized = tokenize_paragraph(raw)
            stats.update(raw, tokenized)

            # One line per sentence: tokens joined by spaces
            for sent_tokens in tokenized:
                if sent_tokens:
                    sentence_buffer.append(" ".join(sent_tokens))

            doc_count += 1

            if len(sentence_buffer) >= batch_size:
                table = pa.Table.from_pydict(
                    {"tokenized_sentence": sentence_buffer}, schema=schema
                )
                writer.write_table(table)
                sentence_buffer.clear()
                print(f"  Processed {doc_count} documents …")

            if max_documents and doc_count >= max_documents:
                break

        # Flush remaining sentences
        if sentence_buffer:
            table = pa.Table.from_pydict(
                {"tokenized_sentence": sentence_buffer}, schema=schema
            )
            writer.write_table(table)

    finally:
        writer.close()

    report = stats.report()
    stats_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== {dataset_name} Corpus Statistics ===")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print(f"\nTokenized data  → {parquet_path}")
    print(f"Statistics      → {stats_path}")

    return report
