"""
Generate proper statistics from existing parquet file.
Use this when the original processing was interrupted before final stats were written.
"""

import json
from pathlib import Path
import pyarrow.parquet as pq

def generate_stats_from_parquet(parquet_path, stats_path):
    """Generate corpus statistics from tokenized parquet file."""
    
    print(f"Reading parquet file: {parquet_path}")
    
    # Read the parquet file
    table = pq.read_table(parquet_path)
    sentences = table.column('tokenized_sentence').to_pylist()
    
    print(f"Found {len(sentences)} sentences in parquet file")
    
    # Initialize statistics
    stats = {
        "total_sentences": 0,
        "total_words": 0,
        "total_characters": 0,
        "total_word_chars": 0,
        "token_types": set()
    }
    
    # Process each sentence
    for i, sentence in enumerate(sentences):
        if not sentence or not sentence.strip():
            continue
            
        # Split sentence into tokens
        tokens = sentence.strip().split()
        if not tokens:
            continue
            
        stats["total_sentences"] += 1
        stats["total_words"] += len(tokens)
        
        # Count characters in original sentence (approximation)
        stats["total_characters"] += len(sentence)
        
        # Process each token
        for token in tokens:
            stats["total_word_chars"] += len(token)
            stats["token_types"].add(token)
        
        # Progress indicator
        if (i + 1) % 100000 == 0:
            print(f"  Processed {i + 1:,} sentences...")
    
    # Calculate final statistics
    avg_sent_len = stats["total_words"] / stats["total_sentences"] if stats["total_sentences"] else 0.0
    avg_word_len = stats["total_word_chars"] / stats["total_words"] if stats["total_words"] else 0.0
    type_token_ratio = len(stats["token_types"]) / stats["total_words"] if stats["total_words"] else 0.0
    
    # Final report
    report = {
        "total_sentences": stats["total_sentences"],
        "total_words": stats["total_words"],
        "total_characters": stats["total_characters"],
        "average_sentence_length": round(avg_sent_len, 4),
        "average_word_length": round(avg_word_len, 4),
        "unique_token_types": len(stats["token_types"]),
        "type_token_ratio": round(type_token_ratio, 6),
    }
    
    # Save statistics
    stats_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    
    print(f"\n=== IndicCorpV2 Telugu Corpus Statistics (Generated from Parquet) ===")
    for key, value in report.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")
    
    print(f"\nStatistics saved to: {stats_path}")
    return report

if __name__ == "__main__":
    parquet_path = Path("output/indiccorp/indiccorp_te_tokenized.parquet")
    stats_path = Path("output/indiccorp/indiccorp_te_statistics.json")
    
    if not parquet_path.exists():
        print(f"Error: Parquet file not found: {parquet_path}")
        exit(1)
    
    generate_stats_from_parquet(parquet_path, stats_path)
    