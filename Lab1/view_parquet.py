import pyarrow.parquet as pq
from pathlib import Path

# Get the folder where this Python file is located
base_folder = Path(__file__).parent

# Build the correct path
file = base_folder / "output" / "indiccorp" / "indiccorp_hi_tokenized.parquet"

print("Looking for file:")
print(file)

print("\nFile exists:", file.exists())

pf = pq.ParquetFile(file)

print("\nRows:", pf.metadata.num_rows)
print("Columns:", pf.schema.names)

for batch in pf.iter_batches(batch_size=10):

    sentences = batch.column("tokenized_sentence").to_pylist()

    for i, sentence in enumerate(sentences):
        print("\n-----------------------------")
        print("Sentence", i + 1)
        print("-----------------------------")
        print("Sentence:", sentence)
        print("Words:", sentence.split())

    break
