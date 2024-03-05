"""
Find most common bigram in dataset.
"""

from data import read_txts
from tokenizer import tokenize

texts = read_txts("data/")
bigram_counts = {}
for text in texts:
    text_tokenized = tokenize(text)
    for bigram in zip(text_tokenized[:-1], text_tokenized[1:]):
        bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

bigram_counts_sorted = sorted(bigram_counts, key=bigram_counts.get)
print(bigram_counts_sorted[-30:])
