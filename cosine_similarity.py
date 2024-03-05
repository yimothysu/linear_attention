"""
Find cosine similarity between two word embeddings.
"""

import numpy as np
from tqdm import tqdm

import csv
import json

embedding_file = "models/embeddings.txt"

# Load word embeddings
words, embedding_matrix = [], []
with open(embedding_file, "r") as f:
    reader = csv.reader(f, delimiter="\t")
    for row in reader:
        if len(row) != 2:
            print(f"Warning: Invalid row: {row}")
            continue
        word, embedding_str = row
        embedding = json.loads(embedding_str)

        words.append(word)
        embedding_matrix.append(embedding)
embedding_matrix = np.array(embedding_matrix)
print(f"Loaded {len(words)} word embeddings from {embedding_file}.")
print()


def cosine_similarity(v1, v2):
    v1 = np.array(list(v1))
    v2 = np.array(list(v2))

    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    return dot_product / (magnitude_v1 * magnitude_v2)


def cosine_similarity_matrix(vectors):
    # Normalize the vectors to unit vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms
    # Compute the cosine similarity matrix
    similarity_matrix = np.dot(normalized_vectors, normalized_vectors.T)
    return similarity_matrix


def find_highest_similarity(words, similarity_matrix):
    # Zero out the diagonal to ignore self-similarity
    np.fill_diagonal(similarity_matrix, 0)
    # Find the index of the highest similarity
    max_index = np.argmax(similarity_matrix)
    # Convert the 1D index to 2D coordinates
    i, j = np.unravel_index(max_index, similarity_matrix.shape)
    return words[i], words[j], similarity_matrix[i, j]


# def find_top_n_similarities(words, similarity_matrix, n):
#     n *= 2
#     # Ensure the diagonal is zero to ignore self-similarities
#     np.fill_diagonal(similarity_matrix, 0)
#     # Flatten the matrix and get the indices of the top N similarities
#     flat_indices = np.argpartition(similarity_matrix.flatten(), -n)[-n:]
#     # Convert flat indices to 2D indices to get pairs of words
#     row_indices, col_indices = np.unravel_index(flat_indices, similarity_matrix.shape)

#     # Collect the top N pairs with their similarity scores
#     top_pairs = []
#     for row, col in zip(row_indices, col_indices):
#         if (
#             row < col
#         ):  # This check ensures we don't include (word1, word2) and (word2, word1) both
#             pair = (words[row], words[col], similarity_matrix[row, col])
#             top_pairs.append(pair)

#     # Sort the pairs by similarity score in descending order
#     top_pairs.sort(key=lambda x: x[2], reverse=True)

#     return top_pairs[:n]  # Return the top N pairs


def find_top_n_similarities(
    words,
    similarity_matrix,
    n,
    stopwords={
        "the",
        "is",
        "at",
        "which",
        "on",
        "of",
        "and",
        "a",
        "my",
        "in",
        "his",
        "she",
        "he",
        "our",
        "their",
        "your",
        "her",
        "have",
        "we",
        "i",
        "be",
        "am",
        "are",
        "thou",
        "thy",
        "whose",
        "has",
        "had",
        "too",
        "so",
        "this",
        "to",
        "me",
        "us",
        "was",
        "tis",
        "but",
        "or",
        "you",
        "they",
        "should",
        "what",
        "that",
        "were",
        "was",
        "no",
        "him",
        "them",
        "did",
        "can",
        "make",
        "makes",
        "some",
        "many",
        "'ll",
        "for",
        "been",
        "'",
        ".",
        "?",
        "!",
        "-",
        "'s",
        ":",
        ",",
    },
):
    # Ensure the diagonal is filled with low values to ignore self-similarities
    np.fill_diagonal(similarity_matrix, -1)
    # Flatten the matrix and get the indices of all elements, sorted by similarity descending
    flat_indices = np.argsort(similarity_matrix.flatten())[::-1]
    # Convert flat indices to 2D indices to get pairs of words
    row_indices, col_indices = np.unravel_index(flat_indices, similarity_matrix.shape)

    # Collect the top N pairs with their similarity scores, ignoring stopwords
    top_pairs = []
    for row, col in zip(row_indices, col_indices):
        if len(top_pairs) >= n:
            break  # Stop when we have enough pairs
        if row < col:  # Ensure we don't include duplicate pairs
            if words[row] not in stopwords and words[col] not in stopwords:
                pair = (words[row], words[col], similarity_matrix[row, col])
                top_pairs.append(pair)

    # No need to sort since we are already iterating in descending order of similarity
    return top_pairs


def find_bottom_n_similarities(words, similarity_matrix, n):
    n *= 2
    # Ensure the diagonal is filled with high values to ignore self-similarities
    np.fill_diagonal(similarity_matrix, 1)
    # Flatten the matrix and get the indices of the bottom N similarities
    flat_indices = np.argpartition(similarity_matrix.flatten(), n)[:n]
    # Convert flat indices to 2D indices to get pairs of words
    row_indices, col_indices = np.unravel_index(flat_indices, similarity_matrix.shape)

    # Collect the bottom N pairs with their similarity scores
    bottom_pairs = []
    for row, col in zip(row_indices, col_indices):
        if (
            row < col
        ):  # This check ensures we don't include (word1, word2) and (word2, word1) both
            pair = (words[row], words[col], similarity_matrix[row, col])
            bottom_pairs.append(pair)

    # Sort the pairs by similarity score in ascending order
    bottom_pairs.sort(key=lambda x: x[2])

    return bottom_pairs[:n]  # Return the bottom N pairs


csm = cosine_similarity_matrix(embedding_matrix)
top_n = find_top_n_similarities(words, csm, 10)
bottom_n = find_bottom_n_similarities(words, csm, 10)

print("Top 10 most similar word pairs")
print("---")
for pair in top_n:
    print(f"{pair[0]}, {pair[1]}: {pair[2]}")
print()
print("Top 10 least similar word pairs")
print("---")
for pair in bottom_n:
    print(f"{pair[0]}, {pair[1]}: {pair[2]}")

while True:
    word_1 = input("Word 1: ")
    word_2 = input("Word 2: ")

    if word_1 not in words:
        print(f"Word {word_1} not in vocabulary.")
        continue
    if word_2 not in words:
        print(f"Word {word_2} not in vocabulary.")
        continue

    embedding_1 = embedding_matrix[words.index(word_1)]
    embedding_2 = embedding_matrix[words.index(word_2)]

    print("Cosine Similarity:", cosine_similarity(embedding_1, embedding_2))
    print()
