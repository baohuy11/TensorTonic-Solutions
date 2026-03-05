import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    tokenized_docs = [doc.lower().split() for doc in documents]
    vocab = sorted(list(set(word for doc in tokenized_docs for word in doc)))
    word_to_idx = {word: i for i, word in enumerate(vocab)}

    num_docs = len(documents)
    num_words = len(vocab)

    idf = {}
    for word in vocab:
        df_t = sum(1 for doc in tokenized_docs if word in doc)
        idf[word] = math.log(num_docs / df_t)

    tfidf_matrix = []
    for doc in tokenized_docs:
        cnts = Counter(doc)
        total_terms = len(doc)

        vector = [0.0] * num_words
        for word, cnt in cnts.items():
            if word in word_to_idx:
                tf = cnt / total_terms
                vector[word_to_idx[word]] = tf * idf[word]
        tfidf_matrix.append(vector)
    return tfidf_matrix, vocab