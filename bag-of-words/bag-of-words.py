import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    bow_vector = np.zeros(len(vocab), dtype=int)
    vocab_map = {word: i for i, word in enumerate(vocab)}
    for token in tokens:
        if token in vocab_map:
            idx = vocab_map[token]
            bow_vector[idx] += 1
    return bow_vector
