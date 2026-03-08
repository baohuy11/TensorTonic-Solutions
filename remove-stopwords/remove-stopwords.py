def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    stop_set = set(stopwords)
    filtered_tokens = [word for word in tokens if word not in stop_set]
    return filtered_tokens