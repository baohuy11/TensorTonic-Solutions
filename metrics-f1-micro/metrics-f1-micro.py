def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    tp = 0
    fp = 0
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp: tp += 1
        else:
            fp += 1
            fn += 1
    denominator = (2 * tp ) + fp + fn
    if denominator == 0:
        return 0.0
    return (2 * tp) / denominator