import numpy as np
def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    prob_distributions = np.asarray(prob_distributions)
    actual_tokens = np.asarray(actual_tokens)
    
    n = len(actual_tokens)
    
    target_probs = prob_distributions[np.arange(n), actual_tokens]
    
    avg_nll = -np.mean(np.log(target_probs + 1e-10))
    res_perplexity = np.exp(avg_nll)
    
    return res_perplexity