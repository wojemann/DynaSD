"""Statistical helpers."""

import numpy as np


def cohens_d(group1, group2):
    """Cohen's d effect size between two 1D samples.

    Returns the standardized mean difference using a pooled standard
    deviation (sample-sd, ddof=1).
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    return (mean1 - mean2) / pooled_std
