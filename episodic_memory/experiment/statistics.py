import numpy as np
from scipy.stats import norm


def get_binomial_mean_confidence_interval(X, alpha=0.05):
    """
    Get the mean and confidence interval delta for the observations X

    Parameters
    ----------
    X : ndarray
        The set of observations of a binomial random variable
    alpha : float
        The error in mean that is tolerated when computing the confidence interval (default: 0.05)

    Returns
    -------
    mu : float
        The empirical mean of the binomial samples
    conf_delta : float
        The confidence delta of the confidence interval. The interval will be [mu - conf_delta, mu + conf_delta]
    """
    x_mean = np.mean(X)
    z = norm.ppf(1 - (alpha/2))
    return x_mean, z*np.sqrt(x_mean*(1-x_mean)/len(X))
