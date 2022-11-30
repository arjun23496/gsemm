import numpy as np

from scipy.special import logsumexp


def normalize(x):
    return (x-np.min(x))/np.clip(np.max(x) - np.min(x), 1e-10, None)


def scaled_tanh(alpha=1):
    """
    Returns a scaled tanh function. Basically stretches the codomain of the np.tanh function

    Parameters
    ----------
    alpha : float
        scaling constant (default=1)

    Returns
    -------
    ufunc
        tanh function scaled by alpha
    """
    return lambda x: alpha * np.tanh(x)


def create_softmax(beta=1):
    """
    Returns the softmax function with a temperature parameter beta

    Parameters
    ----------
    beta : float
        The temperature paramter of the softmax function (default=1)

    Returns
    -------
    ufunc
        softmax function with the temperature parameter beta
    """

    def _softmax(x, beta):
        x = x.copy()

        # v1
        # x = x - np.min(x) # numerical stability
        # return np.exp(beta * x - np.log(np.clip(np.sum(np.exp(beta * x)), 1e-10, None)))

        # v2
        # exp_in = beta * x - logsumexp(np.clip(beta * x, -1e10, 1e3))

        # v3
        exp_in = beta * x - logsumexp(beta * x)

        if (exp_in > 1e4).any():
            res = np.zeros(exp_in.shape)
            res[np.argmax(exp_in)] = 1
            return res
        else:
            return np.clip(np.exp(exp_in), 0, 1)

    return lambda x: _softmax(x, beta)


def dscaled_tanh(alpha=1):
    """
    Returns the derivative of the scaled tanh fucntion

    Parameters
    ----------
    alpha : float
        scaling constant (default=1)

    Returns
    -------
    ufunc
        derivative of the scaled tanh function
    """
    return lambda x: alpha * (1 - (np.tanh(x)) ** 2)


def identity(x: np.ndarray) -> np.ndarray:
    """
    Simply the identity function

    Parameters
    ----------
    x : type
        any type input

    Returns
    -------
    x : type
        Output the same value
    """
    return x
