import numpy as np


class DynamicalSystem:
    def __init__(self, step_size=0.01, approximation='range-kutta'):
        """
        Superclass for all memory models implemented using dynamical systems. This class has functions that
        can create trajectory solutions using the euler method or range-kutta

        Parameters
        ----------
        step_size : float
            The step size to be used by the approximator
        approximation : str
            The approximation method to use. Currently (euler|range-kutta)
        """
        self.step_size = step_size
        self.approximation = approximation

    def get_step_approximation(self, func, param):
        """
        Get the delta approximation for the dt solution of the dynamical system problem.

        Parameters
        ----------
        func : ufunc
            Function that evaluates the current gradient of the variable
        param : ndarray
            Current state of the dynamical variable

        Returns
        -------
        ndarray
            The step to take for the dynamical variable given
        """
        if self.approximation == 'euler':
            return self.step_size * func(param)
        elif self.approximation == 'range-kutta':
            k1 = func(param)
            k2 = func(param + (self.step_size * k1 / 2))
            k3 = func(param + (self.step_size * k2 / 2))
            k4 = func(param + (self.step_size * k3))
            return (self.step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def get_d(self, func, param):
        """
        A utility function that just calls the get_step_approximation function

        Parameters
        ----------
        func : ufunc
            Function that evaluates the current gradient of the variable
        param : ndarray
            Current state of the dynamical variable

        Returns
        -------
        ndarray
            The step to take for the dynamical variable given
        """
        return self.get_step_approximation(func, param)
