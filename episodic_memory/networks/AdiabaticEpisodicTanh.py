"""
    The Tanh (Model A) version of the adiabatic episodic memory systems
"""

import numpy as np

from episodic_memory.dynSim import DynamicalSystem


class AdiabaticEpisodicTanh(DynamicalSystem):
    def __init__(self, N=100, T_f=0.1, gamma=1.0, alpha_s=1.0, alpha_c=1.0, T_d=1, beta_d=1.0, T_syn=50,
                 step_size=0.01, approximation='euler'):
        """
        The Tanh (Model A) version of the adiabatic episodic memory systems

        Parameters
        ----------
        N : int
            Number of feature layer neurons
        T_f : float
            Time constant associated with the feature layer
        T_d : float
            Time constant associated with the delay signal
        T_syn : float
            Time constant associated with learning dynamics
        gamma : float
            tanh activation function parameter
        alpha_s : float
            multiplication factor for self connections (associative memory part)
        alpha_c : float
            multiplication factor for cross connections (episodic memory part)
        beta_d : float
            multiplicative factor associated with delay signal
        step_size : float
            step size of the given approximation method
        approximation : str
            Approximation method to use ('euler'|'range-kutta')
        """
        super().__init__(step_size, approximation)

        # model parameters
        self.N = N
        self.T_f = T_f
        self.T_d = T_d
        self.beta_d = beta_d
        self.gamma = gamma
        self.alpha_s = alpha_s
        self.alpha_c = alpha_c
        self.T_syn = T_syn
        self.beta_d = beta_d

        # activation function
        self.g = lambda x: np.tanh(gamma * x)

        # state variables
        self.v = np.zeros(N)
        self.v_d = np.zeros(N)

        # interactions
        self.xi = None
        self.phi = None

    def initialize(self, seed=0, N=100, T_f=1.0, gamma=1.0, alpha_s=0.045, alpha_c=0.004, T_d=100, beta_d=7.0, T_syn=50,
                   n_patterns=5, step_size=0.01, approximation='range-kutta'):
        """
        Initialize a complete network object with the provided arguments.
        (The default arguments work very well for most problems.)

        Parameters
        ----------
        seed : int
            Set seed for the random number generator
        N : int
            Number of feature layer neurons
        T_f : float
            Time constant associated with the feature layer
        T_d : float
            Time constant associated with the delay signal
        T_syn : float
            Time constant associated with learning dynamics
        gamma : float
            tanh activation function parameter
        alpha_s : float
            multiplication factor for self connections (associative memory part)
        alpha_c : float
            multiplication factor for cross connections (episodic memory part)
        beta_d : float
            multiplicative factor associated with delay signal
        n_patterns : int
            number of patterns in episode (to create a circular related episodic memory system)
        step_size : float
            step size of the given approximation method
        approximation : str
            Approximation method to use ('euler'|'range-kutta')
        """
        np.random.seed(seed)

        # experiment_parameters
        self.N = N
        self.T_f = T_f
        self.T_d = T_d
        self.beta_d = beta_d
        self.gamma = gamma
        self.alpha_s = alpha_s
        self.alpha_c = alpha_c
        self.T_syn = T_syn
        self.beta_d = beta_d
        self.step_size = step_size
        self.approximation = approximation

        # activation function
        self.g = lambda x: np.tanh(gamma * x)

        # state variables
        self.v = np.zeros(N)
        self.v_d = np.zeros(N)

        # interactions
        xi = np.random.choice([-1, 1], (N, n_patterns)).astype('float')
        G = np.identity(n_patterns)
        G = np.roll(G, 1, axis=1)

        self.set_interactions(xi, G.T)

    def reset(self):
        """
        Resets some variables in the network
        """
        self.v = np.zeros(self.N)
        self.v_d = np.zeros(self.N)

    def set_interactions(self, xi, phi):
        """

        Parameters
        ----------
        xi : ndarray
            Set associative memory interaction matrix
        phi : ndarray
            Set episodic memory cross interaction matrix
        """
        self.xi = xi
        self.phi = phi

    def func_v_d(self, v_d):
        """
        Internal function that computes the d/dt(v_d) function

        Parameters
        ----------
        v_d : ndarray
            State of the delay signal of size (N,)

        Returns
        -------
        ndarray
            d/dt(v_d) of size (N,)
        """
        return (1 / self.T_d) * (self.beta_d * self.g(self.v) - v_d)

    def func_v(self, v):
        """
        Internal Function that computes the d/dt(v) function

        Parameters
        ----------
        v : ndarray
            State of feature layer neurons of size (N,)

        Returns
        -------
        ndarray
            d/dt(v) of size (N,)
        """

        return (1 / self.T_f) * (self.alpha_s * (self.xi @ self.xi.T) @ self.g(v)
                                 + self.alpha_c * (self.xi @ self.phi @ self.xi.T) @ self.v_d
                                 - v)

    def update(self, i_signal=None, *args, **kwargs):
        """
        Update method of the dynamical system with input forcing

        Parameters
        ----------
        i_signal : ndarray or None
            Input forcing signal
        """

        # handle input
        if i_signal is None:
            i_signal = np.zeros(self.N)

        # update steps
        self.v += self.get_d(lambda x: self.func_v(x)+i_signal, self.v)
        self.v_d += self.get_d(self.func_v_d, self.v_d)

    def get_energy(self, V=None, Vd=None, *args, **kwargs):
        """
        Compute energy of the system at a given state. If no state is provided, computes energy of
        the current state

        Parameters
        ----------
        V : ndarray
            State of the feature layer neurons of size (N,)
        Vd : ndarray
            State of the delay signal of size (N,)
        Returns
        -------
        float
            Scalar value for the computed energy of the system
        """
        if Vd is None:
            Vd = self.v_d

        if V is None:
            V = self.v

        gV = self.g(V)

        t1 = 0
        t1 = (V.T @ gV - (1 / self.gamma) * np.sum(np.log(np.abs(np.cosh(self.gamma * V)))))

        t21 = 0
        t21 = 0.5 * self.alpha_s * gV.T @ self.xi @ self.xi.T @ gV

        t22 = 0
        t22 = self.alpha_c * gV.T @ self.xi @ self.phi @ self.xi.T @ Vd

        t23 = 0
        # t23 = (0.5*self.alpha_c/self.alpha_s) * Vd.T @ self.xi @ self.phi.T @ self.phi @ self.xi.T @ Vd

        t2 = - (t21 + t22 + t23)

        return t1 + t2

    def get_energy_jacobian_v(self, V=None, Vd=None):
        if Vd is None:
            Vd = self.v_d

        if V is None:
            V = self.v

        jac = (1 - np.tanh(self.gamma * V) ** 2) * np.identity(self.N)

        t1 = self.gamma * V.T @ jac
        t2 = self.gamma * self.alpha_s * self.g(V).T @ self.xi @ self.xi.T @ jac
        t3 = self.alpha_c * self.gamma * Vd @ self.xi @ self.phi.T @ self.xi.T @ jac

        return t1 - t2 - t3

    def get_energy_hessian_v(self, V=None, Vd=None):
        if Vd is None:
            Vd = self.v_d

        if V is None:
            V = self.v
            
        jac = (1 - np.tanh(self.gamma * V)**2) * np.identity(self.N)

        t1 = self.gamma * (jac - 2 * self.gamma * (V * self.g(V)) * (1 - self.g(V)**2) * np.identity(self.N))

        t21 = jac @ self.xi @ self.xi.T @ jac
        t22 = -2 * ((self.xi @ self.xi.T @ self.g(V)) * self.g(V)) * (1 - self.g(V) ** 2) * np.identity(self.N)
        t2 = self.gamma**2 * self.alpha_s * (t21 + t22)

        t3 = self.gamma**2 * self.alpha_c * (- 2 * ((self.xi @ self.phi @ self.xi.T @ Vd) * self.g(V)) *
                                             (1 - self.g(V)**2) * np.identity(self.N))

        return t1 - t2 - t3
