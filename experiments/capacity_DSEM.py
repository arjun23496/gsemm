import sys

from episodic_memory.utils.all_imports import *
from decouple import config  # obtain environment files

import argparse
import os

from episodic_memory.experiment import temporal_pattern_from_output, \
                                       find_best_chain_length, \
                                       pattern_changes_from_temporal_pattern
from episodic_memory.utils import add_experiment_id

from tqdm import tqdm

# Experiment Parameters

import numpy as np

from scipy.special import logsumexp

from episodic_memory.dynSim import DynamicalSystem
from episodic_memory.utils.functions import create_softmax, identity


class EpisodicModernHopfield(DynamicalSystem):
    def __init__(self, N_v=100, N_h=16, T_v=0.05, T_h=0.4, T_y=1.0, T_syn=10, beta_h=1.0, beta_y=1.0, step_size=0.01,
                 approximation='euler', lamda_multiplier=200, h_decay=1.0, y_decay=1.0, num_y_layers=0, j_multiplier=0,
                 h_context_size=0, h_context_multiplier=0.0):
        """
        The modern Hopfield version of the generalized episodic memory using y control layer introduced in KurikawaNet

        Parameters
        ----------
        N : int
            Number of neurons in 1 layer of the MultiTimescale Network
        T_v : float
            Time constant associated with the v layer
        T_h : float
            Time constant associated with the h layer
        T_y : float
            Time constant associated with the y layer
        T_syn : float
            Time constant associated with learning dynamics
        beta_h : float
            multiplier for h layer activation
        beta_y : float
            multiplier for y layer activation
        step_size : float
            step size of the given approximation method
        approximation : str
            Approximation method to use ('euler'|'range-kutta')
        """
        super().__init__(step_size, approximation)
        self.N_v = N_v
        self.N_h = N_h
        self.T_v = T_v
        self.T_h = T_h
        self.T_y = T_y
        self.T_syn = T_syn
        self.beta_h = beta_h
        self.h_decay = h_decay
        self.y_decay = y_decay
        self.v_activation = identity
        self.h_activation = create_softmax(beta_h)
        self.y_activation = create_softmax(beta_y)
        self.v = np.zeros(N_v)
        self.h = np.zeros(N_h)
        self.lamda_multiplier = lamda_multiplier
        self.j_multiplier = j_multiplier
        self.num_y_layers = num_y_layers
        self.h_context_size = h_context_size
        self.h_context_multiplier = h_context_multiplier

        if h_context_size > 0:
            self.W = np.zeros((h_context_size, N_h))

        if num_y_layers > 0:
            self.y_layers = np.zeros((num_y_layers, N_h))

    def initialize(self, seed=0, N_v=100, N_h=16, T_v=0.05, T_h=0.4, T_y=1.0, T_syn=10, beta_h=1.0, beta_y=1.0,
                   lamda_multiplier=200, h_decay=1.0, y_decay=1.0, num_y_layers=0, j_multiplier=0,
                   step_size=0.01, approximation='range-kutta', h_context_size=0, h_context_multiplier=0.0):
        """
        Initialize a complete network object with the provided arguments.
        (The default arguments work very well for most problems.)

        Parameters
        ----------
        seed : int
            Set seed for the random number generator
        N_v : int
            Number of visual layer neurons
        N_h : int
            Number of hidden layer neurons
        T_v : float
            Time constant associated with the v layer
        T_h : float
            Time constant associated with the fast layer
        T_y : float
            Time constant associated with the slow layer
        T_syn : float
            Time constant associated with learning dynamics
        beta_h : float
            Multiplier for h layer activation
        beta_y : float
            Multiplier for y layer activation
        step_size : float
            Step size of the given approximation method
        approximation : str
            Approximation method to use
        """
        np.random.seed(seed)

        # model initialization
        xi = np.random.uniform(-1, 1, size=(N_v, N_h))  # intialize xi to uniform random memories

        lamda0 = np.ones((N_h, N_h)) * 1e-10  # intialize lamda to a small positive number
        np.fill_diagonal(lamda0, 0)  # no self loops

        if num_y_layers > 0:
            J0 = np.ones((N_h, N_h)) * 1e-10
            np.fill_diagonal(J0, 0)
            J0 = np.tile(J0, (num_y_layers, 1))
        else:
            J0 = None

        v0 = np.random.uniform(-1, 1, size=N_v)
        h0 = np.zeros(N_h)
        y_layer0 = np.zeros((num_y_layers, N_h))

        # experiment_parameters
        self.N_v = N_v
        self.N_h = N_h
        self.T_v = T_v
        self.T_h = T_h
        self.T_y = T_y
        self.T_syn = T_syn
        self.beta_h = beta_h
        self.num_y_layers = num_y_layers

        self.beta_y = beta_y
        self.lamda_multiplier = lamda_multiplier
        self.h_decay = h_decay
        self.y_decay = y_decay

        self.v_activation = identity
        self.h_activation = create_softmax(beta_h)
        self.y_activation = create_softmax(beta_y)
        self.step_size = step_size
        self.approximation = approximation
        self.reset()

        self.set_interactions(xi, lamda0.copy(), J0)

        ## initialization section
        self.I = np.zeros(N_v)
        self.v = v0.copy()
        self.h = h0.copy()
        self.y = np.zeros(N_h)  # not used - kept for backward compatibility
        self.target = np.zeros(N_h)

        if num_y_layers > 0:
            self.y_layers = y_layer0.copy()
        self.j_multiplier = j_multiplier

        self.h_context_size = h_context_size
        if h_context_size > 0:
            self.h_context_vector = np.zeros(h_context_size)
            self.h_context_multiplier = h_context_multiplier
            self.W = np.ones((h_context_size, N_h)) * 1e-10

        self.debug_mode = False  # Debug mode to check the inner workings of the network

    def reset(self):
        """
        Resets some variables in the network

        Warnings
        -------
        Does not reset state variables of slow and fast layer
        """
        pass
        # self.v = np.random.normal(0, 1 / self.N_v, size=self.N_v)
        # self.h = np.random.normal(0, 1 / self.N_h, size=self.N_h)
        # self.y = np.random.normal(0, 1 / self.N_h, size=self.N_h)

    def get_state(self):
        """
        Returns the current state of  the network as arrays

        Returns
        -------
        v_state : ndarray
            State of v neurons (N_v,)
        h_state : ndarray
            State of h neurons (N_h,)
        y_state : ndarray
            State of y neurons (N_h,)
        """
        return self.v.copy(), self.h.copy(), self.y_layers.copy()

    def set_interactions(self, xi, lamda, J=None):
        """
        Function to set the interactions in the network

        Parameters
        ----------
        xi : ndarray
            Set the V-H interactiono matrix (N_v, N_h)
        lamda : ndarray
            Set H->H interaction matrix (N_h, N_h)
        """
        if self.num_y_layers > 0:
            self.J = J
        self.xi = xi
        self.lamda = lamda

    def func_y_layers(self, y_layers):
        """
        Internal function that computes the dy/dt function

        Parameters
        ----------
        y : ndarray
            State of y neurons (N,)

        Returns
        -------
        ndarray
            dy/dt(y) of size (N,)
        """
        y_layers_update = y_layers.copy()

        if self.T_y > 0:
            y_layers_update[0] = self.y_activation(self.h) - self.y_decay * y_layers[0]
            for layer_id in range(1, self.num_y_layers):
                y_layers_update[layer_id] = self.y_activation(y_layers[layer_id - 1]) - self.y_decay * y_layers[
                    layer_id]

            return (1 / self.T_y) * y_layers_update

        else:  # discrete case

            for layer_id in range(self.num_y_layers - 1, 0, -1):
                y_layers_update[layer_id] = y_layers[layer_id - 1] - self.y_decay * y_layers[layer_id]

            if not hasattr(self, "h_prev"):
                self.h_prev = np.zeros(self.N_h)
            y_layers_update[0] = self.h_prev - self.y_decay * y_layers[0]
            self.h_prev = self.y_activation(self.h)

            return y_layers_update

    def func_h(self, h):
        """
        Internal Function that computes the dh/dt function

        Parameters
        ----------
        h : ndarray
            State of h neurons (N_h,)

        Returns
        -------
        ndarray
            dh/dt(h) of size (N_h,)
        """
        g = self.v_activation(self.v)
        f = self.h_activation(h)
        t1 = (g @ self.xi).flatten()
        t2 = self.lamda_multiplier * (f @ self.lamda.T).flatten()
        t3 = - self.h_decay * h

        if self.debug_mode:
            self.h_t2 = t2.copy()

        ########## y-interaction logic
        if self.num_y_layers > 0:
            t4 = self.j_multiplier * (self.y_layers.flatten() @ self.J).flatten()
            # t2 = t2 * t4 # multiplicative interactions
            # t4 = np.zeros(t4.shape)
            if self.debug_mode:
                self.h_t4 = t4
        else:
            t4 = 0
        ######### end y-interaction logic

        ######### context logic
        if self.h_context_size > 0:
            t5 = self.h_activation(self.h_context_vector @ self.W).flatten()
            t2 = t2 * t5  # multiplicative interaction (modulatory signal)
            # t2 = t2 + self.h_context_multiplier*t5  # multiplicative interaction (modulatory signal)

            if self.debug_mode:
                self.h_t5 = t5.copy()
        ######### end context logic

        assert not (np.isnan(t1).any() or np.isnan(t2).any() or np.isnan(t3).any() or np.isnan(t4).any()), \
            print("h t1: {}, t2:{}, t3:{}, t4: {}".format(t1, t2, t3, t4))

        if self.T_h == 0:
            return t1 + t2 + t3 + t4
        else:
            return (1 / self.T_h) * (t1 + t2 + t3 + t4)

    def func_v(self, v):
        """
        Internal functioon that computes the dv/dt function

        Parameters
        ----------
        v : ndarray
            State of v neurons (N_v, )

        Returns
        -------
        ndarray
            dv/dt of size (N_v, )
        """
        f = self.h_activation(self.h)
        t1 = self.xi @ f
        t2 = -v

        assert not (np.isnan(t1).any() or np.isnan(t2).any()), print("v t1: {}, t2:{}".format(t1, t2))

        return (1 / self.T_v) * (t1 + t2)

    def func_lamda(self, lamda):
        """
        Computes dlambda/dt function for learning lambda parameters

        Parameters
        ----------
        lamda : ndarray
            Lambda interaction matrix (N_h, N_h)

        Returns
        -------
        ndarray
            Computed dlambda/dt
        """
        dlamda = (self.target - self.h_activation(self.h)).reshape((-1, 1)) * (
                self.h_activation(self.h).reshape((-1, 1)) - (
                np.matmul(self.lamda * (np.ones(self.lamda.shape) - np.identity(self.N_h)),
                          self.h_activation(self.h)).reshape((-1, 1)) * lamda)).T
        dlamda = (1 / self.T_syn) * (1 / self.N_h) * dlamda
        np.fill_diagonal(dlamda, 0)

        return dlamda

    def func_xi(self, xi):
        """
        Computes dlambda/dt function for learning lambda parameters

        Parameters
        ----------
        lamda : ndarray
            Lambda interaction matrix (N_h, N_h)

        Returns
        -------
        ndarray
            Computed dlambda/dt
        """
        # dxi = (self.xi_target - self.v).reshape((-1, 1)) * (
        #         self.v - (
        #         np.matmul(xi,
        #                   self.v.reshape((-1, 1))).T))
        # dxi = (self.xi_target.reshape((-1, 1)) * self.xi) # hebb
        dxi = self.v.reshape((-1, 1)) * (self.h_activation(self.h).reshape((1, -1)) - self.v.reshape((-1, 1)) * self.xi)

        # dxi -= self.v.reshape((-1, 1)) * self.v.reshape((1, -1)) #antihebbian

        dxi = (1 / self.T_syn) * (1 / self.N_v) * dxi

        return dxi

    def func_W(self, W):
        """
        Computes dlambda/dt function for learning lambda parameters

        Parameters
        ----------
        lamda : ndarray
            Lambda interaction matrix (N_h, N_h)

        Returns
        -------
        ndarray
            Computed dlambda/dt
        """
        # target_y = np.tile(self.target, self.num_y_layers)
        target_y = self.target

        target_y = target_y.reshape((-1, 1))
        current_y = self.h_context_vector.reshape((-1, 1))

        dW = target_y * current_y.T

        dW = (1 / self.T_syn) * (1 / self.N_h) * dW

        return dW.T

    def func_J(self, J):
        """
        Computes dlambda/dt function for learning lambda parameters

        Parameters
        ----------
        lamda : ndarray
            Lambda interaction matrix (N_h, N_h)

        Returns
        -------
        ndarray
            Computed dlambda/dt
        """
        # target_y = np.tile(self.target, self.num_y_layers)
        target_y = self.target
        tiled_identity = np.tile(np.identity(self.N_h), (self.num_y_layers, 1))

        target_y = target_y.reshape((-1, 1))
        current_y = self.y_layers.reshape((-1, 1))

        dJ = target_y * current_y.T
        # print(dJ.shape)
        # dJ -= current_y * current_y.T
        # dJ -= target_y.T * J.T * J
        # dJ += current_y.T * J.T * J

        dJ = (1 / self.T_syn) * (1 / self.N_h) * dJ

        dJ = dJ.T

        ## Remove diagonal
        dJ = dJ - dJ * tiled_identity

        # print(np.unravel_index(np.argmax(dJ, axis=None), dJ.shape))

        return dJ

    def learn_lamda(self, target):
        """
        Logic for learning the lambda interactions

        Parameters
        ----------
        target : ndarray
            The target signal that is supposed to be output by h-layer (N,)
        """
        self.xi_target = target
        self.target = np.zeros(self.N_h)
        self.target[np.argmax(self.h_activation(target @ self.xi))] = 1

        assert not np.isnan(self.target).any(), print("target: ", self.target)

        self.lamda += self.get_step_approximation(self.func_lamda, self.lamda)

        #
        self.xi += self.get_step_approximation(self.func_xi, self.xi)
        #

        if self.num_y_layers > 0:
            self.J += self.get_step_approximation(self.func_J, self.J)

        if self.h_context_size > 0:
            self.W += self.get_step_approximation(self.func_W, self.W)

    def update(self, eta=None):
        """
        Update method of the dynamical system

        Parameters
        ----------
        eta : None
            context parameter
        """
        if eta is not None:
            self.h_context_vector = eta

        if self.T_h == 0.0:
            self.h += self.func_h(self.h)
        else:
            self.h += self.get_d(self.func_h, self.h)

        self.v += self.get_d(self.func_v, self.v)

        if self.num_y_layers > 0:
            if self.T_y == 0:
                self.y_layers += self.func_y_layers(self.y_layers)
            else:
                self.y_layers += self.get_d(self.func_y_layers, self.y_layers)

    def get_energy(self, v, h, y):
        """
        Get energy of the network with the class parameters for different state settings (v, h, y)
        Parameters
        ----------
        v : ndarray
            State of v neurons (N_v, )
        h : ndarray
            State of h neurons (N_h, )
        y : ndarray
            State of y neurons (N_y, )

        Returns
        -------
        float
            A scalar value of the current energy of the system
        """
        t1 = np.sum((1 / 2) * v ** 2)
        t3 = -1 * logsumexp(h)
        t4 = - (v.reshape(1, -1) @ self.xi @ self.h_activation(h).reshape((-1, 1)))
        t5 = - (self.h_activation(h) @ self.lamda @ self.h_activation(h))

        return t1 + t3 + t4 + t5

    def get_jacobian(self):
        jacobian = np.zeros((self.N_v + self.N_h, self.N_v + self.N_h))

        # dvi/dt = f_i(h, v)
        # dh_{\mu}/dt = g_{\mu}(h, v)

        # df_i/v_j
        jacobian[:self.N_v, :self.N_v] = np.identity(self.N_v) * -1

        # df_i/h_{\mu}
        jacobian[:self.N_v, self.N_v:] = (self.xi - self.xi @ self.h_activation(self.h).reshape((-1, 1))) * \
                                         self.h_activation(self.h)

        # dg_mu/v_i
        jacobian[self.N_v:, :self.N_v] = self.xi.T

        # dg_mu/h_{\nu}
        jacobian[self.N_v:, self.N_v:] = self.lamda * self.h_activation(self.h).reshape((1, -1))
        jacobian[self.N_v:, self.N_v:] += np.identity(self.N_h) * -1
        jacobian[self.N_v:, self.N_v:] += self.lamda @ self.h_activation(self.h) * \
                                          self.h_activation(self.h).reshape((1, -1))

        return jacobian

parser = argparse.ArgumentParser(description='Memory Limit Experiment.')

parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator')
parser.add_argument('--n_memories', type=int, default=0, help='Number of memories in chain')
parser.add_argument('--path', type=str, default="tmp/episodic_hopfield", help='Location of the result directory')
parser.add_argument('--experiment_id', type=str, default="0", help='UID for experiment result files')

args = parser.parse_args()

SEED = args.seed
N_MEMORIES = args.n_memories
OUTPUT_DIR = os.path.join(config('EXPERIMENT_OUTPUT_DIR'), args.path)
EXPERIMENT_ID = args.experiment_id

# create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "final"), exist_ok=True)

print("Experiment Setup Complete")

print("Running ...")


def evaluate_at_neurons(N_NEURONS):
    mnet = EpisodicModernHopfield()
    mnet.initialize(seed=SEED,
                    N_v=N_NEURONS,
                    N_h=N_MEMORIES,
                    T_h=0.4,
                    T_v=0.05,
                    beta_h=1.0,
                    approximation="euler")

    np.random.seed(SEED)

    for alpha in np.arange(50, 500, 50):
    # for alpha in [150]:
        T = np.identity(N_MEMORIES)
        T = np.roll(T, 1, axis=1)
        T = T * alpha

        memories = np.random.choice([-1, 1], (N_NEURONS, N_MEMORIES))

        mnet.xi = memories
        mnet.lamda = T.T

        # create patterns
        pattern_order = list(range(N_MEMORIES))

        # test network
        mnet_test = copy.deepcopy(mnet)
        simulated = []

        # cue is a slight perturbation applied to
        # one of the memories
        cue = memories.T[0, :].astype('float').copy() + np.random.normal()
        mnet_test.v = cue.flatten()
        for i in tqdm(range(1 * len(pattern_order) * 10000 // 6)):
            mnet_test.update()
            if i > 0:
                per = 1
            else:
                per = 50
            if i % per == 0:
                simulated.append((mnet_test.v.copy(), None))

        pattern_correlations_test = np.array([var[0] for var in simulated]) @ memories / mnet.N_h

        # Obtain the temporal pattern by smoothing over the transition region
        smoothing_window = 1
        smoothed_output = []
        signal = temporal_pattern_from_output(pattern_correlations_test, ignore_steps=0)[0]
        for i in range(len(pattern_correlations_test) - smoothing_window):
            window = signal[i:i + smoothing_window]
            counts = np.bincount(window)
            smoothed_output.append(np.argmax(counts))

        system_output = pattern_changes_from_temporal_pattern(smoothed_output)

        print("Output: {}".format(system_output))

        # Check fidelity of output
        best_output_chain_length = find_best_chain_length(system_output, pattern_order)
        best_possible_length = len(system_output) / N_MEMORIES
        print("best model: {} best possible: {}".format(best_output_chain_length, best_possible_length))
        if best_output_chain_length / best_possible_length > 0.8 and len(system_output) > 5*N_MEMORIES:
            lowest_neurons = N_NEURONS

            print("seed: {}, n_memories: {}, lowest_neurons: {}, alpha: {}".format(SEED,
                                                                                   N_MEMORIES,
                                                                                   lowest_neurons,
                                                                                   alpha))
            return True

    return False


# Step 1: Find upper and lower bounds on number of neurons
print("Step 1")
upper_n_neurons = 100
lower_n_neurons = 2
while upper_n_neurons <= 1000:
    print("evaluating upper bound: {}".format(upper_n_neurons))
    if evaluate_at_neurons(upper_n_neurons):
        break

    lower_n_neurons = upper_n_neurons
    upper_n_neurons += 100

# If not in 1000 neurons, there is no solution
if upper_n_neurons > 1000:
    print("ERROR: range not found")
    sys.exit()

print("Step 2 range: {} - {}".format(lower_n_neurons, upper_n_neurons))
lowest_neurons = -1

while upper_n_neurons - lower_n_neurons > 1:
    new_limit = (upper_n_neurons + lower_n_neurons) // 2
    if evaluate_at_neurons(new_limit):
        print("New limit solved - changing upper to  {}".format(new_limit))
        upper_n_neurons = new_limit
    else:
        print("New limit did not solve - changing lower to  {}".format(new_limit))
        lower_n_neurons = new_limit
    print("New Range: {} - {}".format(lower_n_neurons, upper_n_neurons))

lowest_neurons = upper_n_neurons

if lowest_neurons > 0:
    result = {
        "seed": SEED,
        "n_memories": N_MEMORIES,
        "lowest_neurons": lowest_neurons
    }

    with open(os.path.join(OUTPUT_DIR, "final", add_experiment_id("result.pkl", EXPERIMENT_ID)), "wb") as f:
        pkl.dump(result, f)
