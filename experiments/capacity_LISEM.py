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

from episodic_memory.dynSim import DynamicalSystem


class EpisodicTanh(DynamicalSystem):
    def __init__(self, N=100, T_h=0.1, T_y=200, T_syn=50, beta_x=3.0, beta_y=5.0, step_size=0.01,
                 approximation='euler'):
        """
        The tanh version of the generalized episodic memory motivated from MultiTimescale Networks of Kurikawa

        Parameters
        ----------
        N : int
            Number of neurons in 1 layer of the MultiTimescale Network
        T_h : float
            Time constant associated with the fast layer
        T_y : float
            Time constant associated with the slow layer
        T_syn : float
            Time constant associated with learning dynamics
        beta_x : float
            multiplier for h layer activation
        beta_y : float
            multiplier for y layer activation
        step_size : float
            step size of the given approximation method
        approximation : str
            Approximation method to use ('euler'|'range-kutta')
        """
        super().__init__(step_size, approximation)
        self.N = N
        self.T_h = T_h
        self.T_y = T_y
        self.T_syn = T_syn
        self.beta_x = beta_x
        self.beta_y = beta_y
        self.h_activation = np.tanh
        self.y_activation = np.tanh
        self.dactivation = lambda x: (1 - np.tanh(x) ** 2)
        self.h = np.zeros(N)
        self.y = np.zeros(N)

    def initialize(self, seed=0, N=100, T_h=0.1, T_y=200, T_syn=50, beta_x=3.0, beta_y=5.0,
                   step_size=0.01, approximation='range-kutta'):
        """
        Initialize a complete network object with the provided arguments.
        (The default arguments work very well for most problems.)

        Parameters
        ----------
        seed : int
            Set seed for the random number generator
        N : int
            Number of Neurons
        T_h : float
            Time constant associated with the fast layer
        T_y : float
            Time constant associated with the slow layer
        T_syn : float
            Time constant associated with learning dynamics
        beta_x : float
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
        J = (7 / np.sqrt(N)) * (2 * np.random.randint(0, 2, (N, N)) - 1) * np.where(np.random.rand(N, N) < 0.1, 1, 0)
        lamda0 = (2 * np.random.randint(0, 2, (N, N)) - 1) * (1 / np.sqrt(N - 1))
        np.fill_diagonal(lamda0, 0)  # no self loops

        h0 = np.random.uniform(-1, 1, size=N)
        y0 = np.random.uniform(-1, 1, size=N)

        # experiment_parameters
        self.N = N
        self.T_h = T_h
        self.T_y = T_y
        self.T_syn = T_syn
        self.beta_x = beta_x
        self.beta_y = beta_y
        self.step_size = step_size
        self.approximation = approximation
        self.reset()

        self.set_interactions(J, lamda0.copy())

        ## initialization section
        self.I = np.zeros(N)
        self.h = h0.copy()
        self.y = y0.copy()

    def reset(self):
        """
        Resets some variables in the network

        Warnings
        -------
        Does not reset state variables of slow and fast layer
        """
        self.h = np.random.normal(0, 1 / self.N, size=self.N)
        self.y = np.random.normal(0, 1 / self.N, size=self.N)
        self.eta = np.random.normal(0, 1 / self.N, size=self.N)

    def get_state(self):
        """
        Returns the current state of  the network as arrays

        Returns
        -------
        h_state : ndarray
            State of h neurons (N,)
        y_state : ndarray
            State of y neurons (N,)
        """
        return self.h.copy(), self.y.copy()

    def set_interactions(self, J, lamda):
        """

        Parameters
        ----------
        J : ndarray
            Set Y->H interaction matrix (N, N)
        lamda : ndarray
            Set H->H interaction matrix (N, N)
        """
        self.J = J
        self.lamda = lamda

    def func_h(self, h):
        """
        Internal Function that computes the dh/dt function

        Parameters
        ----------
        h : ndarray
            State of h neurons (N,)

        Returns
        -------
        ndarray
            dh/dt(h) of size (N,)
        """
        return (1 / self.T_h) * (np.matmul(self.lamda * (np.ones(self.lamda.shape) - np.identity(self.N)),
                                           self.h_activation(self.beta_x * h)) \
                                 # + np.matmul(self.J, self.y_activation(self.y)) \
                                 - h)

    def func_y(self, y):
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
        return (1 / self.T_y) * (self.y_activation(self.beta_y * self.h_activation(self.beta_y * self.h)) - y)

    def func_lamda(self, lamda):
        """
        Computes dlambda/dt function for learning lambda parameters

        Parameters
        ----------
        lamda : ndarray
            Lambda interaction matrix (N,N)

        Returns
        -------
        ndarray
            Computed dlambda/dt
        """
        dlamda = (self.target - self.h_activation(self.beta_x * self.h)).reshape((-1, 1)) * (
                self.h_activation(self.beta_x * self.h).reshape((-1, 1)) - (
                    np.matmul(self.lamda * (np.ones(self.lamda.shape) - np.identity(self.N)),
                              self.h_activation(self.beta_x * self.h)).reshape((-1, 1)) * lamda)).T
        dlamda = (1 / self.T_syn) * (1 / self.N) * dlamda
        np.fill_diagonal(dlamda, 0)
        return dlamda

    def learn_lamda(self, target):
        """
        Logic for learning the lambda interactions

        Parameters
        ----------
        target : ndarray
            The target signal that is supposed to be output by h-layer (N,)
        """
        self.target = target
        self.lamda += self.get_step_approximation(self.func_lamda, self.lamda)

    def update(self, eta=None):
        """
        Update method of the dynamical system

        Parameters
        ----------
        eta : None
            Parameter not used but kept for consistency
        """
        self.h += self.get_d(self.func_h, self.h)
        self.y += self.get_d(self.func_y, self.y)


parser = argparse.ArgumentParser(description='Memory Limit Experiment.')

parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator')
parser.add_argument('--n_memories', type=int, default=0, help='Number of memories in chain')
parser.add_argument('--path', type=str, default="tmp/episodic_tanh", help='Location of the result directory')
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
    mnet = EpisodicTanh()
    mnet.initialize(seed=SEED,
                    N=N_NEURONS,
                    T_h=1.0,
                    beta_x=20.0,
                    approximation="euler")

    np.random.seed(SEED)

    # create patterns
    pattern_order = list(range(N_MEMORIES))

    def evaluate_at_alpha(alpha):
        T = np.identity(N_MEMORIES)
        T = np.roll(T, 1, axis=1)
        T = T * alpha

        memories = np.random.choice([-1, 1], (N_NEURONS, N_MEMORIES))

        mnet.lamda = memories @ T.T @ memories.T

        # test network
        mnet_test = copy.deepcopy(mnet)
        simulated = []

        # cue is a slight perturbation applied to
        # one of the memories
        cue = memories.T[0, :].astype('float').copy() + np.random.normal()
        mnet_test.h = cue.flatten()
        for i in range(int(0.5 * len(pattern_order) * 10000 / 6)):
            mnet_test.update(np.zeros(mnet.N))
            if i > 0:
                per = 1
            else:
                per = 50
            if i % per == 0:
                simulated.append((mnet.h_activation(mnet_test.h).copy(), None))

        pattern_correlations_test = np.array([var[0] for var in simulated]) @ memories / mnet.N

        # Obtain the temporal pattern by smoothing over the transition region
        smoothing_window = 10
        smoothed_output = []
        signal = temporal_pattern_from_output(pattern_correlations_test, ignore_steps=0)[0]
        for i in range(len(pattern_correlations_test) - smoothing_window):
            window = signal[i:i + smoothing_window]
            counts = np.bincount(window)
            smoothed_output.append(np.argmax(counts))

        system_output = pattern_changes_from_temporal_pattern(smoothed_output)

        return system_output

    # Find alpha min and max
    alpha_min = 0.1
    alpha_max = 2.0

    # Find alpha min
    while True:
        print("Evaluating min range: {} - {}".format(alpha_min, alpha_max))
        if alpha_min < 1e-3:
            print("WARNING: alpha min not found")
            sys.exit()

        system_output = evaluate_at_alpha(alpha_min)

        if len(system_output) < 5*N_MEMORIES:
            break
        else:
            alpha_max = alpha_min
            alpha_min = alpha_min/2

    # Find alpha max
    while True:
        print("Evaluating max range: {} - {}".format(alpha_min, alpha_max))
        if alpha_max > 10.0:
            print("WARNING: alpha max not found")
            sys.exit()

        system_output = evaluate_at_alpha(alpha_max)

        if len(system_output) > 5 * N_MEMORIES:
            break
        else:
            alpha_min = alpha_max
            alpha_max += 1

    print("Range found: {} - {}".format(alpha_min, alpha_max))
    print(system_output)
    # Evaluate alpha max
    best_output_chain_length = find_best_chain_length(system_output, pattern_order)
    best_possible_length = len(system_output) / N_MEMORIES

    print("best output: {}, best possible: {}".format(best_output_chain_length, best_possible_length))

    if best_output_chain_length / best_possible_length > 0.8 and len(system_output) > 5*N_MEMORIES:
        lowest_neurons = N_NEURONS

        print("seed: {}, n_memories: {}, lowest_neurons: {}, alpha: {}".format(SEED,
                                                                               N_MEMORIES,
                                                                               N_NEURONS,
                                                                               alpha_max))
        return True

    # Find best alpha
    while alpha_max - alpha_min > 1e-2:
        print("Finding alpha in: {} - {}".format(alpha_min, alpha_max))
        alpha = (alpha_max + alpha_min) / 2

        system_output = evaluate_at_alpha(alpha)
        print(system_output)
        if len(system_output) > 5 * N_MEMORIES:
            best_output_chain_length = find_best_chain_length(system_output, pattern_order)
            best_possible_length = len(system_output) / N_MEMORIES

            print("best output: {}, best possible: {}".format(best_output_chain_length, best_possible_length))

            if best_output_chain_length / best_possible_length > 0.8:
                print("seed: {}, n_memories: {}, lowest_neurons: {}, alpha: {}".format(SEED,
                                                                                       N_MEMORIES,
                                                                                       N_NEURONS,
                                                                                       alpha))
                return True
            else:
                alpha_max = alpha
        else:
            alpha_min = alpha

    return False


# Step 1: Find upper and lower bounds on number of neurons
print("--------------------- Step 1: Finding bounds")
upper_n_neurons = 100
lower_n_neurons = 2
while upper_n_neurons <= 400:
    print("evaluating upper bound range: {} - {}".format(lower_n_neurons, upper_n_neurons))
    if evaluate_at_neurons(upper_n_neurons):
        break

    lower_n_neurons = upper_n_neurons
    upper_n_neurons += 100

# If not in 1000 neurons, there is no solution
if upper_n_neurons > 1000:
    print("ERROR: range not found")
    sys.exit()

print("------------------------ Step 2 range: {} - {}".format(lower_n_neurons, upper_n_neurons))
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
