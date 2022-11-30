import numpy as np
# from tqdm.notebook import tqdm
# from tqdm.notebook import trange
from .stopping_conditions import *
from .post_train_perturbations import *
from .learning_rules import *
import copy


def train_episodic_network(mnet, patterns, pattern_order, eta, tracking_variables=[], tracking_functions=[],
                           epoch_tracking_functions=[],
                           diagnostic_functions=[],
                           stopping_condition=mmu_ysuf_condition,
                           post_net_perturbations=[multitime_net_perturbation],
                           learning_rule=episodic_learning_rule,
                           epochs=20, show_step_progress=True, max_steps=100000, tqdm_custom=None,
                           diagnostic_frequency=100):
    """
    Function to train episodic memory networks using some stopping criterion.
    Function is designed for maximum customizability but the default version can be used to
    train Kurikawa MultiTime net architecture

    Parameters
    ----------
    mnet : object
        The dynamical systems object that we need to train
    patterns : list of ndarray (N,)
        A list of nd arrays representing the number of possible states
        a part of the dynamical system can be in as target
    pattern_order : list of pattern indices
        A list indicating the order of patterns that the target network is supposed to
        output. The pattern order is treated as a limit cycle
    eta : ndarray (N,)
        Context vector to be passed to the dynamical system. Can also be treated as an
        input vector
    tracking_variables : list of str
        list with the states of the network that we need to track along the dynamical trajectory
        each element of the list is a str with the name of the variable
    tracking_functions : list of ufunc
        list of functions that tracks some function of states of network along the dynamical trajectory.
        The functions should take the network object as parameter and return something. func(mnet) -> something
    diagnostic_functions : list of ufunc
        Similar list of functions to tracking functions but the functions are not tracked. You can have
        functions that print out some aspects of the training using these functions. func(mnet, **kwargs) -> None
    stopping_condition : ufunc
        Function that returns if the training should stop or not. Default: Kurikawa stopping criterion.
        The function should take as input the current state of the dynamical system as object and the target
        vector. The function returns True or False indicating if the stopping criterion is satisfied or not.
    epochs : int
        Number of epochs to run the training procedure (default: 20)
    show_step_progress : bool
        Indicates if the inner progress bar should be shown or not (default: True)
    max_steps : int
        The maximum number of steps for each epoch in case stopping criterion is never satisfied (default: 100000)
    tqdm_custom : tqdm object
        The current instant of tqdm passed to the function. This allows this function to work with both jupyter
        notebook progressbars and terminal progerss bars
    diagnostic_frequency : int
        How frequently to execute the diagnostic functionalities like extracting state of network (default : 100)

    Returns
    -------
    mnet : object
        The trained object of the dynamical system
    network_variable_trajectory : dict of lists
        a dictionary of lists with each key the variable of the net that is tracked and
        the value is the trajectory of the variable
    network_function_trajectory : dict of lists
        a dictionary of lists with each key the function of the net that is tracked and
        the value is the trajectory of the variable
    epoch_function_trajectory : dict of lists
        a dictionary of lists with each key a fucntion of net and inputs that are tracked at each epoch
    """
    tqdm = tqdm_custom
    outer_pbar = tqdm(range(epochs*len(pattern_order)))
    inner_pbar = tqdm(None, total=10, disable=not show_step_progress)
    inner_pbar.reset(max_steps)

    network_variable_trajectory = { var: [] for var in tracking_variables }
    network_function_trajectory = { func.__name__: [] for func in tracking_functions }
    epoch_function_trajectory = { func.__name__: [] for func in epoch_tracking_functions }
    for learning_step in outer_pbar:
        outer_pbar.set_description("Processing memory: {}".format(pattern_order[learning_step%len(pattern_order)]))
        inner_pbar.reset(max_steps)
        for single_step in range(max_steps):
            memory_idx = learning_step%len(pattern_order)

            inner_pbar.update(1)

            mnet.update(eta)
            target_pattern = patterns[pattern_order[memory_idx]]

            if single_step % diagnostic_frequency == 0 or single_step < 1:
                # Obtain information about tracked variabless
                for var_idx, tracking_variable in enumerate(tracking_variables):
                    var_val = getattr(mnet, tracking_variable).copy()
                    network_variable_trajectory[tracking_variable].append(var_val)

                # Obtain information about tracked functions
                for func in tracking_functions:
                    network_function_trajectory[func.__name__].append(func(mnet))

                # Compute diagnostic functions
                for func in diagnostic_functions:
                    func(mnet, target_pattern=target_pattern)

            if stopping_condition(mnet, target_pattern):
                ## pertubations after a round of training a single memory
                for perturbation in post_net_perturbations:
                    perturbation(mnet)
                break
            # mnet.learn_lamda(target_pattern)
            learning_rule(mnet,
                          target=target_pattern,
                          source=patterns[pattern_order[memory_idx-1]],
                          eta=eta)

        if learning_step % len(pattern_order) == 0:
            for func in epoch_tracking_functions:
                epoch_function_trajectory[func.__name__].append(func(mnet,
                                                                     patterns=patterns,
                                                                     pattern_order=pattern_order))

    return mnet, network_variable_trajectory, network_function_trajectory, epoch_function_trajectory
