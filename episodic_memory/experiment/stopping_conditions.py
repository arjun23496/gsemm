import numpy as np
from episodic_memory.networks import EpisodicTanh, EpisodicModernHopfield


def no_stopping_condition(*args, **kwargs) -> bool:
    """
    No stopping training, so essentially stopping the training should be handled elsewhere
    in the algorithm

    Parameters
    ----------
    No Parameters :
        parameters that are passed are not used

    Returns
    -------
    False : bool
        Return False to indicate that training never stops
    """
    return False


def mmu_condition_EpisodicModernHopfield(mnet: EpisodicModernHopfield, target: np.ndarray) -> bool:
    """
    Stopping criterion for episodicModernHopfield network

    Parameters
    ----------
    mnet : object
        object of a class of dynamical systems. The object needs to have h which is the
        neurons that need to match target
    target : ndarray
        Array indicating the state that h neurons needs to be in when training is complete

    Returns
    -------
    bool
        True/False indicating if the condition is satisfied or not
    """
    mnet_h = mnet.h_activation(mnet.h)

    # get the propagated target
    aux_target = np.zeros(mnet.N_h)
    aux_target[np.argmax(mnet.h_activation(target @ mnet.xi))] = 1

    mmu = np.sum(aux_target * mnet_h)
    if mmu > 0.9:
        return True
    else:
        return False


def mmu_ysuf_condition_episodicTanh_net(mnet: EpisodicTanh, target_pattern: np.ndarray) -> bool:
    """
    Stopping criterion for episodicTanh network

    Parameters
    ----------
    mnet : object
        object of a class of dynamical systems. The object needs to have h which is the
        neurons that need to match target
    target_pattern : ndarray
        Array indicating the state that h neurons needs to be in if trained well.

    Returns
    -------
    bool
        True/False indicating if the condition is satisfied or not
    """
    # normalize
    aux_target = target_pattern
    mnet_h = np.tanh(mnet.beta_x * mnet.h)
    mmu = np.mean(aux_target * mnet_h)
    ysuf = np.mean(mnet_h * np.tanh(mnet.beta_y * mnet.y))
    if mmu > 0.85 and ysuf > 0.5:
        return True
    else:
        return False


def mmu_ysuf_condition(mnet, target_pattern):
    """
    Default stopping condition for Kurikawa Network

    Parameters
    ----------
    mnet : object
        object of a class of dynamical systems. The object needs to have h which is the
        neurons that need to match target
    target_pattern : ndarray
        Array indicating the state that h neurons needs to be in if trained well.

    Returns
    -------
    bool
        True/False indicating if the condition is satisfied or not
    """
    # normalize
    aux_target = target_pattern
    mnet_h = mnet.h
    mmu = np.mean(aux_target * mnet_h)
    ysuf = np.mean(mnet.h * mnet.y)
    if mmu > 0.85 and ysuf > 0.5:
        return True
    else:
        return False
