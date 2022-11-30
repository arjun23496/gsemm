import numpy as np


def pattern_changes_from_temporal_pattern(outputs):
    """
    Extract only the temporal patern changes from the output temporal patterns

    Parameters
    ----------
    outputs : temporal pattern from the `temporal_pattern_from_output` function

    Returns
    -------
    list of int
            list of int elements where each element indicates the memory index of the stable states
    """
    res = [outputs[0]]
    for i in range(len(outputs)):
        if res[-1] != outputs[i]:
            res.append(outputs[i])

    return res


def temporal_pattern_from_output(outputs, ignore_steps=0):
    """
    Obtain the pattern of temporal information from the output. What this means is that given
    input information about the similarity between the network state at a point in time and
    the possible stable states(memories) the system can take, the function outputs the order stable states
    along the dynamical trajectory.

    Parameters
    ----------
    outputs : list of ndarray
        list of ndarray elements representing the similarity of the system state to some list of
        poossible stable states
    ignore_steps : int
        number of steps to ignore after a transition (default 20). This function is mainly useful
        when you dont want to consider the transition region where there is general confusion about
        which state the system is in.

    Returns
    -------
    list of int
        list of int elements where each element indicates the memory index of the stable states
    """
    memory_output = []
    for output in outputs:
        memory_output.append(np.argmax(output))

    # print(memory_output)
    cur_memory = memory_output[0]
    temporal_pattern = []
    ignoring = 0
    for i in range(1, len(memory_output)):
        if cur_memory != memory_output[i]:
            if ignoring >= ignore_steps:
                ignoring = 0
                temporal_pattern.append(cur_memory)
                cur_memory = memory_output[i]
            else:
                ignoring += 1
        else:
            ignoring = 0

    return memory_output, temporal_pattern


def find_best_chain_length(pattern1, pattern2):
    """
    Obtain the maximum number of times the smaller pattern is present in the longer pattern
    as a chain. So only repeated occurrences of the smaller pattern counts towards chain length.
    This function can be used to evaluate the stability of dynamical system chains.

    The order of parameter passing is not important as the function determines which pattern is
    shorter

    Parameters
    ----------
    pattern1 : list
        pattern as input to function
    pattern2
        pattern as input to function

    Returns
    -------
    int
        longest chain length of the shorter pattern present in the longer pattern
    """
    small_pattern = np.array(pattern1) if len(pattern1) < len(pattern2) else pattern2
    big_pattern = np.array(pattern1) if len(pattern1) > len(pattern2) else pattern2
    n_possible_patterns = len(big_pattern)//len(small_pattern)
    best_chain_length_till_now = -1
    for i in range((len(big_pattern)-len(small_pattern))+1):
        chain_length = 0
        for chain_start in range(i, (len(big_pattern)-len(small_pattern))+1, len(small_pattern)):
            acc = np.mean(big_pattern[chain_start:chain_start+len(small_pattern)] == small_pattern)

            if acc >= 1.0:
                chain_length += 1
            else:
                break
        if chain_length > best_chain_length_till_now:
            best_chain_length_till_now = chain_length

    return best_chain_length_till_now
