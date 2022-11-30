import numpy as np

def episodic_learning_rule(mnet, target, *args, **kwargs):
    mnet.learn_lamda(target)


def discrete_episodic_hopfield_learning_rule(mnet, target, source, eta=None, *args, **kwargs):
    mnet.v = source + np.random.normal(scale=0.1, size=source.shape)
    mnet.h = mnet.h_activation(source @ mnet.xi) + np.random.normal(scale=0.1, size=mnet.h.shape)
    mnet.update(eta)

    mnet.learn_lamda(target)
