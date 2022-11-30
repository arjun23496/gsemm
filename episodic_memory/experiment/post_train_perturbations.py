import numpy as np

def episodic_modern_hopfield_perturbation(mnet):
    pass


def episodic_tanhnet_perturbation(mnet):
    mnet.h = mnet.h*np.arctanh(np.random.uniform(0, 1, mnet.h.shape))


def multitime_net_perturbation(mnet):
    mnet.h = mnet.h*np.random.uniform(0, 1, mnet.h.shape)
