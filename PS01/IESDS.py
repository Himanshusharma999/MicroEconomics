import numpy as np

def iesds(u1: np.array, u2: np.array):
    assert(u1.ndim == u2.ndim)
    
    d = 1
    N = u1.ndim
    S1 = np.arange(1,u1.shape[0])
    S2 = np.arange(1,u2.shape[0])


    for payoffs in u1,u2:
        for i in range(1,u1.shape[0]):
            if payoffs[i] > payoffs[i+1]:
                S1 = S1[:i]
        
