#%% Modules
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.optimize import minimize 

#%% Functions
def u1(y1, y2): 
    return y1*y2 - 1./3. * (y1)**3

def u2(y2, y1): # note the order!
    return y1*y2 - 1./2. * (y2)**2

def u(y): 
    assert y.size == 2 
    y1, y2 = y
    return u1(y1,y2) + u2(y2,y1)

y0 = np.array([0.5,0.5])

# %% Nash 
# The Nash equilibrium is found where both players are best responding to each other. 
# We write the best response functions as numerical functions here; then we can re-use 
# them in other settings too 
def BR2(y1): 
    y2 = y1 / 2
    return y2
    
def BR1(y2):
    y1 = np.sqrt(y2)
    return y1

yy = np.linspace(0., 4., 100)
yy1 = np.array([BR1(y) for y in yy])
yy2 = np.array([BR2(y) for y in yy])

plt.plot(yy, yy2, label=f'BR2(y1)')
plt.plot(yy1, yy, label=f'BR1(y2)')
plt.legend(); 
plt.xlabel('$y_1$');
plt.xlabel('$y_2$');

#%% Iterated Bast Response
# Not always guaranteed to converge, but when it does we know that we have a Nash equilibrium. 
# we set a max number of iterations to avoid infinite loops
def IBR(y0:np.ndarray, maxit=100, tol=1e-5) -> np.ndarray: 
    assert y0.size == 2 
    y1,y2 = y0 
    success = False

    for it in range(maxit): 
        # copy the old values
        y1_ = y1*1.
        y2_ = y2*1.

        # update y1 and y2 
        y2 = BR2(y1_)
        y1 = BR1(y2_)

        if (np.abs(y1-y1_).max() < 1e-6) and (np.abs(y2-y2_).max() < 1e-6):
            print(f'IBR successful after {it} iterations')
            success = True
            break 

    if not success: 
        print(f'IBR failed after {it} iterations')

    return np.array([y1,y2])

# call your function 
yNE = IBR(y0)
print(f'yNE = {yNE}')

# %% Sequential

# FILL IN 

ySEQ = np.array([1, 1/2]) # FILL IN 

print(f'Sequential: {ySEQ.round(3)}')

# %% Social optimum (SO)
# This occurs when we maximize the sum of the two players' utilities wrt. both their choices
y0 = np.array([0.5, 0.5])
ySO = np.array([2,2]) # FILL IN 
print(f'SO: {yNE.round(3)}')

plt.plot(yy, yy2, label=f'BR2(y1)')
plt.plot(yy1, yy, label=f'BR1(y2)')
plt.plot(ySO[0], ySO[1], 'o', label='SO')
plt.plot(yNE[0], yNE[1], 'o', label='NE')
plt.legend(); 
plt.xlabel('$y_1$');
plt.xlabel('$y_2$');



# %%
