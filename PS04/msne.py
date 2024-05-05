import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import nashpy as n

def BR_1(Eu1): 
    '''Best response in binary actions 
    '''
    x,G = Eu1.shape
    assert x == 2, f'Eu must be 2*G'
    
    br = np.empty((G,))
    
    I = Eu1[0,:] > Eu1[1,:]
    br[I] = 1 # the 0th action gives the highest payoff, so Pr(0th action) = 100% 
    I = Eu1[0,:] < Eu1[1,:]
    br[I] = 0 # the last action gives the highest payoff, so Pr(first action) = 0%
    I = Eu1[0,:] == Eu1[1,:]
    br[I] = 0.5 # actually, the best response is *any* probability in [0;1]. But for plotting purposes, we use 
    
    return br

def solve_for_MSNE(U1, U2, DOPLOT=True): 
    g = n.Game(U1, U2)
    eqs = list(g.lemke_howson_enumeration())
    print(f'Found {len(eqs)} equilibria:')
    for i,eq in enumerate(eqs):
        print(f'{i+1}: {eq}')
    if DOPLOT: 
        eqs = np.array(eqs) # 3-dim: (equilibrium, player, action)
        i_action = 0 # the x and y axes will show Pr(first action chosen)
        plt.scatter(eqs[:,0,i_action], eqs[:,1,i_action], color='black', label='MSNE (Lemke-Howson) ') # the axes show 
        for eq in np.array(eqs): 
            plt.scatter(eq[0,0], eq[1,0], color='black')
    else: 
        return eqs

def plot_BR_functions_2x2(U1, U2, G=100):
    '''
    Inputs:
        G: (int) number of grid points 
    Outputs: None
    '''
    p = np.linspace(0,1,G)
    pp = np.hstack([p,1-p]).reshape(2,G)

    Eu1 = U1 @ pp
    br1 = BR_1(Eu1)
    Eu2 = pp.T @ U2
    br2 = BR_1(Eu2.T)

    # bonus stuff to handle when there is mixing in the extremes 
    ps = dict()
    brs = dict()
    for i,br in enumerate([br1, br2]): 
        assert not (br==0.5).all() , f'Not implemented for pure mixing'
        ps[i] = p.copy()
        brs[i] = br.copy()
        if br[0] == 0.5: # indifference occurs at the very first element
            ps[i] = np.insert(ps[i], 0, 0.0) # the first probability is 0%
            if br[1] == 1.0: # increasing 
                brs[i]    = np.insert(br, 0, 0.0) # 0.0 to 0.5 to 1.0
            else: # decreasing 
                brs[i]    = np.insert(br, 0, 1.0) # 1.0 to 0.5 to 0.0
        if br[-1] == 0.5: # indifference at the last point 
            ps[i] = np.append(ps[i], 1.0) # the last probability is 100%
            if br[-2] == 1.0: # decreasing 
                brs[i]    = np.append(br, 0.0) # from 1.0 to 0.5 to 0.0 
            else: 
                brs[i]    = np.append(br, 1.0) # from 0.0 to 0.5 to 1.0 

    fig,ax = plt.subplots()
    ax.plot(brs[0], ps[0], '-',  color='r', label=f'Player 1: $BR_1(a_2)$');
    ax.plot(ps[1], brs[1], '--', color='b', label=f'Player 2: $BR_2(a_1)$');

    # add the MSNE on top of the graph 
    eqs = solve_for_MSNE(U1, U2, DOPLOT=True)
    
    ax.legend(loc='best');
    ax.set_xlabel('$\Pr(a_1 = 0)$');
    ax.set_ylabel('$\Pr(a_2 = 0)$');    
