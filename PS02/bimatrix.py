import numpy as np 
import pandas as pd 

# useful printing function 
def print_payoffs(U1, U2): 
    '''
    Inputs: Two payoff matrices (with identical dimensions)
    Output: 
        Pandas dataframe
    '''
    na1,na2 = U1.shape
    tuple_list = [[(U1[r,c],U2[r,c]) for c in range(na2)] for r in range(na1)]
    return pd.DataFrame(tuple_list) 

def check_dominance_for_1(U1, DOPRINT=False):
    '''
    Input: 
        U1: (na1*na2) numpy array of utilities for the row player
    Returns: 
        dominated_actions: list of integers (or empty) for the dominated actions
    '''
    na1,na2 = U1.shape
    dominated_actions = []
    for a in range(na1):
        for a_ in range(na1): 
            if a_ == a: 
                continue
            if (U1[a, :] < U1[a_, :]).all():
                if DOPRINT: 
                    print(f'a={a} is strictly dominated by a={a_}')
                dominated_actions.append(a)
                break
    return dominated_actions

def IESDS(U1_in, U2_in, maxit=100, DOPRINT=False): 
    U1 = U1_in.copy()
    U2 = U2_in.copy()
    na1,na2 = U1.shape
    aa1 = np.arange(na1)
    aa2 = np.arange(na2)
    
    for it in range(maxit): 
        d = False
        
        # check for player 1
        a_del = check_dominance_for_1(U1, DOPRINT)
        a_keep = [a for a in range(U1.shape[0]) if a not in a_del]
        aa1 = aa1[a_keep]
        U1 = U1[a_keep, :]
        U2 = U2[a_keep, :]
        
        if len(a_del) > 0: 
            d = True   
        
        # check for player 2 
        a_del = check_dominance_for_1(U2.T)
        a_keep = [a for a in range(U2.shape[1]) if a not in a_del]
        aa2 = aa2[a_keep] 
        U1 = U1[:, a_keep]
        U2 = U2[:, a_keep]
        
        if len(a_del) > 0: 
            d = True    
            
        if not d: 
            print(f'No further strategies to delete after {it} iterations')
            break 
        
    assert it < maxit, f'Algorithm did not converge'
    
    return U1, U2, aa1, aa2