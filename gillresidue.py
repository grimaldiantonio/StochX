# -*- coding: utf-8 -*-
"""
Simulate hydrogen exchange at resiude level by Gillespie algorithm.

Linderstrom-Lang theory, no back-exchange.
    
Transitions
-----------
    H_cl -> H_op : k_op * H_cl
    H_op -> H_cl : k_cl * H_op = k_op * P * H_op
    H_op -> D : k_ch * H_op
    
"""

import time

import numpy as np
import pandas as pd

import gillespie

import matplotlib.pyplot as plt

def simulate(k_op, k_cl, k_ch, t, N_replicas=100, show_time=False, print_info=False):
    """
    Parameters
    ----------
    k_op : float
        opening rate constant, s^-1
    k_cl : float
        closing rate constant, s^-1
    k_ch : float
        intrinsic exchange (chemical) rate constant, s^-1
    t : float
        time at which the simulation can stop, s
    N_replicas : int, optional
        number of replicas of the simulated residue. Default to 100.
    show_time : bool, optional
        print elapsed time for the calculation
    print_info : bool, optional
        print information on the proceeding of the calculation
        
    Returns
    -------
    gillespie_res : dictionary with following keywords
        - 't' : np.array
            timepoints of Gillespie simulation (random, not on a regular grid)
        - 'H_cl' : np.array, len(H_cl) = len(t)
            H_cl population at the timepoints of the simulation
        - 'H_op' : np.array, len(H_op) = len(t)
            H_op population at the timepoints of the simulation
        - 'D' : np.array, len(D) = len(t)
            D population at the timepoints of the simulation
        - 'N_replicas': int
            simulated number of replicas of the residue
    """
    
    if show_time:
        tic = time.time()
    if print_info:
        print('Simulating a residue with Gillespie algorithm ... ')
    
    P = k_cl/k_op   # protection factor
    
    initials = [int(np.round(N_replicas * P/(1+P))),  # [H_cl(0),
                int(np.round(N_replicas * 1/(1+P))),  #  H_op(0),
                0]                                    #     D(0)]
    
    propensities = [lambda h_cl, h_op, d: k_op * h_cl, # H_cl -> H_op
                    lambda h_cl, h_op, d: k_cl * h_op, # H_op -> H_cl
                    lambda h_cl, h_op, d: k_ch * h_op] # H_op -> D
    
    stoichiometry = [[-1, 1, 0], # H_cl -> H_op
                     [1, -1, 0], # H_op -> H_cl
                     [0, -1, 1]] # H_op -> D
    
    """
    Transitions
    -----------
    H_cl -> H_op : k_op * H_cl
    H_op -> H_cl : k_cl * H_op = k_op * P * H_op
    H_op -> D : k_ch * H_op
    """
    
    # simulation
    
    t, species = gillespie.simulate(initials, propensities, stoichiometry, t)
    H_cl, H_op, D = zip(*species)
    
    # store results

    gillespie_res = {'t' : np.array(t),
                     'H_cl' : np.array(H_cl),
                     'H_op' : np.array(H_op),
                     'D' : np.array(D),
                     'N_replicas' : N_replicas}
    
    if show_time:
        toc = time.time()
        print('Elapsed time: %.3f s' % (toc - tic))
    else:
        if print_info:
            print('Done!')
    
    return gillespie_res

def dict_to_df(gillespie_res):
    df = pd.DataFrame()
    df['t'] = gillespie_res['t']
    df['H_cl'] = gillespie_res['H_cl']
    df['H_op'] = gillespie_res['H_op']
    df['D'] = gillespie_res['D']
    return df

def Hcrop(gillespie_res, concentrations=False):
    """
    Parameters
    ----------
    gillespie_res: dict
        Output of Gillespie simulation of an individual residue.
    concentrations: bool
        If not False, the result is normalized by N_replicas. Default to False.
    
    Returns
    -------
    df : pandas.DataFrame
        Dataframe of reduced dimensions, keeping only HDX events.
    """
    
    df = dict_to_df(gillespie_res)
    
    df['H'] = df['H_cl'] + df['H_op']
    df.drop(columns=['H_cl', 'H_op'], inplace=True)
    
    df.drop_duplicates(subset='D', inplace=True)
    df.reset_index(inplace=True)
    df.drop(labels='index', axis=1, inplace=True)
    
    if concentrations:
        N_replicas = gillespie_res['N_replicas']
        df['H'] /= N_replicas
        df['D'] /= N_replicas
    
    return df

def plot(gillespie_res, H=False, save=False, title=None, concentrations=False):
    
    t, H_cl, H_op, D = gillespie_res['t'], gillespie_res['H_cl'], gillespie_res['H_op'], gillespie_res['D']
    
    if concentrations:
        N_replicas = gillespie_res['N_replicas']
        H_cl = np.array(H_cl)/N_replicas
        H_op = np.array(H_op)/N_replicas
        D = np.array(D)/N_replicas
        
    plt.figure(figsize=(7,5), dpi=100)
    plt.plot(t, H_cl, label="H_cl", color="darkblue")
    plt.plot(t, H_op, label="H_op", color='cornflowerblue')
    if H:
        H = H_cl + H_op
        plt.plot(t, H, label='H', color='slategray')
    plt.plot(t, D, label="D", color="deeppink")
    if title:
        plt.title(title)
    plt.xlabel("time (s)", fontsize=11)
    plt.ylabel("population", fontsize=11)
    plt.legend()
    if save:
        plt.savefig(save)
    plt.show()
