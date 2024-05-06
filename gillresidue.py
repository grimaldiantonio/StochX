# -*- coding: utf-8 -*-
"""
Simulate hydrogen exchange time course of an individual residue by Gillespie algorithm.
    
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

def simulate(k_op, k_cl, k_ch, t, N_replicas=100, show_time=False):
    """
    Parameters
    ----------
    k_op : float
        opening rate constant, in s^(-1)
    k_cl : float
        closing rate constant, in s^(-1)
    k_ch : float
        intrinsic exchange (chemical) rate constant, in s^(-1)
    t : float
        time at which the simulation can stop, in s
    N_replicas : int, optional
        number of replicas of the simulated residue. The default is 100.
    show_time : bool, optional
        print elapsed time for the calculation
    Returns
    -------
    gillespie_dict : dictionary with keywords
        - 't' : np.array
            timepoints of Gillespie simulation (random, not on a regular grid)
        - 'H_cl' : np.array, len(H_cl) = len(t)
            H_cl population at the timepoints of the simulation
        - 'H_op' : np.array, len(H_op) = len(t)
            H_op population at the timepoints of the simulation


    """
    print('Simulating Gillespie algorithm ... ')
    
    N_replicas = N_replicas
    
    if show_time:
        tic = time.time()
    
    P = k_cl/k_op
    
    initials = [int(N_replicas * P/(1+P)),  # [H_cl,
                int(N_replicas * 1/(1+P)),  #  H_op,
                0]                          #   D   ]
    
    propensities = [lambda h_cl, h_op, d: k_op * h_cl, # H_cl -> H_op
                    lambda h_cl, h_op, d: k_cl * h_op, # H_op -> H_cl
                    lambda h_cl, h_op, d: k_ch * h_op] # H_op -> D]
    
    """
    Transitions
    H_cl -> H_op : k_op * H_cl
    H_op -> H_cl : k_cl * H_op = k_op * P * H_op
    H_op -> D : k_ch * H_op
    """
    
    stoichiometry = [[-1, 1, 0],
                     [1, -1, 0],
                     [0, -1, 1]]
    
    t, species = gillespie.simulate(initials, propensities, stoichiometry, t)
    H_cl, H_op, D = zip(*species)

    gillespie_dict = {'t' : np.array(t),
                      'H_cl' : np.array(H_cl),
                      'H_op' : np.array(H_op),
                      'D' : np.array(D),
                      'N_replicas' : N_replicas}
    
    if show_time:
        toc = time.time()
        print('Elapsed time: %.3f s' % (toc - tic))
    else:
        print('Done!')
    
    return gillespie_dict    

def plot(gillespie_dict, H=False, save=False, title=None, concentrations=False):
    
    t, H_cl, H_op, D = gillespie_dict['t'], gillespie_dict['H_cl'], gillespie_dict['H_op'], gillespie_dict['D']
    
    if concentrations:
        N_replicas = gillespie_dict['N_replicas']
        H_cl = np.array(H_cl)/N_replicas
        H_op = np.array(H_op)/N_replicas
        D = np.array(D)/N_replicas
        
    plt.figure(figsize=(7,5), dpi=100)
    plt.plot(t, H_cl, label="H_cl", color="darkblue")
    plt.plot(t, H_op, label="H_op", color='cornflowerblue')
    if H:
        H = H_cl+H_op
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
