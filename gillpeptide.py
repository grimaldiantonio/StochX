# -*- coding: utf-8 -*-
"""
    After simulating a series of N residues constituting a peptide, we have N
'gillespie_dict' objects, output of 'simulate' function from gillresidue.
    We shall now build up deuterium uptakes for all peptide to compute peptide-
level uptakes.
"""

import time

import numpy as np
import pandas as pd

from copy import deepcopy

import gillresidue as Gr

#%%

def simulate(peptide, t, N_replicas=100,
             store_fullsim=True, show_time=False, print_info=False):
    """
    Parameters
    ----------
    peptide : pandas.DataFrame
        each column represents a residue;
        - Index : range(len(peptide))
        - Columns: 'residx', 'letter', 'k_op', 'k_cl', 'k_ch'
    t : float
        time at which the simulation can stop, s
    N_replicas : int, optional (default to 100)
        number of replicas of each simulated residue
    store_sim : bool, optional (default to True)
        store the results of the simulation for all residues in the peptide
        not in terms of [H,D] but in terms of [H_cl, H_op, D]
    show_time : bool, optional (default to False)
        print elapsed time for the calculation
    print_info : bool, optional (default to False)
        print information on the proceeding of the calculation

    Returns
    -------
    gillespie_pep : dictionary with following keywords
        - 't' : np.array
            timepoints of Gillespie simulation at which one exchange event has
            occurred(across the whole peptide)
        - 'D' : np.array, len(D) = len(t)
            D population at the timepoints of the simulation
        - 'seq' : str, len(seq) = N_peptide
            sequence of the peptide
        - 'N_peptide' : int
            length of the peptide (number of residues)
        - 'N_replicas': int
            simulated number of replicas per residue
        - 'sim' : list
            list of dictionaries of the type gillespie_res, each containing
            simulation results for individual residues
    """
    
    if show_time:
        tic = time.time()
    if print_info:
        print('Simulating Gillespie algorithm ... ')
    
    N_peptide = len(peptide)
    
    peptide_seq = ''
    
    sim = []
    times = []
    if store_fullsim:
        sim_full = []
    
    for idx in range(N_peptide):
        peptide_seq += peptide['letter'][idx]
        if print_info:
            print(' - residue %2i/%2i' % (idx+1, len(peptide)))
        if store_fullsim:
            sim_full.append(
                    Gr.simulate(k_op = peptide['k_op'][idx],
                                k_cl = peptide['k_cl'][idx],
                                k_ch = peptide['k_ch'][idx],
                                t=t, N_replicas=N_replicas,
                                show_time=False, print_info=False)
                    )
            sim.append(Gr.Hcrop(sim_full[-1]))
        else:
            sim.append(Gr.Hcrop(
                    Gr.simulate(k_op = peptide['k_op'][idx],
                                k_cl = peptide['k_cl'][idx],
                                k_ch = peptide['k_ch'][idx],
                                t=t, N_replicas=N_replicas,
                                show_time=False, print_info=False)
                    ))
        times.append(deepcopy(np.array(sim[-1]['t'])))

    time_unique = np.sort(np.unique(np.concatenate(times[:]))) # delete 0s
        
    # Create dictionary saving the result
    
    gillespie_pep = {'t' : time_unique,
                     'D' : np.arange(0, len(time_unique), step=1),
                     'seq' : peptide_seq,
                     'N_peptide' : N_peptide,
                     'N_replicas' : N_replicas}
    
    if store_fullsim:
        gillespie_pep['sim'] = sim_full
    else:
        gillespie_pep['sim'] = sim
    
    if show_time:
        toc = time.time()
        print('Elapsed time: %.3f s' % (toc - tic))
    else:
        print('Done!')

    return gillespie_pep

#%% peptides' pandas.DataFrame construction

import math
import random

def df_peptide():
    """
    Create a pandas.DataFrame object with keys:
        - 'residx' : residue index (position in sequence)
        - 'letter' : residue type (one-letter code)
        - 'k_op' : list of opening rate constants k_op for each residue
        - 'k_cl' : list of closing rate constants k_op for each residue
        - 'k_ch' : list of chemical rate constants k_op for each residue
    """
    colnames = ['residx', 'letter', 'k_op', 'k_cl', 'k_ch']
    return pd.DataFrame(columns=colnames)

def add_residue(df, residx, letter, k_op, k_cl, k_ch):
    """
    Add a residue to the peptide dataframe.
    """
    line = len(df)
    df.loc[line] = [residx, letter, k_op, k_cl, k_ch]
    return df

def synthetic_peptide(seq):
    """
        Create a pandas.DataFrame object representing a peptide randomly chosen
    from a given sequence.
    
    Parameters
    ----------
    seq : str
        Protein sequence, either real or fictitious.
    
    Returns
    -------
    df : pandas.DataFrame
        Peptide dataframe to run the simulation.
    """
    
    df = df_peptide()
    
    # Randomly define rate constants for the whole sequence
    k_op = np.random.rand(len(seq))*8.29*(10**-4)
    k_cl = np.random.rand(len(seq))*31.6*k_op
    k_ch = np.random.rand(len(seq))*3.75*(10**-3)
    
    # Randomly define peptide's first and last residue
    first_residue = 0
    last_residue = 0
    while first_residue >= last_residue:
        first_residue = math.floor(len(seq)*random.random())+1
        last_residue = math.floor(len(seq)*random.random())+1
        N_peptide = last_residue - first_residue + 1
        if N_peptide < 3 or N_peptide > 10:
            last_residue = first_residue
    
    # Actually build the dataframe
    for residx in range(int(first_residue), int(last_residue)+1):
        idx = residx - 1 
        df = add_residue(df, residx, seq[idx], k_op[idx], k_cl[idx], k_ch[idx])
    
    return df

#%% plot

import matplotlib.pyplot as plt

def plot(gillespie_pep, H=False, save=False, normalize=True, title=True):
    
    t = gillespie_pep['t']
    D = gillespie_pep['D']
    
    if normalize is True:
        N_replicas = gillespie_pep['N_replicas']
        D = D/N_replicas

    plt.figure(figsize=(7,5), dpi=100)
    plt.plot(t, D)
    if title is True:
        plt.title('Peptide %s' % gillespie_pep['seq'])
    plt.xlabel("time (s)", fontsize=11)
    plt.ylabel("uptake", fontsize=11)
    if normalize is True:
        N_peptide = gillespie_pep['N_peptide']
        plt.ylim(0, N_peptide)
    if save:
        plt.savefig(save)
    plt.show()
