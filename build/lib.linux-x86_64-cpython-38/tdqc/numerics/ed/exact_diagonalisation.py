#!/usr/bin/env python
# coding: utf-8

import matplotlib
matplotlib.use('Agg')
import cmath
import math
import numpy as np
import matplotlib.pyplot as plt
#import import_ipynb
import tdqc.numerics.ed.models_ed



# CONVENTIONS
# spin down: [1,0]
# spin up: [0,1]
h_bar = 1 # 1.054571817*10**(-34) # in J.s

def thermal_exp_value(eig_values,eig_vectors,observable,inv_temperature):
    '''
    Thermal expectation values of an observable from ED.
    '''
    KB = 1 # Boltzmann constant: .380649*10**(-23) # in J.K**(-1) if 1.380649*10**(-23) or 8.617333262145*10**(-5) in eV.K**(-1)
    z = 0 # The partition function
    exp_value = 0
    for index,energy in enumerate(eig_values):
        coeff = cmath.exp(-energy*inv_temperature)#/(KB*temperature))
        z += coeff
        eig_vector = eig_vectors[:,index]
        exp_value += np.dot(np.dot(observable, eig_vector),eig_vector.conj())*coeff
    exp_value = exp_value/z
    return exp_value

def ground_states(eig_values,eig_vectors):
    length_vector = eig_vectors.shape[0]
    min_indices = np.asarray(abs(eig_values-eig_values.min())<10**(-12)).nonzero() #np.where(eig_values == eig_values.min())
    min_indices = np.asarray(min_indices)[0]
    ground_states = np.zeros([length_vector,min_indices.shape[0]],dtype='complex128')
    for idx, value in enumerate(min_indices):
        eig_vector = eig_vectors[:,value]
        eig_vector = eig_vector[:]
        ground_states[:, idx] = eig_vector
    return ground_states

def ground_state(eig_values,eig_vectors):
    min_index = np.argmin(eig_values)
    ground_state = eig_vectors[:,min_index]
    return ground_state

def ground_state_energy_per_site(eig_values,L):
    return min(eig_values)/L

class ExactDiagonalization(object):
    """  Quantum environment using the exact diagonalization for the time evolution.
    NOTE: It is not encapsulated in QuantumEnv().
    """
    def __init__(self,
            model,
            n_sites,
            initial_state,
            t_final,
            t_initial=0,
            step=0.001,
            seed=None,
            **other_params):
        self.model = model
        self.n_sites = n_sites
        self.initial_state = initial_state
        self.t_final = t_final
        self.t_initial = t_initial
        self.step = step
        self.amplitudes = None
        self.H = self.model.hamiltonian
        self.eig_values,self.eig_vectors = np.linalg.eigh(self.H)
        self.ground_state = None

    def set_ground_state(self):
        min_index = np.argmin(self.eig_values)
        self.ground_state = self.eig_vectors[:,min_index]

    def get_ground_state(self):
        if self.ground_state == None:
            self.set_ground_state()
        return self.ground_state

    def get_ground_state_energy_per_site(self):
        return min(self.eig_values)/self.n_sites
