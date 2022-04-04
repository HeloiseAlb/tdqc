#!/usr/bin/env python
# coding: utf-8

import matplotlib
matplotlib.use('Agg')
import cmath
import math
import numpy as np
import matplotlib.pyplot as plt
#import import_ipynb
import models_ed



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
        self.eig_values,self.eig_vectors = np.linalg.eig(self.H)
        self.ground_state = None

    def set_ground_state(self):
        min_index = np.argmin(self.eig_values)
        self.ground_state = self.eig_vectors[:,min_index]

    def get_ground_state(self):
        if self.ground_state == None:
            set_ground_state()
        return self.ground_state

    def get_ground_state_energy_per_site(self):
        return min(self.eig_values)/self.n_sites

    def run_time_evolution(self):
        # This function run the time evolution and store the amplitudes in self.amplitudes. 
        psi_t_n = models_ed.State(self.initial_state)
        site_list = [l for l in range(1,self.n_sites,1)]
        t_list = [t for t in np.arange(self.t_initial,self.t_final,self.step)]
        amplitudes = np.zeros([int((self.t_final-self.t_initial)/self.step),2**self.n_sites],dtype='complex128') # [None] * int((t_max-t_min)/step) #np.zeros([int((t_max-t_min)/step)])
        inv_temperature = 1 
        for idx,t_n in enumerate(t_list):
            amplitudes[idx,:] = psi_t_n.amplitudes
            ### Time evolution
            ### Imaginary time evolution 
            psi_t_n.time_step_ed(self.model,-1j*self.step)
        # Check of the sum of probabilities
        #print(thermal_exp_value(eig_values,eig_vectors,H,0))
        self.amplitudes = amplitudes

    def get_time_evolution(self):
        # This function returns the amplitudes of the time evolution.
        return self.amplitudes

"""
for _,l in enumerate(site_list):
    H = xxz_model(l,Jxy,Jzz,PDB=True)
    # Diagonalization
    eig_values,eig_vectors = np.linalg.eig(H) 
    gs = ground_state_energy_per_site(eig_values,l)
    gs_per_site_list.append(gs)
print(gs_per_site_list)

# Plot of the Ground-state energy per site vs number of sites
plt.plot(site_list[1:],gs_per_site_list[1:],'-')
plt.xlabel('Number of sites L')
plt.ylabel('Ground-state energy per site E_0/L')
plt.title('Ground-state energy per site vs number of sites')
plt.show()
# The ground state per site converges until the expected values E_0=-0.443147 .

######## After this the code is draft ##########

###  Define model parameters ### 
#To define a model we need for instance a list of string defining the coupling, the local 
#operators from the spin_basis and a corresponding site-coupling list.  

# Example of the XXZ model in a magnetic field
L = 10 #Size of the system
Jxy = 1.0 #np.sqrt(2.0) # xy interaction strength
Jzz = 1.0 #1.0 # zz interaction strength
# Note: Jzz=0 => XX Hamiltonian and Jzz=1 => XXX Hamiltonian
hz = 0.0 #1.0/np.sqrt(3.0) # External field along the z direction
gamma = math.acos(Jzz)


Jzz_list = [[Jzz,i,i+1] for i in range(L-1)]
Jxy_list = [[Jxy ,i,i+1] for i in range(L-1)]
# Periodic boundary conditions
#Jzz_list.append([Jzz,L-1,0])
#Jxy_list.append([Jxy,L-1,0])
hz_list = [[hz,i] for i in range(L)]


H=np.zeros((2**(L),2**(L)),dtype='complex128')
for _,value in enumerate(Jzz_list):
    t_1 = globalize_op(spin_op["sigma_z"],value[1],L)
    t_2 = globalize_op(spin_op["sigma_z"],value[2],L)
    H += (np.dot(t_2,t_1))*value[0]#+np.identity(2**L,dtype='complex128'))*value[0]
for _,value in enumerate(Jxy_list):
    t_1 = globalize_op(spin_op["sigma_x"],value[1],L)
    t_2 = globalize_op(spin_op["sigma_x"],value[2],L)
    H += np.dot(t_2,t_1)*value[0]
    t_1 = globalize_op(spin_op["sigma_y"],value[1],L)
    t_2 = globalize_op(spin_op["sigma_y"],value[2],L)
    H += np.dot(t_2,t_1)*value[0]

for _,value in enumerate(hz_list):
    t_1=globalize_op(spin_op["sigma_z"],value[1],L)
    H+=-t_1*value[0]

H=-H

# Diagonalization
eig_values,eig_vectors = np.linalg.eig(H)
print('Eigenvalues =',eig_values,'Eigenvectors =',eig_vectors)


observable = np.identity(2**L,dtype='complex128')
print(observable)
#print(thermal_exp_value(eig_values,eig_vectors,observable,300.0))

plt.plot(ground_state(eig_values,eig_vectors)**2,'-')
plt.xlabel('States in the basis {|0>,|1>,...|2**L-1>}')
plt.ylabel('amplitudes')
plt.title('Probability amplitudes of the ground state')
plt.show()


plt.plot(eig_values,'-')
plt.xlabel('States in the basis {|0>,|1>,...|2**L-1>}')
plt.ylabel('amplitudes')
plt.title('Probability amplitudes of the ground state')
plt.show()


# Concatenation to create the Hamiltonian
H = np.zeros((2**(L),2**(L)),dtype='complex128')
for _,value in enumerate(Jzz_list):
    t_1 = globalize_op(spin_op["sigma_z"],value[1],L)
    t_2 = globalize_op(spin_op["sigma_z"],value[2],L)
    H += np.dot(t_2,t_1)*value[0]
    
for _,value in enumerate(Jxy_list):
    t_1 = globalize_op(spin_op["sigma_-"],value[1],L)
    t_2 = globalize_op(spin_op["sigma_+"],value[2],L)
    H+=(np.dot(t_2,t_1)+np.transpose(np.conjugate(np.dot(t_2,t_1))))*value[0]
    
for _,value in enumerate(hz_list):
    t_1 = globalize_op(spin_op["sigma_z"],value[1],L)
    H += t_1*value[0]
#print(H)
"""
