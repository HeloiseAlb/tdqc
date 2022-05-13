#!/usr/bin/env python
# coding: utf-8


import numpy as np
from math import log2
import cmath



h_bar = 1 # 1.054571817*10**(-34) # in J.s

spin_op= {
    "I": np.array([[1+0j,0+0j],[0+0j,1+0j]],dtype = 'complex128'),
    "sigma_x": np.array([[0+0j,1+0j],[1+0j,0+0j]],dtype = 'complex128'),
    "sigma_y": np.array([[0+0j,-1j],[1j,0+0j]],dtype = 'complex128'),
    "sigma_z": np.array([[1+0j,0+0j],[0+0j,-1+0j]],dtype = 'complex128'),
    "sigma_+": np.array([[0+0j,1+0j],[0+0j,0+0j]],dtype = 'complex128'),
    "sigma_-": np.array([[0+0j,0+0j],[-1+0j,0+0j]],dtype = 'complex128')} 

def globalize_op(local_op,site,L):
    '''
    Return the tensor product of the local operator and identity operators such that the local operator applies
    on site.
    '''
    tensor_0 = np.identity(1,dtype = 'complex128')
    for i in range(0,site,1):
        tensor_0 = np.kron(tensor_0,np.identity(2,dtype='complex128'))
    tensor_0 = np.kron(tensor_0,local_op)
    for i in range(site+1,L,1):
        tensor_0 = np.kron(tensor_0,np.identity(2,dtype='complex128'))
    return tensor_0

def ground_states(eig_values,eig_vectors):
    length_vector = eig_vectors.shape[0]
    min_indices = np.asarray(abs(eig_values-eig_values.min())<10**(-12)).nonzero() #np.where(eig_values == eig_values.min())
    min_indices = np.asarray(min_indices)[0]
    ground_states = np.zeros([length_vector,min_indices.shape[0]],complex)
    for idx, value in enumerate(min_indices):
        eig_vector = eig_vectors[:,value]
        eig_vector = eig_vector[:]
        ground_states[:, idx] = eig_vector
    return ground_states


class Model(object):
    # Class attribute
    
    def __init__(self, name, model_hamiltonian):
        self.name = name
        self.model_hamiltonian = model_hamiltonian
        self.eig_values = None
        self.eig_vectors = None
        self.ground_states = None

    def parametrize_hamiltonian(self, *parameter):
        fonction = self.model_hamiltonian
        self.hamiltonian = fonction(*parameter)
        eig_values,eig_vectors = np.linalg.eig(self.hamiltonian)
        self.eig_values = eig_values
        self.eig_vectors = eig_vectors
        self.ground_states = ground_states(eig_values,eig_vectors)

    @classmethod
    def class_method(cls):
        return cls, "is class of mathematical models of Hamiltonian."



# Models XXZ
def hamiltonian_xxz(L,Jxy,Jzz,PDB=True):
    Jzz_list = [[Jzz,i,i+1] for i in range(L-1)]
    Jxy_list = [[Jxy ,i,i+1] for i in range(L-1)]
    # Periodic boundary conditions
    if PDB:
        Jzz_list.append([Jzz,L-1,0])
        Jxy_list.append([Jxy,L-1,0])
    H = np.zeros((2**(L),2**(L)),dtype='complex128')
    for _,value in enumerate(Jzz_list):
        t_1 = globalize_op(h_bar/2.0*spin_op["sigma_z"],value[1],L)
        t_2 = globalize_op(h_bar/2.0*spin_op["sigma_z"],value[2],L)
        H += (np.dot(t_2,t_1))*value[0]
    for _,value in enumerate(Jxy_list):
        t_1 = globalize_op(h_bar/2.0*spin_op["sigma_x"],value[1],L)
        t_2 = globalize_op(h_bar/2.0*spin_op["sigma_x"],value[2],L)
        H += np.dot(t_2,t_1)*value[0]
        t_1 = globalize_op(h_bar/2.0*spin_op["sigma_y"],value[1],L)
        t_2 = globalize_op(h_bar/2.0*spin_op["sigma_y"],value[2],L)
        H += np.dot(t_2,t_1)*value[0]
    return H
xxz_model = Model("xxz_model",hamiltonian_xxz)



class State(object):
    '''
    init_vec_state: type <class 'numpy.ndarray'>
    '''
    def __init__(self, init_vec_state):
        self._vec_state = init_vec_state
        self.dimension = init_vec_state.size
        self.n_sites = int(log2(self.dimension))
        self.vec_state_real = init_vec_state.real
        self.vec_state_imag = init_vec_state.imag
        self._density_mat = np.tensordot(np.conjugate(self.vec_state), self.vec_state, axes=0)
    
    @property
    def vec_state(self):
        return self._vec_state
   
    @vec_state.setter
    def vec_state(self, value):
        self._vec_state = value
        # Also updates the density_mat an its real and imaginary parts.
        self._density_mat = np.tensordot(np.conjugate(self.vec_state), self.vec_state, axes=0)

    def time_step_ed(self, model, delta_t, imaginary=False, h_bar=h_bar):
        '''
        Time evolution of a system after a quench using exact diagonalization. 
        It makes the state initial_state evolve according to the Hamiltonian of the model for a time delta_t.
        '''
        if imaginary:
            delta_t = -1j*delta_t
        init_vec_state = self.vec_state
        eig_values,eig_vectors = model.eig_values, model.eig_vectors
        new_vec_state = np.zeros(self.dimension,dtype='complex128')
        for index, vector in enumerate(eig_vectors.T):
            energy = eig_values[index]
            projection = np.dot(vector,np.transpose(init_vec_state))
            new_vec_state += cmath.exp(-1j*energy*delta_t/h_bar)*projection*vector
        norm = np.linalg.norm(new_vec_state)
        if norm!=0:
            new_vec_state = new_vec_state/norm
        self.vec_state = new_vec_state
        self.vec_state_real = new_vec_state.real
        self.vec_state_imag = new_vec_state.imag

    def get_density_matrix(self):
        return self._density_mat
    
    @classmethod
    def class_method(cls):
        return cls, "is class of mathematical models of quantum systems composed of two-level subsystems."    
"""
psi_0 = State(np.zeros([2**4],dtype='complex128'))
L = 4
Jzz = 1.0
Jxy = 1.0
model = xxz_model
model.parametrize_hamiltonian(*[L,Jxy,Jzz])

print(type(psi_0.vec_state))
psi_0.time_step_ed(xxz_model, 1)
print(psi_0.vec_state)
print(psi_0.get_state_format_ml())
"""


