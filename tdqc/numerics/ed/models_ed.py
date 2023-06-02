#!/usr/bin/env python
# coding: utf-8
from scipy.linalg import expm
import numpy as np
from numpy import linalg
import scipy as sci
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
    Return the tensor product of the local operator and identity operators such that the local operator applies on site number site.
    L is the total number of sites in the system on which we want to apply the global operator.
    '''
    tensor_0 = np.identity(1,dtype = 'complex128')
    for i in range(0,site,1):
        tensor_0 = np.kron(tensor_0,np.identity(2,dtype='complex128'))
    tensor_0 = np.kron(tensor_0,local_op)
    for i in range(site+1,L,1):
        tensor_0 = np.kron(tensor_0,np.identity(2,dtype='complex128'))
    return tensor_0


class Model(object):
    # Class attribute
    
    def __init__(self, name, model_hamiltonian):
        self.name = name
        self.model_hamiltonian = model_hamiltonian
        self.__eig_values = None
        self.__eig_vectors = None
        self.__ground_states = None
        self.__hamiltonian = None

    def calculate_ground_states(self, eig_values, eig_vectors):
        length_vector = eig_vectors.shape[0]
        min_indices = np.asarray(abs(eig_values-eig_values.min())<10**(-12)).nonzero() #np.where(eig_values == eig_values.min())
        min_indices = np.asarray(min_indices)[0]
        ground_states = np.zeros([length_vector,min_indices.shape[0]],complex)
        for idx, value in enumerate(min_indices):
            eig_vector = eig_vectors[:,value]
            eig_vector = eig_vector[:]
            ground_states[:, idx] = eig_vector
        return ground_states

    def parametrize_hamiltonian(self, *parameter):
        fonction = self.model_hamiltonian
        self.__hamiltonian = fonction(*parameter)
        eig_values, eig_vectors = np.linalg.eigh(self.__hamiltonian) 
        self.__eig_values = eig_values
        self.__eig_vectors = eig_vectors
        self.__ground_states = self.calculate_ground_states(eig_values,eig_vectors)

    @property
    def hamiltonian(self,):
        if self.__hamiltonian == None:
            raise ValueError("The method parametrize_hamiltonian need to be run before in order to get the hamiltonian.")
        return self.__hamiltonian

    @property
    def eig_values(self,):
        if self.__eig_values == None:
            raise ValueError("The method parametrize_hamiltonian need to be run before in order to get the eig_values.")
        return self.__eig_values

    @property
    def eig_vectors(self,):
        if self.__eig_vectors == None:
            raise ValueError("The method parametrize_hamiltonian need to be run before in order to get the eig_vectors.")
        return self.__eig_vectors

    @property
    def ground_states(self,):
        if self.__ground_states == None:
            raise ValueError("The method parametrize_hamiltonian need to be run before in order to get the ground_states.")
        return self.__ground_states

    @classmethod
    def class_method(cls):
        return cls, "is class of mathematical models of Hamiltonian."



# Models XXZ
def hamiltonian_xxz(L, Jxy, Jzz, PDB=False):
    Jzz_list = [[Jzz, i, i+1] for i in range(L-1)]
    Jxy_list = [[Jxy ,i, i+1] for i in range(L-1)]
    # Periodic boundary conditions
    if PDB:
        Jzz_list.append([Jzz, L-1, 0])
        Jxy_list.append([Jxy, L-1, 0])
    H = np.zeros((2**(L), 2**(L)), dtype='complex128')
    for _,value in enumerate(Jzz_list):
        t_1 = globalize_op(h_bar/2.0 * spin_op["sigma_z"], value[1], L)
        t_2 = globalize_op(h_bar/2.0 * spin_op["sigma_z"], value[2], L)
        H += np.dot(t_2,t_1) * value[0]
    for _,value in enumerate(Jxy_list):
        t_1 = globalize_op(h_bar/2.0 * spin_op["sigma_x"], value[1], L)
        t_2 = globalize_op(h_bar/2.0 * spin_op["sigma_x"], value[2], L)
        H += np.dot(t_2, t_1) * value[0]
        t_1 = globalize_op(h_bar/2.0 * spin_op["sigma_y"], value[1], L)
        t_2 = globalize_op(h_bar/2.0 * spin_op["sigma_y"], value[2], L)
        H += np.dot(t_2,t_1)*value[0]
    return H
xxz_model = Model("xxz_model", hamiltonian_xxz)

def hamiltonian_lri(L,J,alpha,m_x,m_z):
    """
    hamiltonian_lri returns the Hamiltonian matrix of a one-dimensional spin chain with L sites, 
    where each site is represented by a two-level quantum system or a spin-1/2 particle.
    The Hamiltonian includes a longitudinal magnetic field along the z-axis and a
    transverse magnetic field along the x-axis. 

    L: an integer that specifies the number of sites in the spin chain
    J: a floating-point number that controls the strength of the spin-spin interactions
    alpha: a floating-point number that controls the decay of the interactions with distance
    m_x: a floating-point number that specifies the strength of the transverse magnetic field along the x-axis
    m_z: a floating-point number that specifies the strength of the longitudinal magnetic field along the z-axis
    return H: 2^L x 2^L complex array representing the Hamiltonian matrix in the computational basis. 
    """ 
    # Non Periodic Boundary Conditions
    list_glob_operators =  [None] * L
    # Create the list of global operators
    for site in range(0, L, 1):
        list_glob_operators[site] = globalize_op(spin_op["sigma_x"], site, L)    
    H = np.zeros((2**(L), 2**(L)), dtype='complex128')
    for j in range(0, L-1):
        for k in range(j+1, L):
            H += J*((k-j)**(-alpha)) * np.dot(list_glob_operators[j], list_glob_operators[k])
    for j in range(0,L,1):
        H += m_x * list_glob_operators[j] + m_z * globalize_op(spin_op["sigma_z"], j, L)
    return H
lri_model = Model("lri_model", hamiltonian_lri)

def hamiltonian_trans_ising(L, J, h):
    '''
    Hamiltonian of the transverse field Ising model 
    taken from PhysRevLett.111.147205 equation 9. 

    L: an integer that specifies the number of sites in the spin chain
    J: a floating-point number that controls the strength of the spin-spin interactions
    alpha: a floating-point number that controls the decay of the interactions with distance
    h: a floating-point number corresponding to the strength of the transverse field
    return H: 2^L x 2^L complex array representing the Hamiltonian matrix in the computational basis. 
    '''
    # Non Periodic Boundary Conditions
    list_glob_operators =  [None] * L
    # Create the list of global operators
    for site in range(0, L, 1):
        list_glob_operators[site] = globalize_op(spin_op["sigma_x"], site, L)    
    H = np.zeros((2**(L), 2**(L)), dtype='complex128')
    for j in range(0, L-1, 1):
        H += -J * np.dot(list_glob_operators[j], list_glob_operators[j+1])
    for j in range(0, L, 1):
        H += h * globalize_op(spin_op["sigma_z"], j, L)
    return H

trans_ising_model = Model("trans_ising_model", hamiltonian_trans_ising)

class State(object):
    '''
    init_vec_state: type <class 'numpy.ndarray'>
    '''
    def __init__(self, init_vec_state: np.ndarray):
        self.vec_state = init_vec_state
        self.dimension = init_vec_state.size
        self.n_sites = int(log2(self.dimension))
        self.vec_state_real = init_vec_state.real
        self.vec_state_imag = init_vec_state.imag
        self._density_mat = np.tensordot(np.conjugate(self.vec_state), self.vec_state, axes=0)
    """
    @property
    def vec_state(self):
        return self.vec_state
   
    @vec_state.setter
    def vec_state(self, value):
        self._vec_state = value
        # Also updates the density_mat an its real and imaginary parts.
        self._density_mat = np.tensordot(np.conjugate(self.vec_state), self.vec_state, axes=0)
    """

    def time_step_ed(self, model, delta_t, imaginary=False, h_bar=h_bar):
        '''
        Time evolution of a system after a quench using exact diagonalization. 
        It makes the state initial_state evolve according to the Hamiltonian of the model for a time delta_t.
        '''
        # Input to simulate the imaginary time evolution, by default, it is the real time evolution.
        if imaginary:
            delta_t = -1j * delta_t
        init_vec_state = self.vec_state
        eig_values, eig_vectors = model.eig_values, model.eig_vectors
        
        new_vec_state = np.dot(expm(-1j * delta_t * model.hamiltonian), init_vec_state)
        self.vec_state = new_vec_state
        self.vec_state_real = new_vec_state.real
        self.vec_state_imag = new_vec_state.imag

    def get_vector_state(self,):
        return self.vec_state

    def get_density_matrix(self,):
        self._density_mat = np.tensordot(np.conjugate(self.vec_state), self.vec_state, axes=0)
        return self._density_mat
    
    @classmethod
    def class_method(cls):
        return cls, "is class of mathematical models of quantum systems composed of two-level subsystems."    
