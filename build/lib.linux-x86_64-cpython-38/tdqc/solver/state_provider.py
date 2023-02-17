import numpy as np
from math import log2
from scipy.linalg import expm
from typing import Optional
from tdqc.interfaces.solver import Solver

spin_op= {
    "I": np.array([[1+0j,0+0j],[0+0j,1+0j]],dtype = 'complex128'),
    "sigma_x": np.array([[0+0j,1+0j],[1+0j,0+0j]],dtype = 'complex128'),
    "sigma_y": np.array([[0+0j,-1j],[1j,0+0j]],dtype = 'complex128'),
    "sigma_z": np.array([[1+0j,0+0j],[0+0j,-1+0j]],dtype = 'complex128'),
    "sigma_+": np.array([[0+0j,1+0j],[0+0j,0+0j]],dtype = 'complex128'),
    "sigma_-": np.array([[0+0j,0+0j],[-1+0j,0+0j]],dtype = 'complex128')}

def tensor_prod(*arg):
    """tensor_prod(a1, a2) = np.kron(a1, a2).
    tensor_prod(a1, a2, ..., an) = np.kron(tensor_prod(a1, ..., an-1), an)
    """
    res = arg[0]
    for i in range(1, len(arg)):
        res = np.kron(res, arg[i])
    #  res = arg[-1]
    #  for i in range(1, len(arg)):
    #      res = np.kron(res, arg[len(arg) - i - 1])
    return res

def globalize_op(local_op,site,L):
    """" Return the tensor product of the local operator and identity operators such that the local operator applies on site number site.
    L is the total number of sites in the system on which we want to apply the global operator.
    """
    tensor_0 = np.identity(1,dtype = 'complex128')
    for i in range(0,site,1):
        tensor_0 = np.kron(tensor_0,np.identity(2,dtype='complex128'))
    tensor_0 = np.kron(tensor_0,local_op)
    for i in range(site+1,L,1):
        tensor_0 = np.kron(tensor_0,np.identity(2,dtype='complex128'))
    return tensor_0


class StateProvider(Solver):
    """
    Class implementing a state provider solver. It is used to provide a state from a parameterized circuits to check if the deep_q_learning solver 
    will reach a reachable state. There are two modes: 
    - "state_copier": we can load a state which become the target state,
    - "circuit_copier": we can load a set of parameters to parametize the circuit. Then the target state is the output state of the circuit.
    """

    def __init__(self,):
        super().__init__()
        self.__check_validity__()

    def load_settings(self, settings):
        """
        Initialize settings stored in local variable self.__settings
        """
        if not "mode" in settings:
            raise ValueError("Error loading state_provider-solver settings, 'mode' parameter not found")
        self.__mode = settings["mode"]
        if self.__mode == "state_copier":
            if not "state_to_copy" in settings:
                raise ValueError("""Error loading state_provider-solver settings, 'state_to_copy' parameter not found.
                If the mode is state_copier then load a state_to_copy.""")
            self.__state_to_copy = settings["state_to_copy"]
            self.__n_sites = self.__state_to_copy.n_sites
        elif self.__mode == "circuit_copier":
            if not "n_sites" in settings:
                raise ValueError("""Error loading state_provider-solver settings, 'n_sites' parameter not found.""")
            self.__n_sites = settings["n_sites"]
            if not "initial_state" in settings:
                raise ValueError("""Error loading state_provider-solver settings, 'initial_state' parameter not found.
                If the mode is circuit_copier then load a initial_state.""")
            
            if not "seed_initial_state" in settings:
                settings["seed_initial_state"] = None    
            self.set_initial_state(settings["seed_initial_state"],settings["initial_state"])
            # self.__initial_state = settings["initial_state"]
            if not "jx_angle_list" in settings:
                raise ValueError("""Error loading state_provider-solver settings, 'jx_angle_list' parameter not found.
                If the mode is circuit_copier then load a jx_angle_list.""")
            self.__jx_gate_list = settings["jx_angle_list"]
            if not "hx_angle_list" in settings:
                raise ValueError("""Error loading state_provider-solver settings, 'hx_angle_list' parameter not found.
                If the mode is circuit_copier then load a hx_angle_list.""")
            self.__hx_gate_list = settings["hx_angle_list"]
            if not "hz_angle_list" in settings:
                raise ValueError("""Error loading state_provider-solver settings, 'hz_angle_list' parameter not found.
                If the mode is circuit_copier then load a hz_angle_list.""")
            self.__hz_gate_list = settings["hz_angle_list"]
            self.__n_steps = self.__jx_gate_list.size
            if not "gate_order" in settings:
                raise ValueError("""Error loading state_provider-solver settings, 'gate_order' parameter not found.
                If the mode is circuit_copier then load a gate_order.""")
            self.__gate_order = settings["gate_order"]
            self.__n_qubits =  int(self.__hx_gate_list.size/self.__n_steps)
            if not "alpha" in settings:
                raise ValueError("""Error loading state_provider-solver settings, 'alpha' parameter not found.
                If the mode is circuit_copier then load a alpha.""")
            self.__alpha = settings["alpha"]        
            self.set_coupling_matrix()
        self.__final_state = None
            
    
    def solve(self,)-> None:
        # This method runs the time evolution and stores the list of the vec_state in self.__time_evolution.  
        if self.__mode == "state_copier":
            self.__final_state = self.__state_to_copy.get_vector_state()
        elif self.__mode == "circuit_copier":
            self.__final_state = self.apply_gate_sequence()
        else: 
            raise ValueError("The selected mode {} does not exist.".format(self.__mode))

    def apply_gate_sequence(self,)-> np.ndarray:
        """ Apply the sequence of gates onto the initial state and return the final state. """
        # Define the universal quantum gate set used in Markus article. 
        U_x = lambda theta : expm(-1j*theta*spin_op['sigma_x'])
        U_z = lambda theta : expm(-1j*theta*spin_op['sigma_z'])
        sum_U_xx = self.coupling_matrix 
        U_xx = lambda theta : expm(-1j*theta*sum_U_xx)
        state = self.__initial_state
        # Those gate lists are in fact lists of angles.
        jx_angle_list = self.__jx_gate_list
        hx_angle_list = self.__hx_gate_list
        hz_angle_list = self.__hz_gate_list
        for step in range(0,self.__n_steps,1):
            state = np.dot(U_xx(jx_angle_list[step]),state)
            #print('state after U_xx:{} is {}'.format(U_xx(jx_angle_list[step]),state))
            for site in range(0,self.__n_sites,1):
                if self.__gate_order == "xz":
                    U_x_site = globalize_op(U_x(hx_angle_list[step][site]),site,self.__n_sites)
                    state = np.dot(U_x_site,state)
                    U_z_site = globalize_op(U_z(hz_angle_list[step][site]),site,self.__n_sites)
                    state = np.dot(U_z_site,state)
                elif self.__gate_order == "zx":
                    U_z_site = globalize_op(U_z(hz_angle_list[step][site]),site,self.__n_sites)
                    state = np.dot(U_z_site,state)
                    U_x_site = globalize_op(U_x(hx_angle_list[step][site]),site,self.__n_sites)
                    state = np.dot(U_x_site,state)
        return state

    def set_coupling_matrix(self,)-> None:
        dim = int(2**self.__n_sites)
        list_glob_operators =  [None] * self.__n_sites
        for qubit in range(0,self.__n_sites,1):
            list_glob_operators[qubit] = globalize_op(spin_op["sigma_x"],qubit, self.__n_qubits)  
        coupling_matrix = np.zeros((dim,dim),dtype='complex128')
        for l in range(0,self.__n_sites,1):
            matrix_1 = list_glob_operators[l] 
            for k in range(l+1,self.__n_sites,1):
                matrix_2 = list_glob_operators[k]
                coupling_matrix += np.dot(matrix_1,matrix_2)/(k-l)**self.__alpha
        self.coupling_matrix = coupling_matrix
        
    def get_rho_target(self,)-> np.ndarray:
        if not isinstance(self.__final_state, np.ndarray):
            raise ValueError("The method solve need to be run before in order to get the target_state")
        rho_target = np.tensordot(np.conjugate(self.__final_state), self.__final_state, axes=0)
        return rho_target

    def get_state_target(self,)-> np.ndarray:
        if not isinstance(self.__final_state, np.ndarray):
            raise ValueError("The method solve need to be run before in order to get the target_state")
        return self.__final_state
             
    def set_initial_state(self, seed: Optional[int], initial_state: str)-> None:
        if initial_state == 'random_product_state':
            np.random.seed(seed)
            #  randomly directed vector on the unit sphere
            #  f(theta)g(phi)dtheta dphi = sin(theta)/4pi
            #  g(phi) = 1/2pi, f(theta) = sin(theta)/2
            #  -> F(theta) = (1-cos(theta))/2
            #  F^-1(u) = arccos(1 - 2u)
            phis = [2*pi*np.random.rand() for _ in range(self.__n_sites)]
            thetas = [np.arccos(1-2*np.random.rand()) for _ in
                      range(self.__n_sites)]
            spinors = [np.array([np.cos(theta/2),
                                 np.exp(1j*phi)*np.sin(theta/2)])
                       for phi, theta in zip(phis, thetas)]
            rstate = tensor_prod(*spinors)
            self.state_real = rstate.real
            self.state_imag = rstate.imag
        elif initial_state == 'ferro':
            self.state_real = np.zeros(2**self.__n_sites)
            self.state_real[0] = 1.0
            self.state_imag = np.zeros(2**self.__n_sites)
        elif initial_state == 'antiferro':
            self.state_imag = np.zeros(2**self.__n_sites)
            spinors = [np.array([1.0, 0.0]) if _ % 2 == 0
                       else np.array([0.0, 1.0]) for _ in range(self.__n_sites)]
            self.state_real = tensor_prod(*spinors)
        else:
            raise NotImplementedError(f'Initial state of type {initial_state} '
                                      'not implemented.')
        self.__initial_state = self.state_real + 1j*self.state_imag

