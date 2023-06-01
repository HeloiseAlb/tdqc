import numpy as np
# from math import log2
from tdqc.interfaces.solver import Solver
from tdqc.numerics.ed.models_ed import State, Model
from scipy.linalg import expm

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

class AdiaStatePrepa(Solver):
    """
    Class implementing the adiabatic state preparation.
    """

    def __init__(self,):
        super().__init__()
        self.__check_validity__()

    def load_settings(self, settings):
        """
        Initialize settings stored in local variable self.__settings
        """
        if not "model_0" in settings:
            raise ValueError("Error loading asp-solver settings, 'model_0' parameter not found")
        self.__model_0 = settings["model_0"]
        """
        if not "model_f" in settings:
            raise ValueError("Error loading asp-solver settings, 'model_f' parameter not found")
        self.__model_f = settings["model_f"]
        """
        if not "t_initial" in settings:
            raise ValueError("Error loading asp-solver settings, 't_initial' parameter not found")
        self.__t_initial = settings["t_initial"]
        if not "t_final" in settings:
            raise ValueError("Error loading asp-solver settings, 't_final' parameter not found")
        self.__t_final = settings["t_final"]
        if not "n_steps" in settings:
            raise ValueError("Error loading asp-solver settings, 'n_step' parameter not found")
        self.__n_steps = settings["n_steps"]
        if not "system_class" in settings:
            raise ValueError("Error loading asp-solver settings, 'system_class' parameter not found")
        self.__system_class = settings["system_class"]
        if not "ham_params" in settings:
            raise ValueError("Error loading asp-solver settings, 'ham_params' parameter not found")
        self.ham_params = settings["ham_params"]
        if not "n_sites" in settings:
            raise ValueError("Error loading asp-solver settings, 'n_sites' parameter not found")
        self.__n_sites = settings["n_sites"]

        # Define the initial Hamiltonian H_0 and the final Hamiltonian H_f
        #self.H_0 = self.__model_0.model_hamiltonian
        #self.H_f = self.__model_f.model_hamiltonian
        #self.__time_evolution = None
        self.__final_state = None 
        self.set_initial_state()
        self.set_coupling_matrix()

    def set_initial_state(self,)-> None:
        # The initial state is the GS of the initial hamiltonian H_0.
        # Here, I consider the ground state is non-degenerate.
        # To do: Code the case where it is degenerate. 
        ground_states = self.__model_0.ground_states 
        init_vec_state = np.array(ground_states,dtype='complex128')
        norm = np.linalg.norm(init_vec_state)
        init_vec_state = init_vec_state / norm
        self.__initial_state = State(init_vec_state)
    
    def set_coupling_matrix(self,)-> None:
        if self.__system_class == 'LongRangeIsing':
            dim = int(2**self.__n_sites)
            list_glob_operators =  [None] * self.__n_sites
            for site in range(0,self.__n_sites,1):
                list_glob_operators[site] = globalize_op(spin_op["sigma_x"],site, self.__n_sites)  
            coupling_matrix = np.zeros((dim,dim),dtype='complex128')
            for l in range(0,self.__n_sites,1):
                matrix_1 = list_glob_operators[l] 
                for k in range(l+1,self.__n_sites,1):
                    matrix_2 = list_glob_operators[k]
                    coupling_matrix += np.dot(matrix_1,matrix_2)/(k-l)**self.alpha
            self.coupling_matrix = coupling_matrix
        elif self.__system_class == 'TransIsing':
            dim = int(2**self.__n_sites)
            list_glob_operators =  [None] * self.__n_sites
            for site in range(0,self.__n_sites,1):
                list_glob_operators[site] = globalize_op(spin_op["sigma_x"],site, self.__n_sites)  
            coupling_matrix = np.zeros((dim,dim),dtype='complex128')
            for l in range(0,self.__n_sites-1,1):
                matrix_1 = list_glob_operators[l] 
                matrix_2 = list_glob_operators[l+1]
                coupling_matrix += np.dot(matrix_1,matrix_2)
            self.coupling_matrix = -coupling_matrix

    @property
    def time_evolution(self):
        # It returns the amplitudes of the time evolution.
        return self.__time_evolution
    
    """    
    @property
    def H_0(self):
        return self.__H_0

    @property
    def H_f(self):
        return self.__H_f
    """

    # Define the time-dependent Hamiltonian H(t) using a linear schedule
    ### I need to implement the one using a non linear schedule.
   
    def H(self, t):
        T = self.__t_final
        H_t = (1 - t/T)*self.__model_0.model_hamiltonian() + (t/T)*self.__model_f.model_hamiltonian()
        return H_t
    """    
    # Define the time evolution operator U(t, T)
    def U(self, t, T):
        return expm(-1j*self.H(t, T))
    """
    def solve(self,)-> None:
        # This method runs the time evolution and stores the list of the vec_state in self.__time_evolution.  
        state_t_n = self.__initial_state.get_vector_state()
        t_initial = self.__t_initial
        t_final = self.__t_final
        n_steps = self.__n_steps
        step = (t_final-t_initial)/n_steps
        
        n_sites = self.__n_sites
        site_list = [l for l in range(1,n_sites,1)]
        t_list = [t for t in np.linspace(t_initial,t_final,n_steps)]
        time_evolution = np.zeros([n_steps+1,2**n_sites],dtype='complex128') # [None] * int((t_max-t_min)/step) #np.zeros([int((t_max-t_min)/step)])
        inv_temperature = 1
        for idx, t_n in enumerate(t_list):
            time_evolution[idx, :] = state_t_n.reshape(-1)
            coupling_matrix_angle, hx_angle, hz_angle = self.define_gate_angles(t_n)
            state_t_n = self.apply_gate_sequence(state_t_n, coupling_matrix_angle, hx_angle, hz_angle)
        time_evolution[-1,:] = state_t_n.reshape(-1)
        self.__time_evolution = time_evolution
        self.__final_state = State(state_t_n) # It is an instance of the class State()

    @property
    def final_state(self):
        return self.__final_state
        
    def get_rho_target(self,)-> np.ndarray:
        if (self.__final_state == None):
            raise ValueError("The method solve need to be run before in order to get the target_state")
        target = self.__final_state.get_density_matrix()
        return target

    def get_state_target(self,)-> np.ndarray:
        if (self.__final_state == None):
            raise ValueError("The method solve need to be run before in order to get the target_state")
        target = self.__final_state.get_vector_state()
        return target

    def define_gate_angles(self, t_n):
        # The gates are defined to realize the Trotterization of the Hamiltonian.
        # I need to implement difference way to weight the H_0 and H_f. For 
        # the moment, it is only possible linearly. 
        if self.__system_class == 'LongRangeIsing':
            # To be defined according to the choice of H_0 for 
            # LongRangeIsing model. 
            pass
            # return coupling_matrix_angle, hx_angle, hz_angle
        elif self.__system_class == 'TransIsing':
            coupling_matrix_angle = self.ham_params['J']
            hx_angle = self.ham_params['g'] * t_n/self.__t_final
            return coupling_matrix_angle, hx_angle, None
        else:
            raise NotImplementedError('Trotter sequence not implemented'
                                          ' for your system_class. Only implemented'
                                          ' for TransIsing and LongRangeIsing.')
        
    def apply_gate_sequence(self, state, coupling_matrix_angle, hx_angle, hz_angle = None)-> np.ndarray:
        """ Apply the sequence of gates onto the initial state and return the final state. """
        # Define the universal quantum gate set used in Markus article. 
        U_x = lambda theta : expm(-1j*theta*spin_op['sigma_x'])
        sum_U_xx = self.coupling_matrix 
        U_xx = lambda theta : expm(-1j*theta*sum_U_xx)
        # Those gate lists are in fact lists of angles.
        state = np.dot(U_xx(coupling_matrix_angle),state)
        #print('state after U_xx:{} is {}'.format(U_xx(jx_angle_list[step]),state))
        if hz_angle != None:       
            U_z = lambda theta : expm(-1j*theta*spin_op['sigma_z'])
            for site in range(0,self.__n_sites,1):
                if self.__gate_order == "xz":
                    U_x_site = globalize_op(U_x(hx_angle),site,self.__n_sites)
                    state = np.dot(U_x_site,state)
                    U_z_site = globalize_op(U_z(hz_angle),site,self.__n_sites)
                    state = np.dot(U_z_site,state)
                elif self.__gate_order == "zx":
                    U_z_site = globalize_op(U_z(hz_angle),site,self.__n_sites)
                    state = np.dot(U_z_site,state)
                    U_x_site = globalize_op(U_x(hx_angle),site,self.__n_sites)
                    state = np.dot(U_x_site,state)
        else:
            for site in range(0,self.__n_sites,1):
                U_x_site = globalize_op(U_x(hx_angle),site,self.__n_sites)
                state = np.dot(U_x_site,state)
        return state
                