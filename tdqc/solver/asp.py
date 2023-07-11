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
        if not "model_f" in settings:
            raise ValueError("Error loading asp-solver settings, 'model_f' parameter not found")
        self.__model_f = settings["model_f"]
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
        self.__list_fidelities = None
        self.__list_gaps = None
        self.__delta_t = (self.__t_final-self.__t_initial)/self.__n_steps
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
        dim = int(2**self.__n_sites)
        list_glob_operators =  [None] * self.__n_sites
        if self.__system_class == 'LongRangeIsing':
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
        H_t = (1 - t/T)*self.__model_0.hamiltonian + (t/T)*self.__model_f.hamiltonian
        return H_t

    def solve(self,)-> None:
        # This method runs the time evolution and stores the list of the vec_state in self.__time_evolution.  
        state_t_n = self.__initial_state.get_vector_state()

        step = (self.__t_final - self.__t_initial) / self.__n_steps
        site_list = [l for l in range(1, self.__n_sites, 1)]
        t_list = [t for t in np.linspace(self.__t_initial, self.__t_final, self.__n_steps)]
        time_evolution = np.zeros([self.__n_steps + 1, 2**self.__n_sites], dtype='complex128') 
        inv_temperature = 1
        for idx, t_n in enumerate(t_list):
            time_evolution[idx, :] = state_t_n.reshape(-1)
            coupling_matrix_angle, hx_angle, hz_angle = self.define_gate_angles(t_n)
            state_t_n = self.apply_gate_sequence(state_t_n, coupling_matrix_angle, hx_angle, hz_angle)
        time_evolution[-1,:] = state_t_n.reshape(-1)
        self.__time_evolution = time_evolution
        self.__final_state = State(state_t_n) # It is an instance of the class State()
    
    def generate_data_files(self,):
        if (self.__final_state == None):
            # If the method solve have not run yet. 
            self.solve()
        parametername = 'ASP'+str(self.__n_sites)+'n_steps'+str(self.__n_steps)+'t_final'+str(self.t_final)
        self.save_best_encountered_actions('json',
                                                'best_gate_sequence'+parametername+'.json')
        
        # Generate the file with the time evolution amplitudes.
        try:
            amplitude_filename = 'amplitude_'+parametername+'.npy'
            with open(amplitude_filename, 'wb') as f:
                np.save(f, self.time_evolution)
        except Exception as e:
            print(amplitude_filename+' could not be saved.')
            print('--->', e)
        


    @property
    def final_state(self):
        return self.__final_state

    def compute_list_fidelities_and_energy_gaps(self,):
        if self.__final_state == None:
            raise ValueError("The method solve need to be run before in order to get the list of fidelities")
        step = (self.__t_final - self.__t_initial) / self.__n_steps
        site_list = [l for l in range(1, self.__n_sites, 1)]
        t_list = [t for t in np.linspace(self.__t_initial, self.__t_final, self.__n_steps)]
        list_fidelities = np.zeros(self.__n_steps, dtype='complex128')  
        list_gaps = np.zeros(self.__n_steps, dtype='complex128') 
        list_difference_energy_with_gs_hamiltonian = np.zeros(self.__n_steps, dtype='complex128')

        dim = int(2**self.__n_sites)
        list_eigenvalues = np.zeros((self.__n_steps, dim), dtype='complex128')
        for idx, t_n in enumerate(t_list):
            H_t_n = self.H(t_n)
            state_t_n = self.__time_evolution[idx,:]
            ground_state_h_t_n, gap_h_t_n, difference_energy_with_gs_hamiltonian, eig_values = self.compute_ground_states_and_energy_gap(H_t_n, state_t_n, all_gs = False)
            fidelity = abs(np.vdot(np.conj(ground_state_h_t_n), state_t_n))
            list_fidelities[idx] = abs(np.vdot(np.conj(ground_state_h_t_n), state_t_n))
            list_gaps[idx] = gap_h_t_n
            list_difference_energy_with_gs_hamiltonian[idx] = difference_energy_with_gs_hamiltonian
            list_eigenvalues[idx,:] = eig_values
        self.__list_fidelities = list_fidelities
        self.__list_gaps = list_gaps
        self.__list_difference_energy_with_gs_hamiltonian = list_difference_energy_with_gs_hamiltonian
        self.__list_eigenvalues = list_eigenvalues

    def compute_ground_states_and_energy_gap(self, H_matrix, state_vector, all_gs = True):
        """
        Returns:
            ground_states: np.array: the ground states of the Hamiltonian H_matrix.
            gap: float: the gap between the ground state energy and the first excited states.
            difference_energy_with_gs_hamiltonian: float: the difference between the energy
            of the state and the ground state energy of the Hamiltonian.
            eig_values: float: the eigenvalues of the Hamiltonian (it does include the ground state).
        """
        # I need to add an option in case there are several gs.
        energy_state = self.compute_energy(state_vector, H_matrix)

        eig_values, eig_vectors = np.linalg.eigh(H_matrix) 
        length_vector = eig_vectors.shape[0]
        min_indices = np.asarray(abs(eig_values-eig_values.min())<10**(-12)).nonzero() #np.where(eig_values == eig_values.min())
        min_indices = np.asarray(min_indices)[0]
        ground_states = np.zeros([length_vector,min_indices.shape[0]],complex)
        for idx, value in enumerate(min_indices):
            eig_vector = eig_vectors[:,value]
            eig_vector = eig_vector[:]
            ground_states[:, idx] = eig_vector
        gap = self.min_energy_gap(eig_values)
        difference_energy_with_gs_hamiltonian = energy_state - np.min(eig_values)
        return ground_states, gap, difference_energy_with_gs_hamiltonian, eig_values


    def compute_energy(self, state_vector, H_matrix):
        """Calculate the energy of a state for a given Hamiltonian.
        Args:
            state (np.ndarray): The state vector.
            H (np.ndarray): The instantaneous Hamiltonian matrix.
        Returns:
            float: The energy of the state.
        """
        # Calculate the energy
        energy = np.dot(np.conjugate(state_vector), np.dot(H_matrix, state_vector))
        return energy.real  # Return the real part of the energy



    def min_energy_gap(self, eigenvalues):
        sorted_eigenvalues = np.sort(eigenvalues)
        gap = sorted_eigenvalues[1] - sorted_eigenvalues[0]
        return gap

    
    @property
    def list_difference_energy_with_gs_hamiltonian(self,):
        if self.final_state == None:
            raise ValueError("The method solve need to be run before in order to get the list of fidelities")
        else:
            if type(self.__list_difference_energy_with_gs_hamiltonian) == type(None):
                self.compute_list_fidelities_and_energy_gaps()
            return self.__list_difference_energy_with_gs_hamiltonian
    
    @property
    def list_eigenvalues(self,):
        if self.final_state == None:
            raise ValueError("The method solve need to be run before in order to get the list of fidelities")
        else:
            if type(self.__list_difference_energy_with_gs_hamiltonian) == type(None):
                self.compute_list_fidelities_and_energy_gaps()
            return self.__list_eigenvalues
    

    @property
    def list_fidelities(self,):
        if self.final_state == None:
            raise ValueError("The method solve need to be run before in order to get the list of fidelities")
        else:
            if type(self.__list_fidelities) == type(None):
                self.compute_list_fidelities_and_energy_gaps()
            return self.__list_fidelities

    @property
    def list_gaps(self,):
        if self.final_state == None:
            raise ValueError("The method solve need to be run before in order to get the list of energy gaps")
        else:
            if type(self.__list_gaps) == type(None):
                self.compute_list_fidelities_and_energy_gaps()
            return self.__list_gaps

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
            coupling_matrix_angle = self.ham_params['J']*self.__delta_t
            hx_angle = self.ham_params['g'] * t_n/(self.__t_final-self.__t_initial)*self.__delta_t
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
                