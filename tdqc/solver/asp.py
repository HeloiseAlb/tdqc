import numpy as np
import cmath 
from math import log, isnan, cos, sin 
import json
from scipy.linalg import expm, eigh, logm
from typing import Optional 
from tdqc.interfaces.solver import Solver
from tdqc.numerics.ed.models_ed import State, Model

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

def globalize_op(local_op, site, L):
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
        if not "gate_order" in settings and self.__system_class == "LongRangeIsing":
            raise ValueError("Error loading asp-solver settings, 'gate_order' parameter not found")
        else:
            self.__gate_order = settings["gate_order"]
        if "ferro_angle" in settings and settings['ferro_angle'] != 0: 
            self.__gate_along_y = True
            self.__ferro_angle = settings['ferro_angle']
            if not "ferro_gate_order" in settings:
                raise ValueError("Error loading asp-solver settings, 'ferro_gate_order' parameter not found")
            else:
                self.__ferro_gate_order = settings["ferro_gate_order"]
        else: 
            self.__gate_along_y = False
        # Define the initial Hamiltonian H_0 and the final Hamiltonian H_f
        #self.H_0 = self.__model_0.model_hamiltonian
        #self.H_f = self.__model_f.model_hamiltonian
        #self.__time_evolution = None
        self.__final_state = None 
        self.__list_fidelities = None
        self.__list_gaps = None
        self.__list_transition_matrix_element = None 
        self.__final_reward = None 
        self.__delta_t = (self.__t_final-self.__t_initial)/self.__n_steps
        self.set_initial_state()
        self.set_coupling_matrix()

    def set_initial_state(self,)-> None:
        # The initial state is the GS of the initial hamiltonian H_0.
        # Here, I consider the ground state is non-degenerate.
        # To do: Code the case where it is degenerate.
        ground_state = self.__model_0.ground_state 
        init_vec_state = np.array(ground_state,dtype='complex128')
        norm = np.linalg.norm(init_vec_state)
        init_vec_state = init_vec_state / norm
        self.__initial_state = State(init_vec_state)
    
    def set_coupling_matrix(self,)-> None:
        dim = int(2**self.__n_sites)
        list_glob_operators =  [None] * self.__n_sites
        for site in range(0,self.__n_sites,1):
            list_glob_operators[site] = globalize_op(spin_op["sigma_x"],site, self.__n_sites)  
        coupling_matrix = np.zeros((dim,dim),dtype='complex128')
        if self.__system_class == 'LongRangeIsing':
            for l in range(0,self.__n_sites,1):
                matrix_1 = list_glob_operators[l] 
                for k in range(l+1,self.__n_sites,1):
                    matrix_2 = list_glob_operators[k]
                    coupling_matrix += np.dot(matrix_1,matrix_2)/(k-l)**self.ham_params['alpha']
            self.coupling_matrix = coupling_matrix
        elif self.__system_class == 'TransIsing':
            for l in range(0, self.__n_sites-1, 1):
                matrix_1 = list_glob_operators[l] 
                matrix_2 = list_glob_operators[l+1]
                coupling_matrix += np.dot(matrix_1,matrix_2)
            self.coupling_matrix = coupling_matrix
        elif self.__system_class == 'LongRangeTransIsing':
            for l in range(0, self.__n_sites, 1):
                matrix_1 = list_glob_operators[l] 
                for k in range(l+1,self.__n_sites,1):
                    matrix_2 = list_glob_operators[k]
                    coupling_matrix += np.dot(matrix_1,matrix_2)/(k-l)**self.ham_params['alpha']
            self.coupling_matrix = coupling_matrix
        else:
            raise NotImplementedError('Trotter sequence not implemented'
                                          ' for your system_class. Only implemented'
                                          ' for TransIsing and LongRangeIsing.')

    # Define the time-dependent Hamiltonian H(t) using a linear schedule
    ### I need to implement the one using a non linear schedule.
   
    def H(self, t):
        T = self.__t_final
        H_t = (1 - t/T)*self.__model_0.hamiltonian + (t/T)*self.__model_f.hamiltonian
        return H_t

    def solve(self, ED = False)-> None:
        """
        It realizes the discretization of the time to have instantaneous Hamiltonian on a 
        small segment of time. 
        Then, by default, it applies the circuit of the Trotterization for this instantaneous 
        Hamiltonian. If ED = True, it applies realised the ED to compute the state at the end 
        of the small time step.
        It stores the list of the vec_state in self.__time_evolution.  
        """
        state_t_n = self.__initial_state.get_vector_state()
        time_step = self.__delta_t
        site_list = [l for l in range(1, self.__n_sites, 1)]
        t_list = [t for t in np.linspace(self.__t_initial, self.__t_final, self.__n_steps)]
        time_evolution = np.zeros([self.__n_steps + 1, 2**self.__n_sites], dtype='complex128') 
        inv_temperature = 1
        if ED:
            time_evolution[0, :] = state_t_n.reshape(-1)
            for idx, t_n in enumerate(t_list[:]):
                model = Model("instanteneous_hamiltonian", lambda: self.H(t_n))
                model.parametrize_hamiltonian()
                state_t_n = self.apply_ed_evolution(state_t_n.reshape(-1), model, time_step, imaginary=False, h_bar=1)
                time_evolution[idx + 1, :] = state_t_n.reshape(-1)
        else:
            # Else, apply the Trotterization circuit.
            r = state_t_n.reshape(-1)
            time_evolution[0, :] = state_t_n.reshape(-1)
            self.list_coupling_matrix_angles = np.zeros([self.__n_steps, 1], dtype='float') 
            self.list_hx_angle = np.zeros([self.__n_steps, self.__n_sites], dtype='float') 
            self.list_hz_angle = np.zeros([self.__n_steps, self.__n_sites], dtype='float') 
            if self.__gate_along_y:
                self.list_hy_angle = np.ones([self.__n_steps, self.__n_sites], dtype='float') 
            for idx, t_n in enumerate(t_list[:]):
                coupling_matrix_angle, hx_angle, hy_angle, hz_angle = self.define_gate_angles(idx)
                self.list_coupling_matrix_angles[idx] = coupling_matrix_angle
                self.list_hx_angle[idx] = hx_angle
                if self.__gate_along_y:
                    self.list_hy_angle[idx] = hy_angle
                self.list_hz_angle[idx] = hz_angle
                state_t_n = self.apply_gate_sequence(state_t_n, coupling_matrix_angle, hx_angle, hy_angle, hz_angle)
                time_evolution[idx + 1, :] = state_t_n.reshape(-1)
        self.__time_evolution = time_evolution
        self.__final_state = State(state_t_n) # It is an instance of the class State()
    

    def save_gate_sequence(self,):
        parametername = 'ASP_ti'+'N'+str(self.__n_sites)+'n_steps'+str(self.__n_steps)+'t_final'+str(self.__t_final)+'J'+str(self.ham_params['J'])+'h'+str(self.ham_params['h'])
        filename = 'gate_sequence'+parametername+'.json'
        try:
            jx_gates, hx_gates, hz_gates = self.list_coupling_matrix_angles, self.list_hx_angle, self.list_hz_angle 

            steps = [
                [('jx', list(jx_gate)), ('hz', list(hz_gate)),
                    ('hx', list(hx_gate))]
                for jx_gate, hz_gate, hx_gate
                in zip(jx_gates, hz_gates, hx_gates)
            ]
            with open(filename, 'w') as f:
                json.dump(steps, f, indent=2)
            print(f"{filename} written.")
        except Exception as e:
            print(f'`{filename}` could not be saved.')
            print('--> ', e)

    def reduced_density_matrix(self, rho_init: np.ndarray, site1: int, site2: int, n_qubits: int)-> np.ndarray:
        """ Return the reduced density matrix of the subsystem made of sites site1 and site2 for rho. So a 4-by-4 matrix. """    
        rho = rho_init 
        if site1>site2:
            site1, site2 = site2, site1
        if site1>0:
            n1, n2 = int(2**(site1)), int(2**(n_qubits-site1))
            rho = rho.reshape([n1, n2, n1, n2])
            rho = np.trace(rho, axis1=0, axis2=2)
            n_qubits -= site1
            site2 -= site1
        if site2>1:
            n1, n2 = int(2**(site2-1)), int(2**(n_qubits-site2))
            rho = rho.reshape([2,n1,n2,2,n1,n2])
            rho = np.trace(rho, axis1=1, axis2=4)
            n_qubits -= site2-1
        if n_qubits>2:
            n2 = int(2**(n_qubits-2))
            rho = rho.reshape([4, n2, 4, n2])
            rho = np.trace(rho, axis1=1, axis2=3)
        rho = rho.reshape([4, 4])
        return rho

    def relative_entropy(self, rho1: np.ndarray, rho2: np.ndarray, positiveDefinite: bool)-> float:
        if positiveDefinite:
            # Diagonalization the matrix to compute the quantum relative entropy. The matrices must be hermitian positive semidefinite.
            eVals1, eVecs1 = eigh(rho1) 
            eVals1 = np.maximum(eVals1, 0)
            eVals2, eVecs2 = eigh(rho2) 
            eVals2 = np.maximum(eVals2, 0)
            relativeEntropy = 0
            for index1, value1 in enumerate(eVals1):
                subsum_index1 = 0
                if value1 > 0:
                    relativeEntropy += value1 * (log(value1))
                    for index2, value2 in enumerate(eVals2):
                        if value2 > 0 :
                            subsum_index1 += abs( np.dot(np.conj(eVecs2[:, index2]), eVecs1[:, index1]))**2 * log(value2)
                    relativeEntropy -= value1 * subsum_index1
            return np.real(relativeEntropy)
        else:
            return np.trace(np.dot(rho1,(logm(rho1)-logm(rho2))))

    def local_reward(self, rho1: np.ndarray, rho2: np.ndarray, positiveDefinite: Optional[bool]=False)-> float:
        n_qubits = self.__n_sites
        sum_measures = 0
        for j in range(0,n_qubits-1):
            for k in range(j+1, n_qubits):
                sum_measures += cmath.sqrt(self.relative_entropy(self.reduced_density_matrix(rho1, j, k, n_qubits), self.reduced_density_matrix(rho2, j, k, n_qubits), positiveDefinite))
        if sum_measures == float('inf') or isnan(sum_measures.real) or isnan(sum_measures.imag):
            r_local = 0.0 + 1j*0.0
            print("sum_measures was Nan, r_local taken to be 0")
        else:
            r_local = 1 - 2/(n_qubits*(n_qubits-1)) * sum_measures  
        return max(0, r_local.real)


    def compute_property_lists(self,):
        if self.__final_state == None:
            raise ValueError("The method solve need to be run before in order to get the list of fidelities")
        step = (self.__t_final - self.__t_initial) / self.__n_steps
        site_list = [l for l in range(1, self.__n_sites, 1)]
        t_list = [t for t in np.linspace(self.__t_initial, self.__t_final, self.__n_steps)]
        list_fidelities = np.zeros(self.__n_steps, dtype='complex128')  
        list_gaps = np.zeros(self.__n_steps, dtype='complex128') 
        list_difference_energy_with_gs_hamiltonian = np.zeros(self.__n_steps, dtype='complex128')

        dim = int(2**self.__n_sites)
        list_eigenvalues = np.zeros((self.__n_steps, dim))
        list_eigenvectors = np.zeros((self.__n_steps, dim, dim))
        list_transition_matrix_element = np.zeros(self.__n_steps, dtype='float')
        for idx, t_n in enumerate(t_list):
            H_t_n = self.H(t_n)
            state_t_n = self.__time_evolution[idx,:]
            ground_state_h_t_n, gap_h_t_n, difference_energy_with_gs_hamiltonian, eig_values, eig_vectors, transition_matrix_element = self.compute_properties(H_t_n, state_t_n, all_gs = False)
            fidelity = abs(np.vdot(np.conj(ground_state_h_t_n), state_t_n))
            list_fidelities[idx] = abs(np.vdot(np.conj(ground_state_h_t_n), state_t_n))
            list_gaps[idx] = gap_h_t_n
            list_difference_energy_with_gs_hamiltonian[idx] = difference_energy_with_gs_hamiltonian
            list_eigenvalues[idx,:] = eig_values
            list_eigenvectors[idx,:,:] = eig_vectors
            list_transition_matrix_element[idx] = transition_matrix_element
        self.__list_fidelities = list_fidelities
        self.__list_gaps = list_gaps
        self.__list_difference_energy_with_gs_hamiltonian = list_difference_energy_with_gs_hamiltonian
        self.__list_eigenvalues = list_eigenvalues
        self.__list_eigenvectors = list_eigenvectors
        self.__list_transition_matrix_element = list_transition_matrix_element

        
    def generate_data_files(self,):
        if (self.__final_state == None):
            # If the method solve has not run yet. 
            self.solve()
        parametername = 'ASP_ti'+'N'+str(self.__n_sites)+'n_steps'+str(self.__n_steps)+'t_final'+str(self.__t_final)+'J'+str(self.ham_params['J'])+'h'+str(self.ham_params['h'])

        # Generate the file with the time evolution amplitudes.
        try:
            amplitude_filename = 'amplitude_'+parametername+'.npy'
            with open(amplitude_filename, 'wb') as f:
                np.save(f, self.time_evolution)
        except Exception as e:
            print(amplitude_filename+' could not be saved.')
            print('--->', e)
        
        # Generate the file with the final state.
        try:
            finalstate_filename = 'finalstate_'+parametername+'.npy'
            with open(finalstate_filename, 'wb') as f:
                np.save(f, self.final_state.get_vector_state())
        except Exception as e:
            print(finalstate_filename+' could not be saved.')
            print('--->', e)
        final_state = self.__final_state.get_vector_state()
        rho_final = np.tensordot(np.conjugate(final_state), final_state, axes=0)
        self.__final_reward = self.local_reward(rho_final, self.get_rho_target())

        # Generate the file with the parameters.
        info_dic = {
            'Hamiltonian parameters': self.ham_params,
            'Initial_state': str(self.__initial_state.get_density_matrix()),
            'Final fidelity': str(self.__list_fidelities[-1]),
            'H_0 model': self.__model_0.name,
            'H_0 hamiltonian': str(self.__model_0.hamiltonian),
            'H_f model': self.__model_f.name,
            'H_f hamiltonian': str(self.__model_f.hamiltonian),
            'Initial time': self.__t_initial,
            'Final time': self.__t_final,
            'Time step (delta t)': self.__delta_t,
            'Ground state of H_f': str(self.__model_f.ground_states),
            'Minimum energy gap with the GS': str(np.min(self.list_gaps)),
            'max_{0=<s=<1} |<l=1,s|dH/ds|l=0,s>|': np.max(self.__list_transition_matrix_element),
            'Final local reward': self.__final_reward
            }   
        try:
            result_info_filename = 'results_info'+parametername+'.json'
            with open(result_info_filename, 'w') as f:
                json.dump(info_dic, f, indent=2)
            print(result_info_filename+' written.')
        except Exception as e:
            print(result_info_filename+' could not be saved.')
            print('--->', e)


    def compute_properties(self, H_matrix, state_vector, all_gs = True):
        """
        Returns:
            ground_states: np.array: the ground states of the Hamiltonian H_matrix.
            gap: float: the gap between the ground state energy and the first excited states.
            difference_energy_with_gs_hamiltonian: float: the difference between the energy
                of the state and the ground state energy of the Hamiltonian.
            eig_values: float: the eigenvalues of the Hamiltonian (it does include the ground state).
            transition_matrix_element: float: for the time s of the instantaneous matrix, |<l=1,s|dH/ds|l=0,s>|
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
        transition_matrix_element = self.compute_transition_matrix_element(eig_values, eig_vectors)
        return ground_states[:,0], gap, difference_energy_with_gs_hamiltonian, eig_values, eig_vectors, transition_matrix_element

    def compute_energy(self, state_vector, H_matrix):
        """Calculate the energy of a state for a given Hamiltonian.
        Args:
            state (np.ndarray): The state vector.
            H (np.ndarray): The instantaneous Hamiltonian matrix.
        Returns:
            float: The energy of the state.
        """
        energy = np.dot(np.conjugate(state_vector), np.dot(H_matrix, state_vector))
        return energy.real  # Return the real part of the energy

    def min_energy_gap(self, eigenvalues):
        """Calculate the energy gap between the two lower energy eigenvalues.
        Args:
            eigenvalues: np.ndarray: All the eigenvalues.
        Returns:
            float: The energy gap between the two smallest energy eigenvalues.
        """
        sorted_eigenvalues = np.sort(eigenvalues)
        gap = sorted_eigenvalues[1] - sorted_eigenvalues[0]
        return gap       

    def get_rho_target(self,)-> np.ndarray:
        # Attention: it does not correspond to the ground state of the final Hamiltonian 
        # but to the final state. This method aims to be used by the DQL solver. 
        if (self.__final_state == None):
            raise ValueError("The method solve need to be run before in order to get the target_state")
        target = self.__final_state.get_density_matrix()
        return target

    def get_state_target(self,)-> np.ndarray:
        # Attention: it does not correspond to the ground state of the final Hamiltonian 
        # but to the final state. This method aims to be used by the DQL solver. 
        if (self.__final_state == None):
            raise ValueError("The method solve need to be run before in order to get the target_state")
        target = self.__final_state.get_vector_state()
        return target

    def define_gate_angles(self, index:int) -> tuple:
        """
        The gates are defined to realize the adiabatic state preparation of the Hamiltonian.
        """
        # I need to implement difference way to weight the H_0 and H_f. For 
        # the moment, it is only possible linearly w.r.t. the time. 
        if self.__system_class == 'LongRangeIsing':
            # Trotter sequence only implemented for n_directions = 2.
            coupling_matrix_angle = self.ham_params['J'] * self.__delta_t * index / self.__n_steps
            hx_angle = self.ham_params['g'] * self.__delta_t
            hz_angle = self.ham_params['h'] * self.__delta_t
            return coupling_matrix_angle, hx_angle, None, hz_angle
        elif self.__system_class == 'TransIsing' or self.__system_class == 'LongRangeTransIsing':
            coupling_matrix_angle = self.ham_params['J'] * self.__delta_t * index / self.__n_steps
            if self.__gate_along_y:
                # TO BE MODIFIED
                hz_angle = self.ham_params['g'] * self.__delta_t * (index/self.__n_steps +(1-index/self.__n_steps)*cos(self.__ferro_angle))
                hy_angle = self.ham_params['g'] * self.__delta_t * (1- index/self.__n_steps) * sin(self.__ferro_angle)
                return coupling_matrix_angle, None, hy_angle, hz_angle
            else: # If not self.__gate_along_y
                hz_angle = self.ham_params['g'] * self.__delta_t
                return coupling_matrix_angle, None, None, hz_angle
        else:
            raise NotImplementedError('Trotter sequence not implemented'
                                          ' for your system_class. Only implemented'
                                          ' for TransIsing and LongRangeIsing.')
        
    def apply_gate_sequence(self, state, coupling_matrix_angle, hx_angle, hy_angle, hz_angle)-> np.ndarray:
        """ Apply the sequence of gates onto the initial state and return the final state.
        Args:
            state (np.ndarray): The state vector.
            coupling_matrix_angle (float): The angle of the entangling gate.
            hx_angle (float or None): The angle of the x-rotation gates. It is just one float 
                even if one x-rotation gate is applied on each qubit because the angles are all 
                equal. It may be None in the case of the (long range or not) transverse Ising 
                model because then the Trotterization does not require x-rotation gates.
            hz_angle (float): The angle of the z-rotation gates. It is just one float 
                even if one z-rotation gate is applied on each qubit because the angles are all 
                equal.
        Returns:
            np.array: The updated state vector.
        """
        # Define the universal quantum gate set used in Markus article. 
        U_z = lambda theta : expm(-1j*theta*spin_op['sigma_z'])
        if self.__gate_along_y:
            U_y = lambda theta : expm(-1j*theta*spin_op['sigma_y'])
        sum_U_xx = self.coupling_matrix 
        U_xx = lambda theta : expm(-1j*theta*sum_U_xx)
        # Those gate lists are in fact lists of angles.
        state = np.dot(U_xx(coupling_matrix_angle), state)
        if hx_angle != None:       
            U_x = lambda theta : expm(-1j * theta * spin_op['sigma_x'])
            if self.__gate_order == "xz":
                for site in range(0, self.__n_sites, 1):
                    U_x_site = globalize_op(U_x(hx_angle), site, self.__n_sites)
                    state = np.dot(U_x_site, state)
                    U_z_site = globalize_op(U_z(hz_angle), site, self.__n_sites)
                    state = np.dot(U_z_site, state)
            elif self.__gate_order == "zx":
                for site in range(0, self.__n_sites, 1):
                    U_z_site = globalize_op(U_z(hz_angle), site, self.__n_sites)
                    state = np.dot(U_z_site, state)
                    U_x_site = globalize_op(U_x(hx_angle), site, self.__n_sites)
                    state = np.dot(U_x_site, state)
        elif self.__gate_along_y: # Equivalent to hy_angle !=None: 
            if self.__ferro_gate_order == "zy":
                for site in range(0, self.__n_sites, 1):
                    U_z_site = globalize_op(U_z(hz_angle), site, self.__n_sites)
                    state = np.dot(U_z_site, state)
                    U_y_site = globalize_op(U_y(hy_angle), site, self.__n_sites)
                    state = np.dot(U_y_site, state)
            elif self.__ferro_gate_order == "yz":
                for site in range(0, self.__n_sites, 1):
                    U_y_site = globalize_op(U_y(hy_angle), site, self.__n_sites)
                    state = np.dot(U_y_site, state)
                    U_z_site = globalize_op(U_z(hz_angle), site, self.__n_sites)
                    state = np.dot(U_z_site, state)
        else:
            for site in range(0, self.__n_sites, 1):
                U_z_site = globalize_op(U_z(hz_angle), site, self.__n_sites)
                state = np.dot(U_z_site, state)
        return state
                

    def apply_ed_evolution(self, init_vec_state, model, delta_t, imaginary=False, h_bar=1):
        '''
        Time evolution of a system after a quench using exact diagonalization. 
        It makes the state initial_state evolve according to the Hamiltonian of the model for a time delta_t.
        '''
        # Input to simulate the imaginary time evolution, by default, it is the real time evolution.
        if imaginary:
            delta_t = -1j * delta_t
        new_vec_state = np.dot(expm(-1j * delta_t * model.hamiltonian), init_vec_state)
        return new_vec_state


    def compute_transition_matrix_element(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray)-> float:
        """
        This function requires that the initial Hamiltonian is the 
        non-interacting part such as:
        dH(s)/ds =d ((1-s)H_0 +sH_T)/ds=H_coupling_matrix
        It aims at computing for each 0=<s=<1 of our discretization, the 
        |<l=1,s|dH(s)/ds|l=0,s>| in the formula (2.9) of arXiv:quant-ph/0001106.
        """
        # Sort eigenvalues and eigenvectors in ascending order
        sorted_indices = np.argsort(eigenvalues)
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Extract eigenvectors corresponding to the smallest and second smallest eigenvalues
        smallest_eigenvector = sorted_eigenvectors[:, 0]
        second_smallest_eigenvector = sorted_eigenvectors[:, 1]

        # Compute dH/ds
        dH_ds = self.ham_params['J'] * self.coupling_matrix
        
        # Compute the projection of the smallest eigenvector onto the second smallest eigenvector
        projection = np.dot(second_smallest_eigenvector, np.dot(dH_ds, smallest_eigenvector))

        return np.abs(projection)

    @property
    def time_evolution(self):
        # It returns the amplitudes of the time evolution.
        return self.__time_evolution

    @property
    def n_steps(self):
        return self.__n_steps

    @property
    def final_state(self):
        return self.__final_state

    @property
    def initial_state(self):
        return self.__initial_state

    @property
    def t_final(self,):
        return self.__t_final

    @property
    def system_class(self,):
        return self.__system_class
    
    @property
    def list_difference_energy_with_gs_hamiltonian(self,)-> np.ndarray:
        if self.final_state == None:
            raise ValueError("The method solve need to be run before in order to get the list of energies.")
        else:
            if type(self.__list_difference_energy_with_gs_hamiltonian) == type(None):
                self.compute_property_lists()
            return self.__list_difference_energy_with_gs_hamiltonian
    
    @property
    def list_eigenvalues(self,)-> np.ndarray:
        if self.final_state == None:
            raise ValueError("The method solve need to be run before in order to get the list of eigenvalues.")
        else:
            if type(self.__list_difference_energy_with_gs_hamiltonian) == type(None):
                self.compute_property_lists()
            return self.__list_eigenvalues

    @property
    def list_eigenvectors(self,)-> np.ndarray:
        if self.final_state == None:
            raise ValueError("The method solve need to be run before in order to get the list of eigenvectors.")
        else:
            if type(self.__list_difference_energy_with_gs_hamiltonian) == type(None):
                self.compute_property_lists()
            return self.__list_eigenvectors

    @property
    def list_fidelities(self,)-> np.ndarray:
        if self.final_state == None:
            raise ValueError("The method solve need to be run before in order to get the list of fidelities.")
        else:
            if type(self.__list_fidelities) == type(None):
                self.compute_property_lists()
            return self.__list_fidelities

    @property
    def list_gaps(self,)-> np.ndarray:
        if self.final_state == None:
            raise ValueError("The method solve need to be run before in order to get the list of energy gaps.")
        else:
            if type(self.__list_gaps) == type(None):
                self.compute_property_lists()
            return self.__list_gaps

    @property
    def list_transition_matrix_element(self,)-> np.ndarray:
        if self.final_state == None:
            raise ValueError("The method solve need to be run before in order to get the list of transition matrix elements of H.")
        else:
            if type(self.__list_transition_matrix_element) == type(None):
                self.compute_property_lists()
            return self.__list_transition_matrix_element

    @property
    def final_reward(self,)-> float:
        if self.final_state == None:
            raise ValueError("The method solve need to be run before in order to get the local reward of the final state.")
        else:
            if type(self.__final_reward) == type(None):
                self.compute_property_lists()
            return self.__final_reward


