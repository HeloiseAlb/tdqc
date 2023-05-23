import numpy as np
from tdqc.interfaces.solver import Solver
from tdqc.numerics.ed.models_ed import State, Model
from scipy.linalg import expm


class AdiaStatePrepa(Solver):
    """
    Class implementing the adiabatic state preparation
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

        # Define the initial Hamiltonian H_0 and the final Hamiltonian H_f
        #self.H_0 = self.__model_0.model_hamiltonian
        #self.H_f = self.__model_f.model_hamiltonian
        # The initial state is the GS of the initial hamiltonian H_0.
        self.__state = State(self.__model_0.ground_states)
        print("state:{}".format(self.__state))
        #self.__time_evolution = None
        self.__final_state = None 
    
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
    def H(self, t, T):
        H_t = (1 - t/T)*self.__model_0.model_hamiltonian() + (t/T)*self.__model_f.model_hamiltonian()
        return H_t
    """
    # Define the time evolution operator U(t, T)
    def U(self, t, T):
        return expm(-1j*self.H(t, T))
    """
    def solve(self,)-> None:
        # This method runs the time evolution and stores the list of the vec_state in self.__time_evolution.  
        state_t_n = self.__state
        t_initial = self.__t_initial
        t_final = self.__t_final
        n_steps = self.__n_steps
        step = (t_final-t_initial)/n_steps

        n_sites = state_t_n.n_sites
        site_list = [l for l in range(1,n_sites,1)]
        t_list = [t for t in np.linspace(t_initial,t_final,n_steps)]
        time_evolution = np.zeros([n_steps,2**n_sites],dtype='complex128') # [None] * int((t_max-t_min)/step) #np.zeros([int((t_max-t_min)/step)])
        inv_temperature = 1
        for idx, t_n in enumerate(t_list):
            #time_evolution[idx,:] = state_t_n.vec_state
            time_evolution[idx,:] = state_t_n.vec_state.flatten()
            # Time evolution
            model = Model("temporary", self.H(t_n, t_final))
            state_t_n.time_step_ed(model, step, imaginary=False)
        self.__time_evolution = time_evolution
        self.__final_state = state_t_n # It is an instance of the class State()

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

