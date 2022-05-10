import numpy as np
from tdqc.interfaces.solver import Solver

class EDSolver(Solver):
    """
    Class implementing ED solver
    """

    def __init__(self,):
        super().__init__()
        self.__check_validity__()

    def load_settings(self, settings):
        """
        Initialize settings stored in local variable self.__settings
        """
        if not "model" in settings:
            raise ValueError("Error loading ed-solver settings, 'model' parameter not found")
        self.__model = settings["model"]
        if not "state" in settings:
            raise ValueError("Error loading ed-solver settings, 'state' parameter not found")
        self.__state = settings["state"]
        if not "t_initial" in settings:
            raise ValueError("Error loading ed-solver settings, 't_initial' parameter not found")
        self.__t_initial = settings["t_initial"]
        if not "t_final" in settings:
            raise ValueError("Error loading ed-solver settings, 't_final' parameter not found")
        self.__t_final = settings["t_final"]
        # step is the space between the values.
        if not "step" in settings:
            raise ValueError("Error loading ed-solver settings, 'step' parameter not found")
        self.__step = settings["step"]
        if not "imaginary" in settings:
            # Is the time evolution an imaginary time evolution?
            self.__imaginary = False
        else:
            self.__imaginary = settings["imaginary"]
        self.__time_evolution = None
        self.__final_state = None 
    
    @property
    def time_evolution(self):
        # It returns the amplitudes of the time evolution.
        return self.__time_evolution

    def solve(self):
        # This method runs the time evolution and stores the list of the vec_state in self.__time_evolution.  
        state_t_n = self.__state
        model = self.__model
        t_initial = self.__t_initial
        t_final = self.__t_final
        step = self.__step
        imaginary = self.__imaginary
        n_sites = state_t_n.n_sites
        site_list = [l for l in range(1,n_sites,1)]
        t_list = [t for t in np.arange(t_initial,t_final,step)]
        time_evolution = np.zeros([int((t_final-t_initial)/step),2**n_sites],dtype='complex128') # [None] * int((t_max-t_min)/step) #np.zeros([int((t_max-t_min)/step)])
        inv_temperature = 1
        for idx,t_n in enumerate(t_list):
            time_evolution[idx,:] = state_t_n.vec_state
            # Time evolution
            state_t_n.time_step_ed(model,step,imaginary=imaginary)
        # Check of the sum of probabilities
        #print(thermal_exp_value(eig_values,eig_vectors,H,0))
            self.__time_evolution = time_evolution
        self.__final_state = state_t_n


    def get_target_state(self):
        if (self.__final_state == None):
            raise ValueError("The method solve need to be run before in order to get the target_state")
        target = self.__final_state.get_state_format_ml()
        return target
