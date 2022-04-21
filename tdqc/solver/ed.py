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
			raise ValueError("Error loading ed-solver settings, 'initial_state' parameter not found")
		self.__initial_state = settings["initial_state"]
		if not "interval" in settings:
			raise ValueError("Error loading ed-solver settings, 'interval' parameter not found")
		self.__interval = settings["interval"]
		if not "steps" in settings:
			raise ValueError("Error loading ed-solver settings, 'steps' parameter not found")
		self.__steps = settings["steps"]
                self.__time_evolution = None
        
        '''
        # TO DO
	def solve(self):
            # This method runs the time evolution and stores the amplitudes in self.__time_evolution.  
            state_t_n = models_ed.State(self.__initial_state)
            site_list = [l for l in range(1,self.n_sites,1)]
            t_list = [t for t in np.arange(self.t_initial,self.t_final,self.step)]
            time_evolution = np.zeros([int((self.t_final-self.t_initial)/self.step),2**self.n_sites],dtype='complex128') # [None] * int((t_max-t_min)/step) #np.zeros([int((t_max-t_min)/step)])
            inv_temperature = 1
            for idx,t_n in enumerate(t_list):
                time_evolution[idx,:] = psi_t_n.amplitudes
                ### Time evolution
                ### Imaginary time evolution 
                psi_t_n.time_step_ed(self.model,-1j*self.step)
            # Check of the sum of probabilities
            #print(thermal_exp_value(eig_values,eig_vectors,H,0))
            self.__time_evolution = time_evolution	
'''

    def get_time_evoltion(self):
        return self.__time_evolution
