"""Module defining the Trotterization algorithms.
"""
from abc import ABCMeta, abstractmethod
import random
from collections import namedtuple
import numpy as np
#import sys 
import json
#from pathlib import Path
import time
#import environments as envs
from tdqc.interfaces.solver import Solver
import tdqc.numerics.deep_q_learning.environments_cpp as envs_cpp
import tdqc.numerics.deep_q_learning.models as models

class Trotterization(Solver):
    def __init__(self,):
        super().__init__()
        self.__check_validity__()

    def load_settings(self, settings):
        """
        Initialize settings stored in local variable self.__settings
        """
        if not "ham_params" in settings:
            raise ValueError("Error loading trotterization-solver settings, 'ham_params' parameter not found")
        self.ham_params = settings["ham_params"]
        if not "t_initial" in settings:
            raise ValueError("Error loading trotterization-solver settings, 't_initial' parameter not found")
        self.t_initial = settings["t_initial"]
        if not "t_final" in settings:
            raise ValueError("Error loading trotterization-solver settings, 't_final' parameter not found")
        self.t_final = settings["t_final"]
        if not "n_steps" in settings:
            raise ValueError("Error loading trotterization-solver settings, 'n_steps' parameter not found")
        self.n_steps = settings["n_steps"]
        self.final_state = None 

        self.time_step = (self.t_final - self.t_initial)/self.n_steps
        if not "target_params" in settings:
            raise ValueError("Error loading deep_q_learning-solver settings, 'target_params' parameter not found")
        self.__target_params = settings["target_params"]
        self.__target_params["t_initial"] = settings["t_initial"]
        self.__target_params["t_final"] = settings["t_final"]
        self.system_class = settings["system_class"]
        self.range_one =  settings["range_one"]
        self.range_all =  settings["range_all"]
        self.n_sites = settings["n_sites"]
        self.env = envs_cpp.DynamicalEvolution(
                 **settings)


        print(f'Instance of {type(self).__name__} initialized with '
              f'the following attributes (showing only str, int and float):')
        for attribute, value in self.__dict__.items():
            if type(value) in (str, int, float):
                print(f'{attribute} = {value}')


    @abstractmethod
    def solve(self):
        raise NotImplementedError('Vanilla DQL has no implementation without'
                                  'ReplayMemory.')
    



    def save_trotterization_actions(self, filetype, filename):
        if filetype == 'txt':
            raise ValueError
        elif filetype == 'json':
            try:
                gates = self.action_trotterization
                """if len(gates) == 3 or self.env.n_directions == 2:
                    jx_gates, hx_gates, hz_gates, *_ = gates
                    steps = [
                        [('jx', jx_gate), ('hz', list(hz_gate)),
                         ('hx', list(hx_gate))]
                        for jx_gate, hz_gate, hx_gate
                        in zip(jx_gates, hz_gates, hx_gates)
                    ]
                elif len(gates) == 4:
                    raise NotImplementedError
                """                
                with open(filename, 'w') as f:
                    json.dump(gates, f, indent=2)
                print(f"{filename} written.")
            except Exception as e:
                print(f'`{filename}` could not be saved.')
                print('--> ', e)
    
    def get_rho_target_from_other_solver(self,)-> np.ndarray:
        target_params = self.__target_params
        solver_for_target = target_params['solver']
        #target_params.pop('solver', None)
        solver_for_target.load_settings(target_params)
        solver_for_target.solve()
        state_target = solver_for_target.get_state_target()
        self.state_target = state_target
        rho_target = solver_for_target.get_rho_target()
        self.rho_target = rho_target
        return rho_target

    def trotterization_circuit(self, all_zeros: bool=False)-> np.ndarray:
        """Return an initial list of actions for each step.

        When the Trotter decomposition exists, this is
        = [[a_1, a_2, ...]]*n_steps
        reminder: each a_j correspond to a gate (single qubit or entangling)

        parameters:
            all_zeros (int): if True, all a_j = 0.0.

        Returns: np.array with shape=(n_steps, action_dim)
        """
        if self.system_class == 'LongRangeIsing':
            if self.env.n_directions == 2:
                a_all = self.ham_params['J'] * self.time_step \
                     / self.range_all
                a_onex = self.ham_params['g'] * self.time_step \
                    / self.range_one
                a_onez = self.ham_params['h'] * self.time_step \
                     / self.range_one
                action = [a_all] + [a_onez] * self.n_sites \
                    + [a_onex] * self.n_sites
                return [action] * self.n_steps
            else:
                raise NotImplementedError('Trotter sequence only implemented'
                                          ' for n_directions = 2.')
        else:
            raise NotImplementedError('Trotter sequence only implemented'
                                          ' for this system_class.')
    
    

    def solve(self):
        # This method runs the Trotterization. 
        rho_target = self.get_rho_target_from_other_solver()
        start_time = time.time()   
        self.action_trotterization = self.trotterization_circuit(False)
        reward_trotterization = self.env.reward(action_sequence=self.action_trotterization,rho_target=rho_target)
        end_time = time.time()
        parametername = 'trotterization_N'+str(self.env.n_sites)+'n_steps'+str(self.n_steps)
        self.save_trotterization_actions('json',
                                                'trotterization_gate_sequence'+parametername+'.json')
        try:
            reward_filename = 'reward'+parametername+'.npy'
            with open(reward_filename, 'wb') as f:
                np.save(f, reward_trotterization)
        except Exception as e:
            print(reward_filename+' could not be saved.')
            print('--->', e)
        info_dic = {
            #  'parameters': parameters,
            'reward_trotterization': reward_trotterization,
            'trotterization_time': end_time - start_time,
            'target_state': str(self.state_target),
            #'final_state':
            }   
        try:
            result_info_filename = 'results_info'+parametername+'.json'
            with open(result_info_filename, 'w') as f:
                json.dump(info_dic, f, indent=2)
            print(result_info_filename+' written.')
        except Exception as e:
            print(result_info_filename+' could not be saved.')
            print('--->', e)



