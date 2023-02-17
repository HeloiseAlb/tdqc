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
        self.ham_params = settings["t_initial"]
        if not "t_final" in settings:
            raise ValueError("Error loading trotterization-solver settings, 't_final' parameter not found")
        self.ham_params = settings["t_final"]
        if not "n_steps" in settings:
            raise ValueError("Error loading trotterization-solver settings, 'n_steps' parameter not found")
        self.ham_params = settings["n_steps"]


        self.best_final_state = None 
        print(f'Instance of {type(self).__name__} initialized with '
              f'the following attributes (showing only str, int and float):')
        for attribute, value in self.__dict__.items():
            if type(value) in (str, int, float):
                print(f'{attribute} = {value}')

    def run(self):
        self.env.reset()
        rewards = np.zeros((self.n_episodes, self.env.n_steps))
        self.best_encountered_actions = self.env.initial_action_sequence()
        if self.env_type in ['DynamicalEvolution',
                             'DynamicalEvolution_cpp',
                             'EnergyMinimizer',
                             'EnergyMinimizer_cpp']:
            self.best_encountered_rewards = (
                [0]*(len(self.best_encountered_actions) - 1)
                + [self.env.reward(self.best_encountered_actions,self.rho_target)]
            )
            
        print('Final reward of the initial action sequence'
              f' is {self.best_encountered_rewards[-1]}')
        self.time_select_action_sum = 0
        self.time_compute_reward_sum = 0
        self.time_fit_network_sum = 0 
        self.exploration_vs_exploitation = []
        for episode in range(self.n_episodes):

            if episode % self.model_update_spacing == 0:
                self.model.update_target()
                print("self.model.update_target() is done")

            verbose = False
            if episode % (self.n_episodes//10) == 0:
                verbose = True
                print(f'Episode {episode}: ')

            mode = 'explore'

            reward_sequence, action_sequence, final_state = self.run_episode(verbose,
                                                                mode=mode)
            print("Episode: {} and Action sequence: {}".format(episode, action_sequence))
            reward_sequence = np.real(reward_sequence)
            print('reward_sequence[-1]:{} '.format(reward_sequence[-1]) )
            print('self.best_encountered_rewards[-1]:{} '.format(self.best_encountered_rewards[-1]))
            if reward_sequence[-1] > self.best_encountered_rewards[-1]:
                self.best_encountered_rewards = reward_sequence
                self.best_encountered_actions = action_sequence
                self.best_final_state = final_state
            #print('rewards[episode, :]:{},reward_sequence:{}'.format(rewards[episode, :],reward_sequence))
            rewards[episode, :] = reward_sequence

            if self.epsilon >= self.epsilon_min:
                self.epsilon *= self.epsilon_decay

        print(f"Final epsilon: {self.epsilon:.2f}.\n")
        print(f'\nBest encountered rewards (i.e. with best final reward): ',
              [f'{r:.4f}' for r in self.best_encountered_rewards])

        self.env.render(action_sequence=self.best_encountered_actions, best_final_state=self.best_final_state)
        return rewards

    def run_episode(self, verbose=False, mode='explore'):
        raise NotImplementedError('Vanilla DQL has no implementation without'
                                  'ReplayMemory.')
    @abstractmethod
    def solve(self):
        raise NotImplementedError('Vanilla DQL has no implementation without'
                                  'ReplayMemory.')
    



    def save_best_encountered_actions(self, filetype, filename):
        if filetype == 'txt':
            raise ValueError
        elif filetype == 'json':
            try:
                gates = self.env.decode_action_sequence(
                    self.best_encountered_actions
                )
                if len(gates) == 3 or self.env.n_directions == 2:
                    jx_gates, hx_gates, hz_gates, *_ = gates
                    steps = [
                        [('jx', jx_gate), ('hz', list(hz_gate)),
                         ('hx', list(hx_gate))]
                        for jx_gate, hz_gate, hx_gate
                        in zip(jx_gates, hz_gates, hx_gates)
                    ]
                elif len(gates) == 4:
                    raise NotImplementedError
                with open(filename, 'w') as f:
                    json.dump(steps, f, indent=2)
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
    

    

class DQLWithReplayMemory(DeepQLearning):
    """DQN with the addition of a replay memory."""

    def run_episode(self, verbose=False, mode='explore'):
        done = False
        #  Not using TD anymore: q_target_sequence is obsolete
        #  q_target_sequence = []
        action_sequence = []
        reward_sequence = []
        state = self.env.reset()
        step = 0
        while not done:
            time_start_select_action = time.time()
            action = self.select_action(mode=mode,
                                        state=state,
                                        step=step)
            time_end_select_action = time.time()
            self.time_select_action_sum += time_end_select_action - time_start_select_action
            action_sequence.append(action)
            # env.step modifies env.current_state
            # (state is env.current_state)
            time_start_compute_reward = time.time()
            state, reward, done, _ = self.env.step(action,rho_target=self.rho_target)
            reward_sequence.append(reward)
            time_end_compute_reward = time.time()
            self.time_compute_reward_sum += time_end_compute_reward - time_start_compute_reward
            step += 1
            if not done:
                #  q_target_sequence.append(
                #      self.model.get_max_output(step=step,
                #                                state=next_state,
                #                                use_target=True)[1]
                #  )
                pass
        final_state = self.env.final_state
        if verbose:
            print(f'\n----------Total Reward: {reward_sequence[-1]:.2f}')


        episode = Episode(action_sequence,
                          reward_sequence)
        self.memory.push(episode)
        time_start_fit_network = time.time()
        self.fit_network()
        time_end_fit_network = time.time()
        self.time_fit_network_sum += time_end_fit_network - time_start_fit_network
        return reward_sequence, action_sequence, final_state


    def solve(self):
        if not self.seetings_replay_memory_loaded:
            raise RuntimeError("The seetings of the replay memory need to be loaded: run self.load_seetings_replay_memory().")
        # This method runs the deep Q-learning algorithm with experience replay memory. 
        # Run the simulation and get a history of the rewards for each episode.
        start_time = time.time()   
        rho_target = self.get_rho_target_from_other_solver()
        intermediate_time = time.time()

        for simul in range(0,self.n_simulations,1):
            # The loop here is not working because it does not initialize the NN. 
            # Get the initial reward (useful to get the Trotter reward).
            # note: when the initial actions are random, the seed is not the same.
            initial_action_sequence = self.env.initial_action_sequence(False)
            initial_reward = self.env.reward(action_sequence=initial_action_sequence,rho_target=rho_target)
            #  print("The initial reward is (Trotter or random) ", initial_reward)
            rewards = self.run()
            end_time = time.time()
            parametername = 'N'+str(self.env.n_sites)+'episode'+str(self.n_episodes)+'simulations'+str(simul)
            self.save_best_encountered_actions('json',
                                                    'best_gate_sequence'+parametername+'.json')
            try:
                reward_filename = 'rewards'+parametername+'.npy'
                with open(reward_filename, 'wb') as f:
                    np.save(f, rewards)
            except Exception as e:
                print(reward_filename+' could not be saved.')
                print('--->', e)
            info_dic = {
                #  'parameters': parameters,
                'initial_reward': initial_reward,
                'best_reward': str(self.best_encountered_rewards),
                'total_time': end_time - start_time,
                'deep_q_learning_time': end_time - intermediate_time,
                'ed_time': intermediate_time - start_time,
                'best_final_state': str(self.best_final_state),
                'target_state': str(self.state_target),
                'time_fit_network_sum': self.time_fit_network_sum,
                'time_compute_reward_sum': self.time_compute_reward_sum,
                'time_select_action_sum': self.time_select_action_sum,
                'time_reduced_density_matrix': self.env.time_reduced_density_matrix_iteration,
                #  'ground_state_energy': ground_state_energy,
                #  'final_reward': rewards[-1],
                }   
            try: 
                exploration_vs_exploitation_filename = 'exploration_vs_exploitation'+parametername+'.npy'
                with open(exploration_vs_exploitation_filename, 'wb') as f:
                    np.save(f, self.exploration_vs_exploitation)
            except Exception as e:
                print(exploration_vs_exploitation_filename+' could not be saved.')
                print('--->', e)
            try:
                result_info_filename = 'results_info'+parametername+'.json'
                with open(result_info_filename, 'w') as f:
                    json.dump(info_dic, f, indent=2)
                print(result_info_filename+' written.')
            except Exception as e:
                print(result_info_filename+' could not be saved.')
                print('--->', e)



