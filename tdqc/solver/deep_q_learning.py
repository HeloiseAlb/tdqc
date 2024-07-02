"""Module defining the deep Q learning algorithms.

Contains the classes DeepQLearning and DQLWithReplayMemory.
Only the derived DQLWithReplayMemory is usable. 
Originally (during developement) other versions were implemented
    (e.g. Deep Q learning with eligibility traces with the backward view, which
    is not compatible with the replay memory)'
    but only DQLWithReplayMemory made it.

For a theoretical description of the algorithms:
    R. S. Sutton and A. G. Barto, Reinforcement learning: An introduction


Update:
    The algorithm currently implemented does not use Temporal Diffence anymore.
    Instead, Monte Carlo updates are performed.
    (Yes, it is not really Q-learning then)

    MC update (currently used):
        target Q(s_t, a_t) = sum_{t'>=t} reward(t')
    Q-learning update (previously used):
        target Q(s_t, a_t) = reward(t) + max Q(s_{t+1}, :)

    (also, in the environement used, reward(t) = 0 for all t < t_final.)

    States are implemented in such a way that a given state can only be visited  
    once during an episode (different steps always have different states).
    Therefore, there is no drawback in waiting for the end of an episode before
    perfoming the update. 
    This is a more efficient because the max Q operation can be quite heavy.

"""
from abc import ABCMeta, abstractmethod
import random
from collections import namedtuple
import numpy as np
import sys 
import json
#from pathlib import Path
import time
#import environments as envs
from tdqc.interfaces.solver import Solver
import tdqc.numerics.deep_q_learning.environments_cpp as envs_cpp
import tdqc.numerics.deep_q_learning.models as models
from tdqc.numerics.ed.models_ed import State

class DeepQLearning(Solver):
    """Basic abstract class for Deep Q-Learning.
    Two Neural Networks are used for the Q-function.
    One for the behavior-policy, and one for the target-policy.
    (behavior NN and target NN)
    The target NN is frozen and gets periodically updated with the parameters
    of the behavior NN, while the behavior NN is continuously trained.

    Only derived classes are fully implemented.
    """
    def __init__(self,):
        super().__init__()
        self.__check_validity__()

    def load_settings(self, settings):
        """
        Initialize settings stored in local variable self.__settings
        """
        if not "n_episodes" in settings:
            raise ValueError("Error loading deep_q_learning-solver settings, 'n_episodes' parameter not found")
        self.n_episodes = settings["n_episodes"]
        if not "epsilon_max" in settings:
            raise ValueError("Error loading deep_q_learning-solver settings, 'epsilon_max' parameter not found")
        self.epsilon = settings["epsilon_max"]
        if not "epsilon_min" in settings:
            raise ValueError("Error loading deep_q_learning-solver settings, 'epsilon_min' parameter not found")
        self.epsilon_min = settings["epsilon_min"]
        if not "epsilon_decay" in settings:
            raise ValueError("Error loading deep_q_learning-solver settings, 'epsilon_decay' parameter not found")
        self.epsilon_decay = settings["epsilon_decay"]
        if not "model_update_spacing" in settings:
            raise ValueError("Error loading deep_q_learning-solver settings, 'model_update_spacing' parameter not found")
        self.model_update_spacing = settings["model_update_spacing"]
        if not "network_type" in settings:
            raise ValueError("Error loading deep_q_learning-solver settings, 'network_type' parameter not found")
        self.network_type = settings["network_type"]
        if not "env_type" in settings:
            raise ValueError("Error loading deep_q_learning-solver settings, 'env_type' parameter not found")
        self.env_type = settings["env_type"]
        if not "exploration" in settings:
            raise ValueError("Error loading deep_q_learning-solver settings, 'exploration' parameter not found")
        self.exploration = settings["exploration"]
        if not "seed" in settings:
            self.seed = None
        else:
            self.seed = settings["seed"]
        if not "ham_params" in settings:
            raise ValueError("Error loading deep_q_learning-solver settings, 'ham_params' parameter not found")
        if not "name_for_file" in settings:
            self.name_for_file = "lrti"
        else: 
            self.name_for_file = settings["name_for_file"]
        self.ham_params = settings["ham_params"]
        self.__target_params = settings["target_params"]
        self.__target_params["t_initial"] = settings["t_initial"]
        self.__target_params["t_final"] = settings["t_final"]
        self.t_final = settings["t_final"]

        if self.env_type == 'DynamicalEvolution_cpp':
            self.env = envs_cpp.DynamicalEvolution(
                 **settings
            )
            #  old environment types
        #  elif env_type == 'EnergyMinimizer_cpp':
        #      self.env = envs_cpp.EnergyMinimizer(
        #          system_class=system_class, **other_params
        #      )
        #  elif env_type == 'DynamicalEvolution':
        #      self.env = envs.ContinuousCurrentGateEnv(system_class,
        #                                               **other_params)
        #  elif env_type == 'EnergyMinimizer':
        #      self.env = envs.ContinuousCurrentGateEnergyMinimizer(
        #          system_class=system_class, **other_params
        #      )

        if self.network_type == 'SingleDense':
            action_dim = self.env.get_action_dim()
            self.model = models.SingleDeepQNetwork(tf_seed=self.seed,
                                                   action_dim=action_dim,
                                                   **settings)
        elif self.network_type == 'MultiInterStep':
            action_dim = self.env.get_action_dim()
            self.model = models.InterStepMultiDQN(tf_seed=self.seed,
                    action_dim=action_dim,
                                                  **settings)
        elif self.network_type == 'MultiIntraStep':
            action_dim = self.env.get_action_dim()
            n_sites = self.env.get_n_sites()
            action_dims = [1, n_sites, n_sites]
            assert sum(action_dims) == action_dim
            self.model = models.IntraStepMultiDQN(tf_seed=self.seed,
                                                  action_dims=self.action_dims,
                                                  **settings)
        else:
            raise ValueError("network_type not recognized.")

        # random is only used for mini_batch sampling
        # (np.random.choice does not like the list of Episode namedtuples)
        random.seed(self.seed)
        np.random.seed(self.seed)
        #self.best_encountered_actions = None
        #self.best_encountered_rewards = None
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
        self.time_gradient_ascent = 0
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
    

    def select_action(self, mode, state, step=0):
        time_start_gradient_ascent = time.time()
        action, _ = self.model.get_max_output(step=step,
                                              state=state,
                                              use_target=False)
        self.time_gradient_ascent += time.time() - time_start_gradient_ascent
        if mode == 'greedy':
            pass
        elif mode == 'explore':
            if self.exploration == 'uniform':
                if np.random.rand() < self.epsilon:
                    action = self.env.random_action()
            elif self.exploration == 'gaussian':
                # add gaussian fluctuation with std = 0.5*ε
                # (ε = 1 -> 2σ = 1 -> 95% inside [-1, 1])
                action_addition = self.epsilon * 0.5 * np.random.randn(*action.shape)
                action += action_addition
                self.exploration_vs_exploitation.append(action_addition)
            else:
                raise NotImplementedError
        else:
            raise ValueError(f'The action selecting mode {mode} does not '
                             'exist.')
        return action

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
        if not "state" in target_params:
            initial_state = self.env.initial_state
            n_sites = self.env.get_n_sites()
            np.reshape(initial_state, (2**n_sites))
            target_params['state'] = State(initial_state)
        solver_for_target = target_params['solver']
        #target_params.pop('solver', None)
        solver_for_target.load_settings(target_params)
        solver_for_target.solve()
        self.state_target = solver_for_target.get_state_target()
        rho_target = solver_for_target.get_rho_target()
        self.rho_target = rho_target
        return rho_target
    

    

class DQLWithReplayMemory(DeepQLearning):
    """DQN with the addition of a replay memory."""

    def __init__(self,):
        super().__init__()
        self.seetings_replay_memory_loaded = False    
        
    def load_seetings_replay_memory(self, capacity, sampling_size, NN_optimizer, n_epochs,loss='logcosh', *args, **kwargs):

        self.model.compile(optimizer=NN_optimizer, loss=loss,
                           metrics=['mse', 'mae'])
        self.loss = loss
        self.n_epochs = n_epochs
        #  metrics=['accuracy']
        self.memory = ReplayMemory(capacity)
        self.sampling_size = sampling_size
        #  self.batch_size = sampling_size * self.env.n_steps
        self.seetings_replay_memory_loaded = True
        
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

    def fit_network(self, memory=None, sampling_size=None, epochs=None):
        """Update policy network using a batch of episodes sampled from memory
        (`episodes` is a list of `Episode` namedtuples.)
        """
        if memory is None:
            memory = self.memory
        if sampling_size is None:
            sampling_size = self.sampling_size
        if epochs is None:
            epochs = self.n_epochs
        if(len(memory) < sampling_size):
            return None
        episodes = memory.sample(sampling_size)
        batch = Episode(*zip(*episodes))

        #  action_sequences: shape(sampling_size, n_steps, action_dim)
        #  ys: shape(sampling_size, n_steps)
        action_sequences = np.array(batch.action_sequence)
        ys = np.array(batch.reward_sequence)

        #  target Q(s_t, a_t) = sum_{t'>=t} r_t = r_{t_final}
        #  NOTE: I use the fact that all rewards are 0.0 except the final one.
        #  More generally, one would have to do a cummulative sum of ys along the
        #  axis=1 starting from the end.
        final_rewards = ys[:, -1].reshape(-1, 1)
        
        # ys becomes an array containing the final_reward (r_{t_final}) of the sampling for each steps of the episode
        # that is for each of r_t.
        ys[:, :-1] = np.tile(final_rewards, (1, ys.shape[1] - 1))

        self.model.fit(action_sequences, ys, sampling_size, epochs)

    def solve(self):
        if not self.seetings_replay_memory_loaded:
            raise RuntimeError("The seetings of the replay memory need to be loaded: run self.load_seetings_replay_memory().")
        # This method runs the deep Q-learning algorithm with experience replay memory. 
        # Run the simulation and get a history of the rewards for each episode.
        start_time = time.time()   
        rho_target = self.get_rho_target_from_other_solver()
        intermediate_time = time.time()

        # note: when the initial actions are random, the seed is not the same.
        initial_action_sequence = self.env.initial_action_sequence(False)
        initial_reward = self.env.reward(action_sequence=initial_action_sequence,rho_target=rho_target)        
        rewards = self.run()
        end_time = time.time()
        if self.name_for_file == "lrti":
            # I need to change the previous parameter files to create that is the parameter file. 
            # Here, I did a change to have it working despite the difference of structure between the 
            # parameter files. 
            parametername = self.name_for_file+'_PD_N'+str(self.env.n_sites)+'episode'+str(self.n_episodes)+'t_final'+str(self.t_final)+'alpha'+str(self.ham_params['alpha'])+'J'+str(self.ham_params['J'])+'h'+str(self.ham_params['h'])+'ferro_angle'+str(sys.argv[4])+'sim'+str(sys.argv[5])
        else: 
            parametername = self.name_for_file
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
            'parameters': self.ham_params,
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
            'time_gradient_ascent': self.time_gradient_ascent, 
            'initial_state': str(self.env.initial_state), 
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
            best_final_state_filename = 'best_final_state'+parametername+'.npy'
            with open(best_final_state_filename, 'wb') as f:
                np.save(f, self.best_final_state)
        except Exception as e:
            print(best_final_state_filename+' could not be saved.')
            print('--->', e)
        try:
            result_info_filename = 'results_info'+parametername+'.json'
            with open(result_info_filename, 'w') as f:
                json.dump(info_dic, f, indent=2)
            print(result_info_filename+' written.')
        except Exception as e:
            print(result_info_filename+' could not be saved.')
            print('--->', e)
        try:
            target_state_filename = 'target_state'+parametername+'.npy'
            with open(target_state_filename, 'wb') as f:
                np.save(f, self.state_target)
        except Exception as e:
            print(reward_filename+' could not be saved.')
            print('--->', e)


Episode = namedtuple('Episode', ('action_sequence',
                                 'reward_sequence'))
                                 

class ReplayMemory(object):
    """Replay memory that stores full episodes during the Q-learning"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, episode, n_pushes=1):
        """Saves a transition."""
        for _ in range(n_pushes):
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = episode
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        #  return np.random.choice(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
