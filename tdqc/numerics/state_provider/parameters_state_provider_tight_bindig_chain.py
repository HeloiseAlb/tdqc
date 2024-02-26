"""
Here, I will run the simulations to prepare the ground state of the tight binding model
in the second quantification. However, the circuit is the one from the previous simulation 
(long range transverse Ising model). That is the structure of the circuit is the 
Trotterization of the long range Ising model. This is expressed in the system_class. 
"""

from tkinter.ttk import LabeledScale
import numpy as np
import copy 
from math import pi 
from tdqc.numerics.ed.models_ed import Model, xxz_model, lri_model, trans_ising_model, lr_trans_ising_model, tb_second_quantization
from tdqc.numerics.ed.models_ed import State
#from tdqc_project.tdqc.solver.state_provider import StateProvider
from tdqc.solver.state_provider import StateProvider
import sys 

def tensor_prod(*arg):
    """
    tensor_prod(a1, a2) = np.kron(a1, a2).
    tensor_prod(a1, a2, ..., an) = np.kron(tensor_prod(a1, ..., an-1), an)
    """
    res = arg[0]
    for i in range(1, len(arg)):
        res = np.kron(res, arg[i])
    #  res = arg[-1]
    #  for i in range(1, len(arg)):
    #      res = np.kron(res, arg[len(arg) - i - 1])
    return res

# Preparation of the target state by taking the ground state of the target Hamiltonian.
# sys.argv = [name_of_the_program, L, J, g]
print("sys.argv:{}".format(sys.argv))
L = int(sys.argv[1])
g = float(sys.argv[2])
ferro_angle = float(sys.argv[3])
alpha = int(2)
model_f = copy.deepcopy(lr_trans_ising_model) # Change it also for system_class !!
model_f.parametrize_hamiltonian(*[L, g])
ground_states = model_f.ground_states 
vector_to_copy = np.array(ground_states, dtype='complex128')
norm = np.linalg.norm(vector_to_copy)
vector_to_copy = vector_to_copy / norm
state_to_copy = State(vector_to_copy)

parameters = {
    # =======================================================================
    # physical system (in deep_q_learning, it is for the initialization of the circuit).
    # =======================================================================
    'n_sites':  L,
    'n_steps': 3,
    't_initial': 0.0,
    't_final': 1.0, # This is the tau in the article.
    #  'periodic_boundary_conditions': True,
    'system_class': 'LongRangeTransIsing',
    #  also sets entangling gate alpha
    'ham_params': {
        'g': g,
    },
    
    # 'initial_state': 'random_product_state', 
    # 'initial_state': 'antiferro',
    'initial_state': 'ferro',#'ferro_with_angle', #'ferro',
    'ferro_angle': ferro_angle*pi,
    'seed_initial_state': None, # 42,

    #  digital simulator:
    'n_directions': 2,  # also affect LRI Hamiltonian
    'gate_order': 'zx',
    'entangling_gates_dir': 'jx',

    # =======================================================================
    # environment and reinforcement learning
    # =======================================================================
    #  'env_type': 'DynamicalEvolution',
    'env_type': 'DynamicalEvolution_cpp',
    'algorithm': 'DQN_ReplayMemory',
    'range_all': 0.2,
    'range_one': 0.4,
    'exploration': 'gaussian',
    #  'exploration': 'uniform'

    #  type of reward
    #  'measurement': 'fidelity',
    'average_exponent': 0.5, #useless

    # q_learning parameters:
    'n_episodes': 50000,#int(5e4),
    #  'n_episodes': 100,

    'epsilon_max': 1.0,
    'epsilon_min': 0.005,
    # corresponds to pp=0.9 with n_episode = 1e5
    'epsilon_decay': 0.9999411315398542,
    'n_epochs': 1,
    'model_update_spacing': 20, #20
    # =======================================================================
    # neural networks
    # =======================================================================
    #  'network_type': 'MultiInterStep',
    #  'network_type': 'MultiIntraStep',
    'network_type': 'SingleDense',
    'seed': None,# 2,
    'architectures': [[(150, 'tanh'),
                       (40, 'relu'),
                       #  (20, 'relu'),
                       (1, 'sigmoid')]],
    #  'architectures': [[(50, 'tanh'),
    #                     (20, 'relu'),
    #                     #  (20, 'relu'),
    #                     (1, 'sigmoid')]],
    #  'architectures': [[(40, 'tanh'),
    #                     (40, 'relu'),
    #                     #  (20, 'relu'),
    #                     (1, 'sigmoid')],
    #                    [(40, 'tanh'),
    #                     (40, 'relu'),
    #                     #  (20, 'relu'),
    #                     (1, 'sigmoid')],
    #                    [(60, 'tanh'),
    #                     (40, 'relu'),
    #                     #  (20, 'relu'),
    #                     (1, 'sigmoid')]],
    #  'max_q_optimizer': {
    #      'algorithm': 'NAG',
    #      'momentum': 0.9,
    #      'learning_rate': 0.6,
    #      #  'learning_rate': 0.2,
    #      #  'n_initial_actions': 5,
    #      'n_initial_actions': 15,
    #      #  'n_initial_actions': 30,
    #      #  'n_iterations': 20,
    #      #  'n_iterations': 500,
    #      'n_iterations': 20,
    #      #  'n_iterations': 5001,
    #      #  'n_iterations': 100,
    #      #  'n_iterations': 6,
    #      'convergence_threshold': 0.005,
    #      #  'convergence_threshold': 0.01,
    #      #  'action_initialization': 'random'
    #      'action_initialization': 'uniform'
    #      #  'action_initialization': 'fixed random'
    #  },

    'max_q_optimizer': {
        # To perform backpropagation on Q_behavior.
        'algorithm': 'adam',
        # The parameters are the 'good default settings' recommended in arXiv:1412.6980.
        'learning_rate': 0.005,#005,#0.6,#005
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-8, 
        #  'n_initial_actions': 5,
        'n_initial_actions': 5, #5 
        #'n_iterations': 1000,
        'n_iterations': 50, #500
        'convergence_threshold': 0.005,
        'action_initialization': 'random'
        #  'action_initialization': 'uniform'
        #  'action_initialization': 'fixed random'
    },

    'target_params':{
            'solver': StateProvider(),
            'mode': 'state_copier',
            'state_to_copy': state_to_copy
            }
    }


parameters_replay_memory = {
    'capacity': 50,
    'sampling_size': 50,
    'NN_optimizer': 'adam',
    'n_epochs': 1
   }