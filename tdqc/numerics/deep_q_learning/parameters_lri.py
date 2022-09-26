#  import math
#  import __main__
import numpy as np
from tdqc.numerics.ed.models_ed import Model, xxz_model, lri_model
from tdqc.numerics.ed.models_ed import State
from tdqc.solver.ed import EDSolver

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

# Initializing model
L = 6 # 10 # Must be the same as n_sites. It is the number of sites in the physical system.
J = 1.0
m_x = 2.0
m_z = 2.0
alpha = int(3)
model = lri_model
model.parametrize_hamiltonian(*[L,J,alpha,m_x,m_z])
# Initializing state
state_imag = np.zeros(2**L,dtype='complex128')
spinors = [np.array([1.0, 0.0],dtype='complex128') if _ % 2 == 0
        else np.array([0.0, 1.0],dtype='complex128') for _ in range(L)]
state_real = tensor_prod(*spinors)
init_vec_state = state_real + 1j*state_imag
norm = np.linalg.norm(init_vec_state)
init_vec_state = init_vec_state / norm
#init_vec_state = np.zeros([2**L],dtype='complex128')
#init_vec_state[0] = 1


parameters = {
    # =======================================================================
    # physical system
    # =======================================================================
    'n_sites':  L,
    'n_steps': 3,
    't_initial': 0.0,
    't_final': 1.0, # This is the tau in the article.
    #  'periodic_boundary_conditions': True,
    'system_class': 'LongRangeIsing',
    #  also sets entangling gate alpha
    'ham_params': {
        'J': 1.0,
        #  #  g: x, h: z
        'g': 2.0,
        'h': 2.0,
        'alpha': 2.0,
        'm_c': 0.5,
        'w_c': 1.0,
        'j_c': 1.0
    },
    
    #  'initial_state': 'random_product_state',
    'initial_state': 'antiferro',
    #  'initial_state': 'ferro',
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
    'average_exponent': 0.5,

    # q_learning parameters:
    'n_episodes': 50000,#int(5e4),
    #  'n_episodes': 100,

    'epsilon_max': 1.0,
    'epsilon_min': 0.005,
    # corresponds to pp=0.9 with n_episode = 1e5
    'epsilon_decay': 0.9999411315398542,
    'n_epochs': 1,
    'model_update_spacing': 20, #20
    'n_simulations': 1,
    # =======================================================================
    # neural networks
    # =======================================================================
    #  'network_type': 'MultiInterStep',
    #  'network_type': 'MultiIntraStep',
    'network_type': 'SingleDense',
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

    'max_q_optimizer': { # To perform backpropagation on Q_behavior
        'algorithm': 'adam',
        # learning rate
        'learning_rate': 0.6,#005,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-8,
        #  'n_initial_actions': 5,
        'n_initial_actions': 5,
        #'n_iterations': 1000,
        'n_iterations': 50, #500
        'convergence_threshold': 0.005,
        'action_initialization': 'random'
        #  'action_initialization': 'uniform'
        #  'action_initialization': 'fixed random'
    },

    'target_params':{
            'solver': EDSolver(),
            'n_steps': int(1/0.001), # time steps 
            'model': model,
            'state': State(init_vec_state)
            }


    }

parameters_replay_memory = {
    'capacity': 50,
    'sampling_size': 50,
    'NN_optimizer': 'adam',
    'n_epochs': 1
   }
