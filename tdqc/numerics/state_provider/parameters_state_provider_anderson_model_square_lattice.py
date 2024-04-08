"""
Here, I will run the simulations to prepare the ground state of the tight binding model
in the second quantification. However, the circuit is the one from the previous simulation 
(long range transverse Ising model). That is the structure of the circuit is the 
Trotterization of the long range Ising model. This is expressed in the system_class. 
For that reason, we also need the parameters for the LRTI model.  
"""
#%%
from tkinter.ttk import LabeledScale
import numpy as np
import copy 
from math import pi, sqrt
from tdqc.numerics.ed.models_ed import Model, State, anderson_impurity_model
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
# sys.argv = [name_of_the_program, L, ferro_angle, sim]
print("sys.argv:{}".format(sys.argv))
L = int(sys.argv[1]) # L=2 <=> 2 impurity sites; L=4 <=> 2 impurity sites and 2 spin sites
g = 1
E_k = np.array([-1.932842482271276863, -1.160297051002382007, -3.869015369842733199, 3.865109256039230834, 1.159906462954387374])
V_k = np.array([5.586605759559575696e-2, 1.001269500844226695e-1 ,1.098938938616941946e-1, 1.098960499514210210e-1,1.001355017292177296e-1])
E = 0
U = 8
h = g
ferro_angle = float(sys.argv[2])
J = 1
alpha = int(2)
model_f = copy.deepcopy(anderson_impurity_model) # Change it also for system_class !!
model_f.parametrize_hamiltonian(*[L, E_k, V_k, E, U])
ground_state = model_f.ground_state 
# print("ground_state:{}".format(ground_state))
vector_to_copy = np.array(ground_state, dtype='complex128')
norm = np.linalg.norm(vector_to_copy)
vector_to_copy = vector_to_copy / norm
state_to_copy = State(vector_to_copy)

# Build the initial state
# It needs to be taken into account in the computation. 
init_vec_state = np.zeros(2**L, dtype='complex128')
init_vec_state[2032] = 1.0/sqrt(2)
init_vec_state[3056] = 1.0/sqrt(2)
init_vec_state = init_vec_state / np.linalg.norm(init_vec_state)


# I put the followimg parameters outside of the dictionary because they also appear in 
# the name_for_file entry. 
n_episodes = 50000
t_final = 1.0 # This is the tau in the article.
parameters = {
    # =======================================================================
    # physical system (in deep_q_learning, it is for the initialization of the circuit).
    # =======================================================================
    'name_for_file': 'Anderson_square_fermions_PD_N'+str(L)+'episode'+str(n_episodes)+'t_final'+str(t_final)+'E'+str(E)+'U'+str(U)+'ferro_angle'+str(ferro_angle)+'sim'+str(int(sys.argv[3])),
    'n_sites': L,
    'n_steps': 3,
    't_initial': 0.0,
    't_final': t_final, # This is the tau in the article.
    #  'periodic_boundary_conditions': True,
    'system_class': 'LongRangeTransIsing',
    #  also sets entangling gate alpha
    'ham_params': {
        'g': g,
        'alpha': int(2),
        'h': h,
        'J': J
    },
    
    # 'initial_state': 'random_product_state', 
    # 'initial_state': 'antiferro',
    'initial_state': 'predefined_state',#'ferro',#'ferro_with_angle', #'ferro',
    'ferro_angle': ferro_angle*pi,
    'seed_initial_state': None, # 42,
    'predefined_init_vec':init_vec_state,

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
    'n_episodes': n_episodes,#50000,#int(5e4),
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
# %%
