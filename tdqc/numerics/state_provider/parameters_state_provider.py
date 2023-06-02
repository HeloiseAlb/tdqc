#  import math
#  import __main__
from tkinter.ttk import LabeledScale
import numpy as np
import copy 
from tdqc.numerics.ed.models_ed import Model, xxz_model, lri_model
from tdqc.numerics.ed.models_ed import State
#from tdqc_project.tdqc.solver.state_provider import StateProvider
from tdqc.solver.state_provider import StateProvider

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
L = 4 # 10 # Must be the same as n_sites. It is the number of sites in the physical system.
J = 1.0
m_x = 2.0
m_z = 2.0
alpha = int(3)
model = copy.deepcopy(lri_model)
model.parametrize_hamiltonian(*[L,J,alpha,m_x,m_z])
# Initializing state

# Antiferromagnetic
# state_imag = np.zeros(2**L,dtype='complex128')
# spinors = [np.array([1.0, 0.0],dtype='complex128') if _ % 2 == 0
#         else np.array([0.0, 1.0],dtype='complex128') for _ in range(L)]
# state_real = tensor_prod(*spinors)
# Ferromagnetic
state_real = np.zeros(2**L)
state_real[0] = 1.0
state_imag = np.zeros(2**L)

init_vec_state = state_real + 1j*state_imag
norm = np.linalg.norm(init_vec_state)
init_vec_state = init_vec_state / norm



# init_vec_state = np.zeros([2**L],dtype='complex128')
# init_vec_state[0] = 1
# The following vector is obtain for a 2-qubit system.
# vec_to_copy = np.array([-0.20294855-0.03784485j,  0.03034599+0.11942764j , 0.03324365+0.02703225j, -0.0497372 +0.36845858j, -0.00942647-0.1628341j  , 0.49918125-0.3455994j, 0.07135091+0.3496813j ,  0.00771429-0.13628548j , 0.0143802 +0.05334011j,  0.13644914+0.27487965j, -0.16346565-0.0102591j  , 0.13475781+0.11704053j, -0.12321056-0.11372689j , 0.04462397-0.13646653j , 0.09835834+0.09514761j, -0.14764132+0.12542135j], dtype="complex128")# The following vector is obtain for a 2-qubit system.
# The following vector is obtain for a 6-qubit system obtained at the simulation 7.
# vec_to_copy = np.array([0.12910847-0.02859011j,  0.01627676-0.10302107j, -0.09539449+0.07708398j, -0.09127984-0.05982489j, -0.01117133+0.04892622j, -0.13184879+0.05410831j,  0.06127943-0.07622338j, -0.1068764 +0.00400272j , 0.00208037+0.09138308j, -0.0917225 -0.11740984j,  0.05448719-0.03157727j, -0.08775319-0.16820066j,  0.02606498+0.16450769j,  0.10223736+0.07141985j, -0.06596614-0.13592716j,  0.05225618-0.07570849j, -0.11580311-0.05289338j,  0.17265692+0.0113005j,  0.01502646+0.05795121j,  0.24276476+0.04615869j, -0.19441815-0.10304019j, -0.12086012+0.08506657j,  0.18614838+0.03252117j,  0.06694314+0.14004478j,  0.09650848+0.12317385j,  0.10061844+0.02544358j, -0.13024877-0.08806883j,  0.00237678-0.07582125j, -0.02751952-0.02997229j, -0.10917915-0.17486734j,  0.0892477 +0.08644263j, -0.07084656-0.12543564j , 0.05783251+0.02270068j, -0.14333351-0.02814199j,  0.02202941-0.02416509j, -0.16793664-0.03848002j,  0.13022367+0.10160678j, 0.12662787-0.01176266j, -0.14762577-0.06318076j, -0.01029603-0.08198313j, -0.0323928 -0.11957714j, -0.09124349+0.00541013j,  0.07317887+0.07519727j, -0.04508756+0.10282854j,  0.05539344+0.00249373j,  0.06366366+0.11639407j, -0.08236307-0.04846192j,  0.00916654+0.10303039j, -0.09830417-0.01655287j, -0.06291298+0.01707831j,  0.10477571-0.00647115j,  0.0267999 +0.02468936j,  0.01343195+0.02041009j,  0.12714445+0.05070409j, -0.07102355-0.02327353j,  0.09398837+0.02589346j, 0.06443832-0.08090522j,  0.00479091+0.09411392j, -0.0633107 +0.02296497j, -0.0395333 +0.14806307j, 0.05397171-0.10760361j, -0.06141135-0.0598209j, -0.00616118+0.09268187j, -0.0893226+0.04435284j]) 
# vec_to_copy.reshape((2**L))
# norm = np.linalg.norm(vec_to_copy)
# vec_to_copy  = vec_to_copy / norm
# The folllowing state is the same as the initial state ie antiferromagnetic state.
vec_to_copy = init_vec_state

# init_vec_state = np.random.rand((2**L)) +1j*np.random.rand((2**L))

print("init_vec_state:{}".format(init_vec_state))

# init_vec_state = np.ones((2**L),dtype='complex128')


state_to_copy = State(vec_to_copy)
print("vec_to_copy:{}".format(vec_to_copy))

parameters = {
    # =======================================================================
    # physical system (in deep_q_learning, it is for the initialization of the circuit).
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
        'alpha': 3.0,#2.0
        #'m_c': 0.5,
        #'w_c': 1.0,
        #'j_c': 1.0
    },
    
    # 'initial_state': 'random_product_state', 
    # 'initial_state': 'antiferro',
    'initial_state': 'ferro',
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
    'n_simulations': 1,
    # =======================================================================
    # neural networks
    # =======================================================================
    #  'network_type': 'MultiInterStep',
    #  'network_type': 'MultiIntraStep',
    'network_type': 'SingleDense',
    'seed': 2,
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
        'n_initial_actions': 5,
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
