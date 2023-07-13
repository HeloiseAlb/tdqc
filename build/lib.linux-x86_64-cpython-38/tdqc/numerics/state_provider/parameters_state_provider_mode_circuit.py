#  import math
#  import __main__
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
L = 6 # 10 # Must be the same as n_sites. It is the number of sites in the physical system.
J = 1.0
m_x = 2.0
m_z = 2.0
alpha = int(3)
model = copy.deepcopy(lri_model)
model.parametrize_hamiltonian(*[L,J,alpha,m_x,m_z])
n_steps = 3


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

   

# The parameters of the gates are the one obtained in the simulations 19 with 6 qubits.

    'target_params':{
            'solver': StateProvider(),
            'mode': 'circuit_copier',
            'n_sites': L,
            'initial_state':  'antiferro',
            # Gates to apply from best_gate_sequenceN6episode50000simulations0_53
            'jx_angle_list':  np.array([0.17608174725785508,0.28184247184071526,0.25794581618280804,]),
            'hx_angle_list': np.array([[ -0.11922617126574848, -0.132527526454309, -0.1271589901619128, 0.17039545638156942, 0.20979017282010481, 0.3400055472457961],[-0.5429056458076297,-0.03220951425070164,-0.057697662588413336,0.10611445027684326,0.041294782570169164,0.16244891006425172],[-0.20195797076040886,0.07677985900551919,-0.09941066928110837,0.07493298762909732,-0.01704175336486498,0.32652464783399565]]),
            'hz_angle_list': np.array([[-0.1387266346582304,-0.21182746854842774,-0.3377540675217936,0.0957978358059864,0.15607311976657468,-0.31086424967833265],[0.3543519384643863,0.18913150484906094,0.225781671983032,-0.021018463741297705,0.21544702299288998,-0.06792831418820755],[-0.08452768620945332,0.18310981462218978,-0.05731513818331313,0.3175992742295699,-0.00926905846541648,-0.2825688787067737]]),
            'gate_order': 'zx',
            'alpha': 3
            }
    }


parameters_replay_memory = {
    'capacity': 50,
    'sampling_size': 50,
    'NN_optimizer': 'adam',
    'n_epochs': 1
   }
