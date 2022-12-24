#  import math
#  import __main__
import numpy as np
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

# init_vec_state = np.zeros([2**L],dtype='complex128')
# init_vec_state[0] = 1
# The following vector is obtain for a 2-qubit system.
# vec_to_copy = np.array([-0.20294855-0.03784485j,  0.03034599+0.11942764j , 0.03324365+0.02703225j, -0.0497372 +0.36845858j, -0.00942647-0.1628341j  , 0.49918125-0.3455994j, 0.07135091+0.3496813j ,  0.00771429-0.13628548j , 0.0143802 +0.05334011j,  0.13644914+0.27487965j, -0.16346565-0.0102591j  , 0.13475781+0.11704053j, -0.12321056-0.11372689j , 0.04462397-0.13646653j , 0.09835834+0.09514761j, -0.14764132+0.12542135j], dtype="complex128")# The following vector is obtain for a 2-qubit system.
# The following vector is obtain for a 6-qubit system obtained at the simulation 7.
vec_to_copy = np.array([-0.00776416-0.00141902j, -0.00045123+0.02713994j,  0.07443662+0.00430856j, -0.00590976+0.0719856j ,  0.02669397+0.01658279j, -0.2615131 +0.04698018j,  0.04720511+0.14883179j , 0.01453464+0.04598341j, -0.04015647+0.05192557j,  0.16845157+0.0354168j ,  0.03434383-0.0147171j  , 0.15139916+0.0639206j,  0.05458165-0.06210865j ,-0.07118071-0.01600911j, -0.04083249+0.04395207j,  0.08594788-0.16115423j , 0.07470854+0.12763305j, -0.15710053+0.10006633j, -0.04253385-0.0671968j , -0.18599517+0.13424171j ,-0.04221719+0.0620238j,  0.3370884 -0.34218147j, -0.24244511-0.08812325j ,-0.01702238-0.03523909j, -0.04416593-0.037682j,   -0.10984262+0.08306095j,  0.0924444 +0.04735838j,-0.16219172-0.06303694j,  0.02695186-0.0178507j,  -0.02873464-0.1207994j, 0.0255809 -0.02367883j, -0.04159889+0.01580215j,  0.01930838-0.04252012j, 0.12750043+0.03421579j , 0.03131049+0.00234575j , 0.06067324+0.09428831j, 0.0112696 -0.03983212j ,-0.27721-0.01175092j , 0.01222635+0.14596057j,  0.02767542-0.05135023j ,-0.01426158-0.00586391j , 0.04024991+0.10337522j,  0.00941154+0.00228621j , 0.04226248+0.15220542j , 0.05138921+0.02356721j, -0.11930622-0.08796607j, -0.05453904+0.09763127j,  0.08351668-0.02695247j, -0.05118652-0.02291108j, -0.04620745+0.0325762j ,  0.03927422+0.00785056j, -0.06658545+0.00838886j, -0.00142783+0.01523059j, -0.00182513+0.03445048j,  0.00708412-0.01371246j , 0.01840916-0.00124699j , 0.04398033+0.06927507j, -0.05785416-0.04651624j, -0.04902033-0.04141333j, -0.00415481-0.01522709j, -0.01137627-0.00660524j , 0.0745512-0.04310416j ,-0.03876914+0.04563832j, -0.06089521-0.1002712j ])
vec_to_copy.reshape((2**L))
norm = np.linalg.norm(vec_to_copy)
vec_to_copy  = vec_to_copy / norm
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
