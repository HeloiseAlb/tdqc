import pytest
import tdqc
import numpy as np
from math import log2, sqrt

def tensor_prod(*arg):
    """
    tensor_prod(a1, a2) = np.kron(a1, a2).
    tensor_prod(a1, a2, ..., an) = np.kron(tensor_prod(a1, ..., an-1), an)
    """
    res = arg[0]
    for i in range(1, len(arg)):
        res = np.kron(res, arg[i])
    return res

@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.fast
def test_reward_fixation_infinity_problem():
    from tdqc.numerics.deep_q_learning.environments_cpp import reduced_density_matrix, local_reward
    psi1 = np.array([0,1,0,0],dtype=np.complex128) #|01>
    psi2 = np.array([1,0,0,0],dtype=np.complex128) #|00>
    rho1 = np.tensordot(np.conjugate(psi1), psi1, axes=0)
    rho2 = np.tensordot(np.conjugate(psi2), psi2, axes=0)
    assert local_reward(rho1,rho2)==0.0, "The local reward between matrices such as supp ( ρ ) ∩ ker ( σ ) ≠ 0 is zero."

@pytest.mark.fast
def test_reward_identical_matrices():
    from tdqc.numerics.deep_q_learning.environments_cpp import local_reward, reduced_density_matrix, globalize_op, relative_entropy
    # It is a case where the quantum relative entropy should be 0 according to 11.3.1 of Nielsen and Chuang.
    psi1 = np.array([0,1,0,0],dtype='complex128') #|01>
    psi2 = np.array([1,0,0,0],dtype='complex128') #|00>
    rho1 = np.tensordot(np.conjugate(psi1), psi1, axes=0)
    rho2 = np.tensordot(np.conjugate(psi2), psi2, axes=0)
    rho_rand = np.random.rand(4,4)
    norm = np.linalg.norm(rho_rand)
    rho_rand = rho_rand/norm
    n_qubits = int(log2(rho1.shape[0]))
    assert local_reward(rho_rand,rho_rand,n_qubits=None)==1.0, "The local reward between 2 identical matrix is 1."

@pytest.mark.fast
def test_reward_correctness_of_the_value():
    # I printed it and checked the values.
    from tdqc.numerics.deep_q_learning.environments_cpp import reduced_density_matrix, local_reward
    np.random.seed(seed=None)
    rho1 = np.matrix(np.random.rand(4,4))    
    rho1 = np.dot(rho1,rho1.getH())
    rho1 = rho1/np.linalg.norm(rho1)
    rho2 = np.matrix(np.random.rand(4,4))
    rho2 = np.dot(rho2,rho2.getH())
    rho2 = rho2/np.linalg.norm(rho2)
    print("rho1:{}".format(rho1))
    print("rho1:{}".format(rho2))
    print("local_reward of rho1 and rho2:{}".format(local_reward(rho1,rho2)))
    #assert local_reward(rho1,rho2)==0.0, "The local reward between matrices such as supp ( ρ ) ∩ ker ( σ ) = 0 is computed correctly."

@pytest.mark.fast
def test_relative_entropy():
    from scipy.linalg import logm, inv
    from tdqc.numerics.deep_q_learning.environments_cpp import reduced_density_matrix, local_reward, relative_entropy
    
    #np.random.seed(seed=1)
    rho1 = np.matrix(np.random.rand(4,4))
    rho1 = np.dot(rho1,rho1.getH()) # To guarantee that rho1 is a Hermitian positive semi-definite matrix.
    rho1 = rho1/np.linalg.norm(rho1)
 
    #np.random.seed(seed=2)
    rho2 = np.matrix(np.random.rand(4,4))
    rho2 = np.dot(rho2,rho2.getH()) # To guarantee that rho2 is a Hermitian positive semi-definite matrix. 
    rho2 = rho2/np.linalg.norm(rho2)
    
    #rho1 = np.matrix([[1,0],[0,1]])
    #rho2 = np.matrix([[2,0],[0,2]])
    #rho2 = rho2/np.linalg.norm(rho2)
    #rho2 = np.matrix([[1/sqrt(2),-1/sqrt(2)],[1/sqrt(2),1/sqrt(2)]])
    #rho2 = np.matrix([[1/2,-1j*sqrt(3)/2],[1j*sqrt(3)/2,1/2]])
    relative_entropy_instable = np.trace(np.dot(rho1,logm(np.dot(rho1, np.linalg.inv(rho2)))))
    relative_entropy_instable2 = np.trace(np.dot(rho1,(logm(rho1)-logm(rho2))))
    relative_entropy_instable3 = np.trace(np.dot(rho1,logm(rho1))-np.trace(np.dot(rho1,logm(rho2))))
    print("relative_entropy_instable:{}".format(relative_entropy_instable))
    print("relative_entropy_instable2:{}".format(relative_entropy_instable2))
    print("relative_entropy_instable3:{}".format(relative_entropy_instable3))
    relative_entropy_stable = relative_entropy(rho1,rho2,1)
    print("relative_entropy_stable:{}".format(relative_entropy_stable))
    assert abs(relative_entropy_stable-relative_entropy_instable2)<= 0.01, "The relative_entropy is computed correctly."


@pytest.mark.fast
def test_reward_negativity():
    # To use it one need to change the line "return max(0,r_local)" with "return r_local" in the function local_reward() in environments_cpp.py.
    #from tdqc.numerics.deep_q_learning.environments_cpp import reduced_density_matrix, local_reward, relative_entropy
    from tdqc.solver.deep_q_learning import DeepQLearning, DQLWithReplayMemory
    from tdqc.numerics.ed.models_ed import Model, xxz_model, lri_model
    from tdqc.numerics.ed.models_ed import State
    from tdqc.solver.ed import EDSolver
    #from tdqc.numerics.deep_q_learning.parameters_lri import parameters
    # Initializing model
    L = 2 # 10 # Must be the same as n_sites. It is the number of sites in the physical system.
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

    parameters = {
    # =======================================================================
    # physical system
    # =======================================================================
    'n_sites':  L,
    'n_steps': 3,
    't_initial': 0.0,
    't_final': 1.0, # This is the tau in the article.
    'system_class': 'LongRangeIsing',
    #  also sets entangling gate alpha
    'ham_params': {
        'J': 1.0,
        'g': 2.0,
        'h': 2.0,
        'alpha': 2.0,
        'm_c': 0.5,
        'w_c': 1.0,
        'j_c': 1.0
    },
    'initial_state': 'antiferro',
    'seed_initial_state': None, # 42,
    'n_directions': 2,  # also affect LRI Hamiltonian
    'gate_order': 'zx',
    'entangling_gates_dir': 'jx',
    'env_type': 'DynamicalEvolution_cpp',
    'algorithm': 'DQN_ReplayMemory',
    'range_all': 0.2,
    'range_one': 0.4,
    'exploration': 'gaussian',
    'average_exponent': 0.5, #  type of reward
    'n_episodes': 10,# q_learning parameters:

    'epsilon_max': 1.0,
    'epsilon_min': 0.005, # corresponds to pp=0.9 with n_episode = 1e5
    'epsilon_decay': 0.9999411315398542,
    'n_epochs': 1,
    'model_update_spacing': 20, # what is that ?
    'network_type': 'SingleDense',
    'architectures': [[(150, 'tanh'),
                       (40, 'relu'),
                       #  (20, 'relu'),
                       (1, 'sigmoid')]],
    'max_q_optimizer': {
        'algorithm': 'adam',
        'learning_rate': 0.6,#005,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-8,
        'n_initial_actions': 5,
        'n_iterations': 50, #500
        'convergence_threshold': 0.005,
        'action_initialization': 'random'
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
    solver = DQLWithReplayMemory()
    solver.load_settings(settings=parameters)
    solver.load_seetings_replay_memory(**parameters_replay_memory)
    solver.solve()
    







test_reward_negativity() 



