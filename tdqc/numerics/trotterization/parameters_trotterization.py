#  import math
#  import __main__
import numpy as np
import copy
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
L = 8 # 10 # Must be the same as n_sites. It is the number of sites in the physical system.
J = 1.0
m_x = 2.0
m_z = 2.0
alpha = int(3)
model = copy.deepcopy(lri_model)
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
        'alpha': 3.0, #2.0
        'm_c': 0.5,
        'w_c': 1.0,
        'j_c': 1.0
    },
    
    #'initial_state': 'random_product_state',
    'initial_state': 'antiferro',
    #'initial_state': 'ferro',
    #'initial_state': 'ground_state',
    'seed_initial_state': 42, # None 42, #useful to determined only if 'initial_state'=='random_product_state'

    #  digital simulator:
    'n_directions': 2,  # also affect LRI Hamiltonian
    'gate_order': 'zx',
    'entangling_gates_dir': 'jx',
    'range_all': 1, # 0.2,
    'range_one': 1, # 0.4,

    'target_params':{
            'solver': EDSolver(),
            'n_steps': int(1/0.1), # Number of time steps 
            'model': model,
            'state': State(init_vec_state)
            }
    }
