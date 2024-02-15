#  import math
#  import __main__
from tkinter.ttk import LabeledScale
import numpy as np
import copy 
import sys
from math import pi
from tdqc.numerics.ed.models_ed import Model, xxz_model, lri_model, trans_ising_model, lr_trans_ising_model, trans_field_model
from tdqc.numerics.ed.models_ed import State
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

# Preparation of the target state by taking the ground state of the target Hamiltonian.
L = int(sys.argv[1])
J = float(sys.argv[2])
g = float(sys.argv[3]) 
ferro_angle = float(sys.argv[4]) 
h = g # The notation can be confusing. It is the h of the Transversal Long range Ising model.
alpha = int(2)
model_f = copy.deepcopy(lr_trans_ising_model)
model_f.parametrize_hamiltonian(*[L, J, alpha, h])
ground_state = model_f.ground_state 
vector_to_copy = np.array(ground_state, dtype='complex128')
norm = np.linalg.norm(vector_to_copy)
vector_to_copy /= norm
state_to_copy = State(vector_to_copy)

model_0 = copy.deepcopy(lr_trans_ising_model)
model_0.parametrize_hamiltonian(*[L, 0, alpha, h])

parameters = {
    # =======================================================================
    # physical system
    # =======================================================================
    'n_sites':  L,
    'n_steps': 3,
    't_initial': 0.0,
    't_final': 1.0, # This is the tau in the article.
    #  'periodic_boundary_conditions': True,
    # system_class corresponds to the system_class of the target/final Hamiltonian. It is used to 
    # create the coupling matrix and to define the gate's angles.
    'system_class': 'LongRangeTransIsing', 
    #  also sets entangling gate alpha
    'ham_params': {
        'J': J,
        #  #  g: x, h: z
        'g': g,
        'h': h,
        'alpha': alpha, # In Adrien's code, it was 2.0 but it make more sense to use 3.0 w.r.t. the model of the Hamiltonian.
        #'m_c': 0.5,
        #'w_c': 1.0,
        #'j_c': 1.0
    },
    'model_0': model_0,
    'model_f': model_f,
    'ferro_angle': ferro_angle*pi,
    'ferro_gate_order': 'zy', # It can be 'zy' or 'yz'.
    'seed_initial_state': None, # 42,

    #  digital simulator:
    'n_directions': 2,  # also affect LRI Hamiltonian
    'gate_order': 'zx',
    'entangling_gates_dir': 'jx',
}

