#%%
import tdqc
import numpy as np
import cmath
import math
import copy 
from tdqc.numerics.ed.exact_diagonalisation import *
from tdqc.solver.ed import EDSolver
from tdqc.numerics.ed.models_ed import State
from tdqc.numerics.ed.models_ed import xxz_model, lri_model, trans_ising_model, lr_trans_ising_model

def save_state(vector_to_save, L, J, g, model_f):
    parameter_name = 'state_ED_N'+str(L)+'J'+str(J)+'g'+str(g)+'model'+str(model_f.name)
    try:
        reward_filename = parameter_name+'.npy'
        with open(reward_filename, 'wb') as f:
            np.save(f, vector_to_save)
    except Exception as e:
        print(reward_filename+' could not be saved.')
        print('--->', e)

L = 12
J = 1
g = 0.2
h = g
alpha = int(2)
model_f = copy.deepcopy(trans_ising_model) 
model_f.parametrize_hamiltonian(*[L, J, g])
ground_states = model_f.ground_states 
vector_to_copy = np.array(ground_states, dtype='complex128')
norm = np.linalg.norm(vector_to_copy)
vector_to_copy = vector_to_copy / norm

save_state(vector_to_copy, L, J, g, model_f)




# %%
