import numpy as np
from tdqc.numerics.ed.models_ed import Model, trans_ising_model
from tdqc.numerics.ed.models_ed import State

# Contain a set of parameters to run the ED code (exact_diagonalization.py) via test1

L = 10 # 10 # Must be the same as n_sites. It is the number of sites in the physical system.
J = 1.0
h = 2.0
alpha = int(3)
model_0 = trans_ising_model
model_0.parametrize_hamiltonian(*[L,0,alpha,h])
model_f = trans_ising_model
model_f.parametrize_hamiltonian(*[L,J,alpha,h])


parameters = {
    # =======================================================================
    # Physical system
    # =======================================================================
    'model_0': model_0,
    'model_f': model_f,

    # =======================================================================
    # Time simulation
    # =======================================================================
    't_initial': 0,
    't_final': 1,
    'n_steps': 100
    }


