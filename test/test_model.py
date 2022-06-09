import pytest
import tdqc
import matplotlib
import matplotlib.pyplot as plt
import cmath
import math
import numpy as np
from tdqc.numerics.ed.models_ed import *
from tdqc.numerics.ed.exact_diagonalisation import *

@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.fast
def test_model_structure():
    from tdqc.numerics.ed.models_ed import Model
    
    # A model must have a Hamilatonian (required for EDSolver.py)
    # A model must have methods to get hamiltonian, eig_values,eig_vectors,ground_state (required for EDSolver.py)

@pytest.mark.fast
def test_lri_model():
    # Models
    model = lri_model
    L = 4
    J = 1
    m_x = 2
    m_z = 2
    alpha = 3
    model.parametrize_hamiltonian(*[L,J,alpha,m_x,m_z])
    gs_per_site_list = []
    site_list = [l for l in range(1,L,1)]
    # The initial state is a uniform superposition.
    initial_state = np.ones([2**L], dtype='complex128')/2**(L-1)
    psi_t_n = State(initial_state)
    #print("psi_t_n.get_state_format_ml():{}".format(psi_t_n.get_state_format_ml()))
    H = model.hamiltonian
    eig_values,eig_vectors = np.linalg.eig(H)
    t_initial = 0
    t_final = 1
    step = 0.1
    exact_diagonalization = ExactDiagonalization(model,L,initial_state,t_final,t_initial,step)
    ground_state = exact_diagonalization.get_ground_state()
    print("psi_t_0:{}".format(psi_t_n._density_mat))
    psi_t_n.time_step_ed( model, delta_t = step, imaginary=False)
    print("psi_t_1:{}".format(psi_t_n._density_mat))
    t_list = [t for t in np.arange(t_initial,t_final,step)]
    pass

test_lri_model()
