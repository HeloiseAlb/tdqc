import pytest
import tdqc
import numpy as np
import sys 
import cmath
import math
from tdqc.numerics.ed.exact_diagonalisation import *
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
"""



@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.fast
def test_ed_solver_load_settings():
    from tdqc.solver.ed import EDSolver
    from tdqc.numerics.ed.models_ed import Model, xxz_model
    from tdqc.numerics.ed.models_ed import State
    settings = dict()
    settings["steps"] = 3
    L = 4
    Jzz = 1.0
    Jxy = 1.0
    model = xxz_model
    model.parametrize_hamiltonian(*[L,Jxy,Jzz])
    settings["model"] = model
    init_vec_state = np.zeros([2**4],dtype='complex128')
    init_vec_state[0] = 1
    settings["state"] = State(init_vec_state)
    settings["t_initial"] = 0.0
    settings["t_final"] = 1.0
    settings["n_steps"] = int(1/0.001)

    solver = EDSolver()
    solver.load_settings(settings)

@pytest.mark.fast
def test_ed_solver_structure():
    from tdqc.solver.ed import EDSolver
    from tdqc.numerics.ed.models_ed import State
    from tdqc.numerics.ed.models_ed import xxz_model
    settings = dict()
    settings["steps"] = 3
    L = 4
    Jzz = 1.0
    Jxy = 1.0
    model = xxz_model
    model.parametrize_hamiltonian(*[L,Jxy,Jzz])
    
    settings["model"] = model
    init_vec_state = np.zeros([2**4],dtype='complex128')
    init_vec_state[0] = 1
    settings["state"] = State(init_vec_state)
    settings["t_initial"] = 0.0
    settings["t_final"] = 1.0
    settings["n_steps"] = int(1/0.001)

    solver = EDSolver()
    solver.load_settings(settings)
    assert callable(getattr(solver, 'solve', None)), "EDSolver has a method solve"
    # It must be possible to get the list of amplitudes obtained from solved.
    assert hasattr(solver, 'time_evolution'), "EDSolver has an attribut time_evolution"
    solver.solve()
    assert isinstance(getattr(solver, 'time_evolution', None), np.ndarray), "EDSolver method 'solve' returns an array"
    rho_target = solver.get_rho_target()
    assert isinstance(rho_target,np.ndarray), "EDSolver method 'get_target_state' returns an array"


@pytest.mark.slow
def test_ed_solver_solve():
    import tdqc.numerics.ed.models_ed
    from tdqc.numerics.ed.parameters_ed import parameters_ed
    pass



