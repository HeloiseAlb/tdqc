import pytest
import tdqc
import numpy as np
import sys 
import cmath
import math

@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.fast
def test_state_provider_load_settings():
    from tdqc.solver.state_provider import StateProvider
    from tdqc.numerics.ed.models_ed import State
    settings = dict()
    settings["mode"] = "state_copier"
    L = 4
    init_vec_state = np.zeros([2**L],dtype='complex128')
    init_vec_state[0] = 1
    settings["state_to_copy"] = State(init_vec_state)
    solver = StateProvider()
    solver.load_settings(settings)

@pytest.mark.fast
def test_state_provider_structure():
    from tdqc.solver.state_provider import StateProvider
    from tdqc.numerics.ed.models_ed import State
    settings = dict()
    settings["mode"] = "state_copier"
    L = 4
    init_vec_state = np.zeros([2**L],dtype='complex128')
    init_vec_state[0] = 1
    settings["state_to_copy"] = State(init_vec_state)
    solver = StateProvider()
    solver.load_settings(settings)
    assert callable(getattr(solver, 'solve', None)), "StateProvider has a method solve"
    # It must be possible to get the list of amplitudes obtained from solve.
    solver.solve()
    rho_target = solver.get_rho_target()
    assert isinstance(rho_target,np.ndarray), "EDSolver method 'get_target_state' returns an array"
    

def test_state_provider_circuit_provider_mode():
    from tdqc.solver.state_provider import StateProvider
    from tdqc.numerics.ed.models_ed import State
    settings = dict()
    settings["mode"] = "circuit_copier"
    L = 4
    n_steps = 3
    init_vec_state = np.zeros([2**L],dtype='complex128')
    init_vec_state[0] = 1
    settings["initial_state"] = State(init_vec_state)
    settings["jx_angle_list"] = np.zeros(n_steps)
    settings["hx_angle_list"] = np.zeros((n_steps,L))
    settings["hz_angle_list"] = np.zeros((n_steps,L))
    settings["gate_order"] = 'zx'
    settings["alpha"] = 3
    solver = StateProvider()
    solver.load_settings(settings)
    assert callable(getattr(solver, 'solve', None)), "StateProvider has a method solve"
    # It must be possible to get the list of amplitudes obtained from solve.
    solver.solve()
    rho_target = solver.get_rho_target()
    assert isinstance(rho_target,np.ndarray), "StateProvider method 'get_target_state' returns an array"
    print('rho_target:{}'.format(rho_target))
