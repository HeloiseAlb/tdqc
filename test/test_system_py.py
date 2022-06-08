import pytest
import tdqc
import numpy as np
from tdqc.numerics.deep_q_learning.system_py.system import SpinSystem

@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.fast
def test_set_coupling_matrix():
    from tdqc.numerics.deep_q_learning.parameters import parameters
    system = SpinSystem()
    n_sites = parameters['n_sites']
    n_steps = parameters['n_steps']
    t_initial = parameters['t_initial']
    t_final = parameters['t_final']
    gate_order = parameters['gate_order']
    ham_params = parameters['ham_params']
    alpha = ham_params['alpha']

    system.set_system(n_sites, n_steps, t_initial, t_final, gate_order, alpha)
    print(system.coupling_matrix)
    assert isinstance(getattr(system,'coupling_matrix', None), np.ndarray), "coupling_matrix is an array"


@pytest.mark.fast
def test_set_gates():
    from tdqc.numerics.deep_q_learning.parameters import parameters
    system = SpinSystem()
    n_sites = parameters['n_sites']
    n_steps = 3 #parameters['n_steps']
    t_initial = parameters['t_initial']
    t_final = parameters['t_final']
    gate_order = parameters['gate_order']
    ham_params = parameters['ham_params']
    alpha = ham_params['alpha']

    system.set_system(n_sites, n_steps, t_initial, t_final, gate_order, alpha)
    jx_angle_list = np.array([0.10,0.08,0.38])
    unit_array = np.array([0.07,0.01,0.05,0.02,0.08,0.01,0.03,0.01,0.04,0.04,0.06,-0.01,0.09,0.01,0.12,0.01])
    hx_angle_list = np.array([unit_array,unit_array,unit_array])
    hz_angle_list = np.array([unit_array,unit_array,unit_array])
    system.set_gates(jx_angle_list, hx_angle_list, hz_angle_list)

@pytest.mark.fast
def test_start():
    
    pass
