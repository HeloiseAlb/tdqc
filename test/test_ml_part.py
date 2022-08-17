from mimetypes import init
import iniparse
import pytest
import tdqc
import numpy as np
from math import log2, pi


@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.fast
def test_apply_gate_sequence1():
    from tdqc.numerics.deep_q_learning.environments_cpp import reduced_density_matrix, local_reward, DynamicalEvolution
    from tdqc.solver.deep_q_learning import DeepQLearning, DQLWithReplayMemory
    from tdqc.numerics.ed.models_ed import State
    from tdqc.numerics.ed.models_ed import xxz_model
    from tdqc.numerics.deep_q_learning.parameters_lri import parameters, parameters_replay_memory
    ham_params= {'J': 1.0,
                'g': 2.0,
                'h': 2.0,
                'alpha': 2.0,
                'm_c': 0.5,
                'w_c': 1.0,
                'j_c': 1.0}

    parameters ={'n_sites':2,
                 'n_steps' :3,
                 'n_directions':2,
                 'gate_order':'zx',
                 'system_class':'LongRangeIsing',
                 'ham_params': ham_params,
                 't_initial':0,
                 't_final':1.0,
                 'initial_state':'antiferro',
                 'seed_initial_state':None,
                 'range_one':0.4,
                 'range_all':0.2,
                 'measurement': None,
                 'bulk_size': 0,
                 'entangling_gates_dir':'jx',
                 'average_exponent':1.0,
                 'periodic_boundary_conditions':False}
    env = DynamicalEvolution(**parameters)
    initial_state = env.state
    action_sequence = np.zeros((env.n_steps,env.action_dim))
    jx_gates, hx_gates, hz_gates = env.decode_action_sequence(action_sequence)
    env.system.set_gates(jx_gates, hx_gates, hz_gates)
    final_state = env.apply_gate_sequence()
    true_table = (final_state == initial_state)
    assert true_table.all(), "Applying identity does not modify the state"

@pytest.mark.fast
def test_apply_gate_sequence2():
    from tdqc.numerics.deep_q_learning.environments_cpp import reduced_density_matrix, local_reward, DynamicalEvolution
    from tdqc.solver.deep_q_learning import DeepQLearning, DQLWithReplayMemory
    from tdqc.numerics.ed.models_ed import State
    from tdqc.numerics.ed.models_ed import xxz_model
    from tdqc.numerics.deep_q_learning.parameters_lri import parameters, parameters_replay_memory
    ham_params= {'J': 1.0,
                'g': 2.0,
                'h': 2.0,
                'alpha': 2.0,
                'm_c': 0.5,
                'w_c': 1.0,
                'j_c': 1.0}

    parameters ={'n_sites':2,
                 'n_steps' :3,
                 'n_directions':2,
                 'gate_order':'zx',
                 'system_class':'LongRangeIsing',
                 'ham_params': ham_params,
                 't_initial':0,
                 't_final':1.0,
                 'initial_state':'antiferro',
                 'seed_initial_state':None,
                 'range_one':0.4,
                 'range_all':0.2,
                 'measurement': None,
                 'bulk_size': 0,
                 'entangling_gates_dir':'jx',
                 'average_exponent':1.0,
                 'periodic_boundary_conditions':False}
    env = DynamicalEvolution(**parameters)
    initial_state = env.state
    action_sequence = np.zeros((env.n_steps,env.action_dim))
    jx_gates, hx_gates, hz_gates = env.decode_action_sequence(action_sequence)
    hx_gates[0,0]=pi/2
    hx_gates[0,1]=pi/2
    env.system.set_gates(jx_gates, hx_gates, hz_gates)
    final_state = env.apply_gate_sequence()
    true_table = (final_state == np.array([ 0,0,-1,0],dtype='complex128'))
    assert true_table.all(), "Applying rotation X works"
    

test_apply_gate_sequence1()
test_apply_gate_sequence2()
