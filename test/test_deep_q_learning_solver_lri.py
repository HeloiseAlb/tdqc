import pytest
import tdqc
import numpy as np

@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.fast
def test_deep_q_learning_solver_load_settings():
    from tdqc.solver.deep_q_learning import DeepQLearning, DQLWithReplayMemory
    from tdqc.numerics.ed.models_ed import Model, xxz_model
    from tdqc.numerics.ed.models_ed import State
    from tdqc.numerics.deep_q_learning.parameters_lri import parameters
    '''
    settings = dict()
    settings["steps"] = 3
    L = 10
    Jzz = 1.0
    Jxy = 1.0
    model = xxz_model
    init_vec_state = np.zeros([2**4],dtype='complex128')
    init_vec_state[0] = 1
    settings["state"] = State(init_vec_state)
    settings["t_initial"] = 0.0
    settings["t_final"] = 1.0
    settings["step"] = 0.001
    settings["capacity"] = 50
    settings["sampling_size"] = 50
    settings["NN_optimizer"] = 'adam'
    settings["n_epochs"] = 1
    '''
    solver = DQLWithReplayMemory()
    solver.load_settings(parameters)

@pytest.mark.fast
def test_deep_q_learning_solver_structure():
    from tdqc.solver.deep_q_learning import DeepQLearning, DQLWithReplayMemory
    from tdqc.numerics.ed.models_ed import State
    from tdqc.numerics.ed.models_ed import xxz_model
    from tdqc.numerics.deep_q_learning.parameters_lri import parameters, parameters_replay_memory
    '''
    settings = dict()
    settings["steps"] = 3
    L = 2
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
    settings["n_episodes"] = int(5e4)
    settings["epsilon_max"] = 1.0
    settings["epsilon_min"] = 0.005
    settings["epsilon_decay"] = 0.9999411315398542 
    settings["model_update_spacing"] = 20
    settings["system_class"] = 'LongRangeIsing'
    settings["network_type"] = 'SingleDense'
    settings["env_type"] = 'DynamicalEvolution_cpp'
    settings["exploration"] = 'gaussian'
    settings["step"] = 0.001
    settings["capacity"] = 50
    settings["sampling_size"] = 50
    settings["NN_optimizer"] = 'adam'
    settings["n_epochs"] = 1
    '''
    pass

@pytest.mark.slow
def test_deep_q_learning_solver_solve():
    from tdqc.solver.deep_q_learning import DeepQLearning, DQLWithReplayMemory
    from tdqc.numerics.ed.models_ed import State
    from tdqc.numerics.ed.models_ed import xxz_model
    from tdqc.numerics.deep_q_learning.parameters_lri import parameters, parameters_replay_memory
    solver = DQLWithReplayMemory()
    solver.load_settings(settings=parameters)
    solver.load_seetings_replay_memory(**parameters_replay_memory)
    assert callable(getattr(solver, 'solve', None)), "DQLWithReplayMemory has a method solve"
    #rho_target = solver.get_rho_target_from_other_solver()
    #assert isinstance(rho_target,np.ndarray), "DQLWithReplayMemory can get the target_state"


    # It must be possible to get the list of amplitudes obtained from solved.
    #assert hasattr(solver, 'time_evolution'), "EDSolver has an attribut time_evolution"
    solver.solve()
    #assert isinstance(getattr(solver, 'time_evolution', None), np.ndarray), "EDSolver method 'solve' returns an array"

#test_deep_q_learning_solver_solve()
