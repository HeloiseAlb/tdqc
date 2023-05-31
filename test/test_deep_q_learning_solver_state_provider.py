import pytest
import tdqc
import numpy as np

@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.slow
def test_dql_state_provider_solve():
    from tdqc.solver.deep_q_learning import DeepQLearning, DQLWithReplayMemory
    from tdqc.numerics.ed.models_ed import State
    from tdqc.numerics.ed.models_ed import xxz_model
    from tdqc.numerics.state_provider.parameters_state_provider import parameters, parameters_replay_memory
    solver = DQLWithReplayMemory()
    solver.load_settings(settings=parameters)
    solver.load_seetings_replay_memory(**parameters_replay_memory)
    #assert callable(getattr(solver, 'solve', None)), "DQLWithReplayMemory has a method solve"
    #rho_target = solver.get_rho_target_from_other_solver()
    #assert isinstance(rho_target,np.ndarray), "DQLWithReplayMemory can get the target_state"
    # It must be possible to get the list of amplitudes obtained from solved.
    #assert hasattr(solver, 'time_evolution'), "EDSolver has an attribut time_evolution"
    solver.solve()
    #assert isinstance(getattr(solver, 'time_evolution', None), np.ndarray), "EDSolver method 'solve' returns an array"

@pytest.mark.slow
def test_dql_state_provider_solve_circuit():
    from tdqc.solver.deep_q_learning import DeepQLearning, DQLWithReplayMemory
    from tdqc.numerics.ed.models_ed import State
    from tdqc.numerics.ed.models_ed import xxz_model
    from tdqc.numerics.state_provider.parameters_state_provider_mode_circuit import parameters, parameters_replay_memory
    solver = DQLWithReplayMemory()
    solver.load_settings(settings=parameters)
    solver.load_seetings_replay_memory(**parameters_replay_memory)
    #assert callable(getattr(solver, 'solve', None)), "DQLWithReplayMemory has a method solve"
    #rho_target = solver.get_rho_target_from_other_solver()
    #assert isinstance(rho_target,np.ndarray), "DQLWithReplayMemory can get the target_state"
    # It must be possible to get the list of amplitudes obtained from solved.
    #assert hasattr(solver, 'time_evolution'), "EDSolver has an attribut time_evolution"
    solver.solve()
    #assert isinstance(getattr(solver, 'time_evolution', None), np.ndarray), "EDSolver method 'solve' returns an array"

@pytest.mark.slow
def test_deep_q_learning_state_provider_solve_trans_ising():
    from tdqc.solver.deep_q_learning import DeepQLearning, DQLWithReplayMemory
    from tdqc.numerics.ed.models_ed import State
    from tdqc.numerics.ed.models_ed import trans_ising_model
    from tdqc.numerics.asp.parameters_asp import parameters, parameters_replay_memory
    solver = DQLWithReplayMemory()
    solver.load_settings(settings=parameters)
    solver.load_seetings_replay_memory(**parameters_replay_memory)
    solver.solve()

test_deep_q_learning_state_provider_solve_trans_ising()
