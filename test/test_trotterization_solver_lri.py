import pytest
import tdqc
import numpy as np

@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.fast
def test_trotterization_solver_load_settings():
    from tdqc.solver.trotterization import Trotterization
    from tdqc.numerics.trotterization.parameters_trotterization import parameters
    solver = Trotterization()
    solver.load_settings(parameters)


@pytest.mark.slow
def test_trotterization_solver_solve():
    from tdqc.solver.trotterization import Trotterization
    #from tdqc.numerics.ed.models_ed import State
    #from tdqc.numerics.ed.models_ed import xxz_model
    from tdqc.numerics.trotterization.parameters_trotterization import parameters
    solver = Trotterization()
    solver.load_settings(settings=parameters)
    #assert callable(getattr(solver, 'solve', None)), "DQLWithReplayMemory has a method solve"
    #rho_target = solver.get_rho_target_from_other_solver()
    #assert isinstance(rho_target,np.ndarray), "DQLWithReplayMemory can get the target_state"
    # It must be possible to get the list of amplitudes obtained from solved.
    #assert hasattr(solver, 'time_evolution'), "EDSolver has an attribut time_evolution"
    solver.solve()
    #assert isinstance(getattr(solver, 'time_evolution', None), np.ndarray), "EDSolver method 'solve' returns an array"

test_trotterization_solver_solve()
