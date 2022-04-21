import pytest
import tdqc

@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.fast
def test_ed_solver_load_settings():
	from tdqc.solver.ed import EDSolver
	from tdqc.numerics.ed.models_ed import Model
	from tdqc.numerics.ed.models_ed import State
	
	settings = dict()
	settings["steps"] = 3
	settings["interval"] = 0.1
	settings["model"] = ???
	settings["state"] = ???

	solver = EDSolver()
	solver.load_settings(settings)

@pytest.mark.fast
def test_ed_solver_structure():
    from tdqc.solver.ed import EDSolver
    assert callable(getattr(EDSolver, 'solve', None)), "EDSolver has a method solve"
    # It must be possible to get the liste of amplitudes obtained from solved.
    assert callable(getattr(State, 'get_time_evolution', None)), "EDSolver  has a method get_time_evolution"


@pytest.mark.slow
def test_ed_solver_solve():
	pass
