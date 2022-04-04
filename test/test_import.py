import pytest
import tdqc

@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.fast
def test_import():
	assert True, "import worked"

@pytest.mark.fast
def test_import_ed_solver():
	from tdqc.solver.ed import EDSolver

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

@pytest.mark.slow
def test_ed_solver_solve():
	pass