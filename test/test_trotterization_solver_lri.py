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
    from tdqc.numerics.trotterization.parameters_trotterization import parameters
    solver = Trotterization()
    solver.load_settings(settings=parameters)
    solver.solve()

test_trotterization_solver_solve()
