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



@pytest.mark.slow
def test_trotterization_solver_solve_range():
    from tdqc.solver.trotterization import Trotterization
    from tdqc.numerics.trotterization.parameters_trotterization import parameters
    range_t_final = np.array([0.06, 0.05, 0.04, 0.08, 0.09])
    for t_final_local in range_t_final:
        solver = Trotterization()
        parameters.update({"t_final":t_final_local})
        solver.load_settings(settings=parameters)
        solver.solve()

test_trotterization_solver_solve()
