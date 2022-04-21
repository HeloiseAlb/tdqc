import pytest
import tdqc

@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.fast
def test_model_structure():
    from tdqc.numerics.ed.models_ed import Model
    
    # A model must have an Hamilatonian (required for EDSolver.py)
    # A model must have methods to get hamiltonian, eig_values,eig_vectors,ground_state (required for EDSolver.py)
