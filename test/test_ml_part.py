import pytest
import tdqc
import numpy as np
from math import log2

@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.fast
def test_ml1():
    from tdqc.numerics.deep_q_learning.environments_cpp import reduced_density_matrix, local_reward
    
    #assert local_reward(rho1,rho2)==0.0, "blabla"


