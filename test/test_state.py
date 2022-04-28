import pytest
import tdqc
import numpy as np

@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.fast
def test_state_structure():
    from tdqc.numerics.ed.models_ed import State
    # The vec_state must be an array from numpy.
    L = 4
    initial_state = np.array((np.random.normal(size=(2**L,2**L))+1j*np.random.normal(size=(2**L,2**L))),dtype='complex128')
    initial_state = initial_state/np.abs(initial_state)
    #initial_state = np.ones([2**L], dtype='complex128')/2**(L-1)
    state = State(initial_state)
    #assert hasattr(state, vec_state), "State has an attribute vec_state"
    # The vec_state must be an array from numpy.
    amplitudes = state.vec_state
    assert isinstance(amplitudes, np.ndarray), "Amplitudes given as an array from numpy"
    # It must be possible to get the amplitudes in the format of the code from Markus. 
    assert callable(getattr(State, 'get_state_format_ml', None)), "State has a method State has a method get_state_format_ml"
    state_real, state_imag = state.get_state_format_ml()
    assert isinstance(state_real, np.ndarray) and isinstance(state_imag, np.ndarray), "Amplitudes format ml given as two arrays from numpy"
