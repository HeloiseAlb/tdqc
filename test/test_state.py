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
    assert callable(getattr(State, 'get_amplitudes', None)), "State has a method get_amplitudes"
    # The amplitudes must be an array from numpy.
    L = 4
    initial_state = np.array(np.random.normal(size=(2**L,2**L))+1j*np.random.normal(size=(2**L,2**L)),dtype='complex128')
    initial_state = initial_state/np.abs(initial_state)
    #initial_state = np.ones([2**L], dtype='complex128')/2**(L-1)
    state = State(initial_state)
    amplitudes = state.get_amplitudes()
    assert isinstance(amplitudes, np.ndarray), "Amplitudes given as an array from numpy"
    # It must be possible to get the amplitudes in the format of the code from Markus. 
    assert callable(getattr(State, 'get_state_format_ml', None)), "State has a method State has a method get_state_format_ml"
    state_real, state_imag = state.get_state_format_ml()
    assert isinstance(state_real, np.ndarray) and isinstance(state_imag, np.ndarray), "Amplitudes format ml given as two arrays from numpy"
