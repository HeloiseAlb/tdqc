import pytest
import tdqc
import numpy as np
from math import log2, sqrt

@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.fast
def test_reward1():
    from tdqc.numerics.deep_q_learning.environments_cpp import reduced_density_matrix, local_reward
    psi1 = np.array([0,1,0,0],dtype=np.complex128) #|01>
    psi2 = np.array([1,0,0,0],dtype=np.complex128) #|00>
    rho1 = np.tensordot(np.conjugate(psi1), psi1, axes=0)
    rho2 = np.tensordot(np.conjugate(psi2), psi2, axes=0)
    assert local_reward(rho1,rho2)==0.0, "The local reward between matrices such as supp ( ρ ) ∩ ker ( σ ) ≠ 0 is zero."

@pytest.mark.fast
def test_reward2():
    from tdqc.numerics.deep_q_learning.environments_cpp import local_reward, reduced_density_matrix, globalize_op, relative_entropy
    # It is a case where the quantum relative entropy should be 0 according to 11.3.1 of Nielsen and Chuang.
    psi1 = np.array([0,1,0,0],dtype='complex128') #|01>
    psi2 = np.array([1,0,0,0],dtype='complex128') #|00>
    rho1 = np.tensordot(np.conjugate(psi1), psi1, axes=0)
    rho2 = np.tensordot(np.conjugate(psi2), psi2, axes=0)
    rho_rand = np.random.rand(4,4)
    norm = np.linalg.norm(rho_rand)
    rho_rand = rho_rand/norm
    n_qubits = int(log2(rho1.shape[0]))
    assert local_reward(rho_rand,rho_rand,n_qubits=None)==1.0, "The local reward between 2 identical matrix is 1."

@pytest.mark.fast
def test_reward3():
    from tdqc.numerics.deep_q_learning.environments_cpp import reduced_density_matrix, local_reward
    np.random.seed(seed=None)
    rho1 = np.matrix(np.random.rand(4,4))    
    rho1 = np.dot(rho1,rho1.getH())
    rho1 = rho1/np.linalg.norm(rho1)
    rho2 = np.matrix(np.random.rand(4,4))
    rho2 = np.dot(rho2,rho2.getH())
    rho2 = rho2/np.linalg.norm(rho2)
    print("rho1:{}".format(rho1))
    print("rho1:{}".format(rho2))
    print("local_reward of rho1 and rho2:{}".format(local_reward(rho1,rho2)))
    #assert local_reward(rho1,rho2)==0.0, "The local reward between matrices such as supp ( ρ ) ∩ ker ( σ ) = 0 is computed correctly."

@pytest.mark.fast
def test_relative_entropy():
    from scipy.linalg import logm, inv
    from tdqc.numerics.deep_q_learning.environments_cpp import reduced_density_matrix, local_reward, relative_entropy
    
    #np.random.seed(seed=1)
    rho1 = np.matrix(np.random.rand(4,4))
    rho1 = np.dot(rho1,rho1.getH()) # To guarantee that rho1 is a Hermitian positive semi-definite matrix.
    rho1 = rho1/np.linalg.norm(rho1)
 
    #np.random.seed(seed=2)
    rho2 = np.matrix(np.random.rand(4,4))
    rho2 = np.dot(rho2,rho2.getH()) # To guarantee that rho2 is a Hermitian positive semi-definite matrix. 
    rho2 = rho2/np.linalg.norm(rho2)
    
    #rho1 = np.matrix([[1,0],[0,1]])
    #rho2 = np.matrix([[2,0],[0,2]])
    #rho2 = rho2/np.linalg.norm(rho2)
    #rho2 = np.matrix([[1/sqrt(2),-1/sqrt(2)],[1/sqrt(2),1/sqrt(2)]])
    #rho2 = np.matrix([[1/2,-1j*sqrt(3)/2],[1j*sqrt(3)/2,1/2]])
    relative_entropy_instable = np.trace(np.dot(rho1,logm(np.dot(rho1, np.linalg.inv(rho2)))))
    relative_entropy_instable2 = np.trace(np.dot(rho1,(logm(rho1)-logm(rho2))))
    relative_entropy_instable3 = np.trace(np.dot(rho1,logm(rho1))-np.trace(np.dot(rho1,logm(rho2))))
    print("relative_entropy_instable:{}".format(relative_entropy_instable))
    print("relative_entropy_instable2:{}".format(relative_entropy_instable2))
    print("relative_entropy_instable3:{}".format(relative_entropy_instable3))
    relative_entropy_stable = relative_entropy(rho1,rho2)
    print("relative_entropy_stable:{}".format(relative_entropy_stable))
    assert abs(relative_entropy_stable-relative_entropy_instable2)<= 0.01, "The relative_entropy is computed correctly."


#test_reward1()    
#test_reward2()
test_reward3()
test_relative_entropy()
"""
    relative_entropy(rho1,rho2)
    local_reward(rho1,rho2,n_qubits=None)
    #educed_density_matrix(rho_init,site1,site2,n_qubits=None)
    relative_entropy(rho1,rho2)
    rho1 = 
    #assert hasattr(state, vec_state), "State has an attribute vec_state"
    # The vec_state must be an array from numpy.
    amplitudes = state.vec_state
    assert isinstance(amplitudes, np.ndarray), "Amplitudes given as an array from numpy"
    # It must be possible to get the amplitudes in the format of the code from Markus. 
    assert callable(getattr(State, 'get_state_format_ml', None)), "State has a method State has a method get_state_format_ml"
    state_real, state_imag = state.get_state_format_ml()
    assert isinstance(state_real, np.ndarray) and isinstance(state_imag, np.ndarray), "Amplitudes format ml given as two arrays from numpy"
"""
