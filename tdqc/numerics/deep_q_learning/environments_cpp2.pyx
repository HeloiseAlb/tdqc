cimport numpy as np
import numpy as np
from math import log2, isnan, log #sqrt, 
import cmath
from scipy.linalg import logm, expm, eigh

spin_op= {
    "I": np.array([[1+0j,0+0j],[0+0j,1+0j]],dtype = 'complex128'),
    "sigma_x": np.array([[0+0j,1+0j],[1+0j,0+0j]],dtype = 'complex128'),
    "sigma_y": np.array([[0+0j,-1j],[1j,0+0j]],dtype = 'complex128'),
    "sigma_z": np.array([[1+0j,0+0j],[0+0j,-1+0j]],dtype = 'complex128'),
    "sigma_+": np.array([[0+0j,1+0j],[0+0j,0+0j]],dtype = 'complex128'),
    "sigma_-": np.array([[0+0j,0+0j],[-1+0j,0+0j]],dtype = 'complex128')}


cpdef np.ndarray U_x(float theta):
    return expm(-1j*theta*spin_op['sigma_x'])
cpdef np.ndarray U_z(float theta):
    return expm(-1j*theta*spin_op['sigma_z'])
cpdef np.ndarray U_xx(float theta, np.ndarray coupling_matrix):
    return expm(-1j*theta*coupling_matrix)

cpdef np.ndarray globalize_op(np.ndarray local_op, int site, int L):
    """" Return the tensor product of the local operator and identity operators such that the local operator applies on site number site.
    L is the total number of sites in the system on which we want to apply the global operator.
    """
    cdef int i
    cdef np.ndarray tensor_0

    tensor_0 = np.identity(1, dtype='complex128')    
    for i in range(0,site,1):
        tensor_0 = np.kron(tensor_0,np.identity(2, dtype='complex128'))
    tensor_0 = np.kron(tensor_0,local_op)
    for i in range(site+1,L,1):
        tensor_0 = np.kron(tensor_0,np.identity(2, dtype='complex128'))
    return tensor_0

cpdef np.ndarray apply_gate_sequence_cpp(np.ndarray state, list system_jx_gate_list, list system_hx_gate_list, \
    list system_hz_gate_list, np.ndarray coupling_matrix, int n_steps, int n_sites, char system_gate_order): 
    ''' Apply the sequence of gates onto the initial state and return the final state.
    The gate lists (system_jx_gate_list, system_hx_gate_list, system_hz_gate_list) are in fact lists of angles. 
    '''    
    cdef int step, site 
    cdef np.ndarray U_x_site, U_z_site

    for step in range(0,n_steps,1):
        state = np.dot(U_xx(system_jx_gate_list[step], coupling_matrix), state)
        for site in range(0, n_sites, 1):
            if system_gate_order == "xz":
                U_x_site = globalize_op(U_x(system_hx_gate_list[step][site]), site, n_sites)
                state = np.dot(U_x_site, state)
                U_z_site = globalize_op(U_z(system_hz_gate_list[step][site]), site, n_sites)
                state = np.dot(U_z_site, state)
            elif system_gate_order == "zx":
                U_z_site = globalize_op(U_z(system_hz_gate_list[step][site]), site, n_sites)
                state = np.dot(U_z_site, state)
                U_x_site = globalize_op(U_x(system_hx_gate_list[step][site]), site, n_sites)
                state = np.dot(U_x_site, state)
    return state


cpdef np.ndarray reduced_density_matrix_cpp(np.ndarray rho_init, int site1, int site2, int n_qubits):
    """ Return the reduced density matrix of the subsystem made of sites site1 and site2 for rho. So a 4-by-4 matrix. """
    cdef np.ndarray rho 
    #cdef int n1, n2
    
    rho = rho_init 
    if n_qubits == None:
        n_qubits = int(log2(rho.shape[0]))

    if site1>site2:
        site1, site2 = site2, site1
    if site1>0:
        n1, n2 =int(2**(site1)), int(2**(n_qubits-site1))
        rho = rho.reshape([n1, n2, n1, n2])
        rho = np.trace(rho, axis1=0, axis2=2)
        n_qubits -= site1
        site2 -= site1
    if site2>1:
        n1, n2 = int(2**(site2-1)), int(2**(n_qubits-site2))
        rho = rho.reshape([2,n1,n2,2,n1,n2])
        rho = np.trace(rho, axis1=1, axis2=4)
        n_qubits -= site2-1
    if n_qubits>2:
        n2 = int(2**(n_qubits-2))
        rho = rho.reshape([4, n2, 4, n2])
        rho = np.trace(rho, axis1=1, axis2=3)
    rho = rho.reshape([4, 4])
    return rho

cpdef float relative_entropy_cpp(np.ndarray rho1, np.ndarray rho2, bint positiveDefinite):
    cdef np.ndarray eVals1, eVals2, eVecs1, eVecs2
    cdef float value1, value2, subsum_index1 
    cdef complex relativeEntropy # I changed "double complex" to "complex"
    cdef int index1, index2
    
    if positiveDefinite:
        # Diagonalization the matrix to compute the quantum relative entropy. The matrices must be hermitian positive semidefinite.
        eVals1, eVecs1 = eigh(rho1) 
        eVals1 = np.maximum(eVals1, 0)
        eVals2, eVecs2 = eigh(rho2) 
        eVals2 = np.maximum(eVals2, 0)
        relativeEntropy = 0
        for index1, value1 in enumerate(eVals1):
            subsum_index1 = 0
            if value1 > 0:
                relativeEntropy += value1 * (log(value1))
                for index2, value2 in enumerate(eVals2):
                    if value2 > 0 :
                        subsum_index1 += abs( np.dot(np.conj(eVecs2[:, index2]), eVecs1[:, index1]))**2 * log(value2)
                relativeEntropy -= value1 * subsum_index1
        return np.real(relativeEntropy)
    else:
        return np.trace(np.dot(rho1,(logm(rho1)-logm(rho2))))


cpdef float local_reward_cpp(np.ndarray rho1, np.ndarray rho2, int n_qubits, bint positiveDefinite): 
    cdef int j, k 
    cdef complex r_local, sum_measures # I changed "double complex" to "complex"
    if n_qubits == None:
        n_qubits = int(log2(rho1.shape[0]))
    sum_measures = 0
    for j in range(0, n_qubits-1):
        for k in range(j+1, n_qubits):
            sum_measures += cmath.sqrt(relative_entropy_cpp(reduced_density_matrix_cpp(rho1, j, k, n_qubits), reduced_density_matrix_cpp(rho2, j, k, n_qubits), positiveDefinite))
    
    if sum_measures == float('inf') or isnan(sum_measures.real) or isnan(sum_measures.imag):
        r_local = 0.0 + 1j*0.0
        print("sum_measures was Nan, r_local taken to be 0")
    else:
        r_local = 1 - 2/(n_qubits*(n_qubits-1)) * sum_measures  
    return max(0, r_local.real)