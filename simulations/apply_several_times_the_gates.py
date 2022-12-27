#%%
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional 
from scipy.linalg import eigh, logm
from math import log2, sqrt, log, isnan 
from scipy.linalg import expm
import cmath
from tdqc.numerics.ed.models_ed import Model, xxz_model, lri_model
from tdqc.numerics.ed.models_ed import State
from tdqc.solver.ed import EDSolver
#%%
###################### Functions and dictionaries ######################

spin_op= {
    "I": np.array([[1+0j,0+0j],[0+0j,1+0j]],dtype = 'complex128'),
    "sigma_x": np.array([[0+0j,1+0j],[1+0j,0+0j]],dtype = 'complex128'),
    "sigma_y": np.array([[0+0j,-1j],[1j,0+0j]],dtype = 'complex128'),
    "sigma_z": np.array([[1+0j,0+0j],[0+0j,-1+0j]],dtype = 'complex128'),
    "sigma_+": np.array([[0+0j,1+0j],[0+0j,0+0j]],dtype = 'complex128'),
    "sigma_-": np.array([[0+0j,0+0j],[-1+0j,0+0j]],dtype = 'complex128')}

def tensor_prod(*arg):
    """tensor_prod(a1, a2) = np.kron(a1, a2).
    tensor_prod(a1, a2, ..., an) = np.kron(tensor_prod(a1, ..., an-1), an)
    """
    res = arg[0]
    for i in range(1, len(arg)):
        res = np.kron(res, arg[i])
    #  res = arg[-1]
    #  for i in range(1, len(arg)):
    #      res = np.kron(res, arg[len(arg) - i - 1])
    return res

def globalize_op(local_op: np.ndarray, site: int, L: int)-> np.ndarray:
    """" Return the tensor product of the local operator and identity operators such that the local operator applies on site number site.
    L is the total number of sites in the system on which we want to apply the global operator.
    """
    tensor_0 = np.identity(1, dtype='complex128')
    for i in range(0,site,1):
        tensor_0 = np.kron(tensor_0,np.identity(2, dtype='complex128'))
    tensor_0 = np.kron(tensor_0,local_op)
    for i in range(site+1,L,1):
        tensor_0 = np.kron(tensor_0,np.identity(2, dtype='complex128'))
    return tensor_0

def get_initial_state_for_ed(seed: Optional[int], initial_state: str, n_sites:int)-> np.ndarray:
    if initial_state == 'random_product_state':
        np.random.seed(seed)
        #  randomly directed vector on the unit sphere
        #  f(theta)g(phi)dtheta dphi = sin(theta)/4pi
        #  g(phi) = 1/2pi, f(theta) = sin(theta)/2
        #  -> F(theta) = (1-cos(theta))/2
        #  F^-1(u) = arccos(1 - 2u)
        phis = [2*pi*np.random.rand() for _ in range(n_sites)]
        thetas = [np.arccos(1-2*np.random.rand()) for _ in
                    range(n_sites)]
        spinors = [np.array([np.cos(theta/2),
                                np.exp(1j*phi)*np.sin(theta/2)])
                    for phi, theta in zip(phis, thetas)]
        rstate = tensor_prod(*spinors)
        state_real = rstate.real
        state_imag = rstate.imag
    elif initial_state == 'ferro':
        state_real = np.zeros(2**n_sites)
        state_real[0] = 1.0
        state_imag = np.zeros(2**n_sites)
    elif initial_state == 'antiferro':
        state_imag = np.zeros(2**n_sites)
        spinors = [np.array([1.0, 0.0]) if _ % 2 == 0
                    else np.array([0.0, 1.0]) for _ in range(n_sites)]
        state_real = tensor_prod(*spinors)
    else:
        raise NotImplementedError(f'Initial state of type {initial_state} '
                                    'not implemented.')
    initial_state = state_real + 1j*state_imag
    norm = np.linalg.norm(initial_state)
    initial_state = initial_state / norm
    return initial_state

def apply_gate_sequence(initial_state: np.ndarray, jx_angle_list: np.ndarray, hx_angle_list: np.ndarray, hz_angle_list: np.ndarray, n_steps: int, n_sites: int, gate_order: str, coupling_matrix: np.ndarray ):
    """ Apply the sequence of gates onto the initial state and return the final state. """
    # Define the universal quantum gate set used in Markus' article. 
    U_x = lambda theta : expm(-1j*theta*spin_op['sigma_x'])
    U_z = lambda theta : expm(-1j*theta*spin_op['sigma_z'])
    sum_U_xx = coupling_matrix 
    U_xx = lambda theta : expm(-1j*theta*sum_U_xx)
    state = initial_state
    # Those gate lists are in fact lists of angles.
    for step in range(0, n_steps, 1):
        state = np.dot(U_xx(jx_angle_list[step]),state)
        #print('state after U_xx:{} is {}'.format(U_xx(jx_angle_list[step]),state))
        for site in range(0, n_sites, 1):
            if gate_order == "xz":
                U_x_site = globalize_op(U_x(hx_angle_list[step][site]),site,n_sites)
                state = np.dot(U_x_site,state)
                U_z_site = globalize_op(U_z(hz_angle_list[step][site]),site,n_sites)
                state = np.dot(U_z_site,state)
            elif gate_order == "zx":
                U_z_site = globalize_op(U_z(hz_angle_list[step][site]),site,n_sites)
                state = np.dot(U_z_site,state)
                U_x_site = globalize_op(U_x(hx_angle_list[step][site]),site,n_sites)
                state = np.dot(U_x_site,state)
    return state

def get_coupling_matrix(n_sites: int, alpha: float )-> np.ndarray:
        dim = int(2**n_sites)
        list_glob_operators =  [None] * n_sites
        for site in range(0,n_sites,1):
            list_glob_operators[site] = globalize_op(spin_op["sigma_x"], site, n_sites)  
        coupling_matrix = np.zeros((dim,dim),dtype='complex128')
        for l in range(0,n_sites,1):
            matrix_1 = list_glob_operators[l] 
            for k in range(l+1,n_sites,1):
                matrix_2 = list_glob_operators[k]
                coupling_matrix += np.dot(matrix_1,matrix_2)/(k-l)**alpha
        return coupling_matrix

###################### Fidelity functions ######################
    
def local_reward(rho1,rho2,n_qubits=None): 
    if n_qubits == None:
        n_qubits = int(log2(rho1.shape[0]))
    sum_measures = 0
    for j in range(0,n_qubits-1):
        for k in range(j+1,n_qubits):
            sum_measures += cmath.sqrt(relative_entropy(reduced_density_matrix(rho1,j,k),reduced_density_matrix(rho2,j,k)))
            #print("sqrt(relative_entropy({},{}))={}".format(reduced_density_matrix(rho1,j,k),reduced_density_matrix(rho2,j,k),cmath.sqrt(relative_entropy(reduced_density_matrix(rho1,j,k),reduced_density_matrix(rho2,j,k)))))
    #print('sum_measures:{}'.format(sum_measures))
    if sum_measures == float('inf') or isnan(sum_measures.real) or isnan(sum_measures.imag):
        r_local = 0
        print("It means sum_measures == float('inf') ")
    else:
        r_local = 1 - 2/(n_qubits*(n_qubits-1))*sum_measures
    return r_local


def reduced_density_matrix(rho_init: np.ndarray, site1: int, site2: int, n_qubits: int=None)-> np.ndarray:
    """ Return the reduced density matrix of the subsystem made of sites site1 and site2 for rho. So a 4-by-4 matrix. """
    rho = rho_init 
    if n_qubits == None:
        n_qubits = int(log2(rho.shape[0]))
    n = n_qubits
    if site1>site2:
        site1, site2 = site2, site1
    if site1>0:
        n1, n2 =int(2**(site1)), int(2**(n-site1))
        rho = rho.reshape([n1, n2, n1, n2])
        rho = np.trace(rho, axis1=0, axis2=2)
        n -= site1
        site2 -= site1
    if site2>1:
        n1, n2 = int(2**(site2-1)), int(2**(n-site2))
        rho = rho.reshape([2,n1,n2,2,n1,n2])
        rho = np.trace(rho,axis1=1,axis2=4)
        n -= site2-1
    if n>2:
        n2 = int(2**(n-2))
        rho = rho.reshape([4, n2, 4, n2])
        rho = np.trace(rho, axis1=1, axis2=3)
    rho = rho.reshape([4, 4])
    return rho

def relative_entropy(rho1: np.ndarray, rho2: np.ndarray, positiveDefinite:bool=True)-> float:
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
                        subsum_index1 += abs( np.dot(eVecs2[:, index2],eVecs1[:, index1]))**2 * log(value2)
                relativeEntropy -= value1 * subsum_index1
        return np.real(relativeEntropy)
    else:
        return np.trace(np.dot(rho1,(logm(rho1)-logm(rho2))))

# The following function need to be optimized it is not at all.
def apply_several_times_the_gates(initial_state: np.ndarray, jx_angle_list: np.ndarray, hx_angle_list: np.ndarray, hz_angle_list: np.ndarray, n_times_apply_gates: int, n_steps: int, n_sites: int, gate_order: str)-> np.ndarray:
    state = initial_state_vector
    for i in range(0, n_times_apply_gates, 1):
        state = apply_gate_sequence(state, jx_angle_list, hx_angle_list, hz_angle_list, n_steps, n_sites, gate_order, coupling_matrix)
    return state
#%%
###################### Parameters necessary ######################

# Initializing model
L = 6 # 10 # Must be the same as n_sites. It is the number of sites in the physical system.
J = 1.0
m_x = 2.0
m_z = 2.0
alpha = int(3)
model = lri_model
model.parametrize_hamiltonian(*[L,J,alpha,m_x,m_z])

initial_state = 'antiferro' # 'ferro','random_product_state',
seed_initial_state = None, # 42, #useful to determined only if 'initial_state'=='random_product_state'
initial_state_vector = get_initial_state_for_ed(seed = seed_initial_state, initial_state = initial_state, n_sites = L )


# Gates to apply
jx_angle_list = np.array([-0.162358301276381,-0.18456291663508823,-0.12726540928208135])
hx_angle_list = np.array([[0.2637649248858866,-0.2919513340767098,0.25363122063889637,-0.31495140182150866,0.19636886086661678,-0.1476383265262698],
        [-0.1275951106866796,-0.09324881925800046,-0.3478107350240427,0.2397464332938758,-0.10758290510123943,-0.011477091452825812],
        [-0.02494443190525083,-0.08999724506795029,0.0950983260078663,-0.0347165449202795,-0.08478163182615452,0.16610157013928178]])
hz_angle_list = np.array([[-0.0002785926319221394,-0.23180188321243123,-0.3646970717946415,-0.3534725761276842,-0.14979920781381867,-0.2667179318036682],
        [0.34237569848427873,0.336902704639016,0.3058013329103093,-0.2621392519686841,-0.016706272085134644,-0.02014099082938045],
        [-0.2260718897884354,-0.23591322474719126,-0.0023830200993602696,0.22345410792527126,0.28409412896804953,0.11404947015361344]])
#%%
alpha =  3.0
coupling_matrix = get_coupling_matrix(L, alpha)
#%%
n_times_apply_gates = 7 # How many times do we want to apply the block of gates?

parameters_ed = {
        'solver': EDSolver(),
        'n_steps': int(1/0.1), # time steps before (int(1/0.001))
        'model': model,
        'state': State(initial_state_vector),
        't_initial': 0.0,
        't_final': 1.0*n_times_apply_gates}
solver = EDSolver()
solver.load_settings(parameters_ed)
solver.solve()
state_target_here = solver.get_state_target()
print('state_target_here:{}'.format(state_target_here))

#%%

state_out_circuit_here = apply_several_times_the_gates(parameters_ed['state'], jx_angle_list, hx_angle_list, hz_angle_list, n_times_apply_gates,3 , L,"zx")
# %%
rho_target_here = np.tensordot(np.conjugate(state_target_here), state_target_here, axes=0)

rho_out_circuit_here = np.tensordot(np.conjugate(state_out_circuit_here), state_out_circuit_here, axes=0)

#%%
norm_rho_target_here = np.linalg.norm(rho_target_here)
norm_rho_out_circuit_here = np.linalg.norm(rho_out_circuit_here)
# print(norm_rho_target_here, norm_rho_out_circuit_here)
reward2  = local_reward(rho_target_here,rho_out_circuit_here)
reward = local_reward(rho_out_circuit_here, rho_target_here)
print("reward:{} and reward2:{}".format(reward, reward2))
#%%
# print("rho_target:{}".format(rho_target))
# print("state_rho:{}".format(state_rho))

        
best_final_state = np.array([-5.11156749e-02-0.01462827j, -1.09540411e-01+0.10468338j,-2.87648111e-02-0.09007341j ,-1.00168369e-01+0.02696696j,  7.22373263e-03-0.02265404j , 2.16416257e-01+0.09808987j, -8.81149725e-02+0.0609535j , -2.28019409e-02-0.03705487j, -5.30736724e-02-0.06656064j ,-1.45496514e-01+0.0539578j,  1.22669447e-02-0.09480844j, -1.41142657e-01-0.05757797j,  2.15934635e-02-0.02900133j,  1.31328957e-01+0.23478142j, -1.18467855e-01-0.01322187j, -3.74844605e-03-0.05765843j, -7.04726096e-02+0.1110757j , -4.81490171e-02+0.03195318j, -9.14318797e-04-0.02537124j , 5.51117917e-03+0.23415355j, -1.01571898e-02-0.03699885j,  3.31148183e-01-0.42944851j,  9.73549857e-02+0.23284842j , 9.94500550e-05-0.03457667j,  1.95796003e-02-0.01403006j , 5.34808851e-02+0.20540893j, -1.20006501e-01-0.04090327j,  7.79798861e-03-0.02436492j, -3.87931337e-02-0.01781133j ,-2.56212487e-02+0.04423236j, -1.16965471e-02-0.02576747j, -9.82023918e-02-0.01043647j,-5.59405007e-02-0.04985908j, -2.43619423e-02-0.00725434j,  5.83139827e-03-0.01145886j, -1.27506456e-01-0.01625406j,  1.72442762e-02-0.00618361j,  2.19231239e-01+0.19426304j, -1.26719016e-01+0.04185672j , 9.69459444e-03-0.00253864j, -4.31088976e-03+0.00481977j, -1.37546395e-01+0.02246465j,  3.48556928e-02-0.08061316j ,-8.39273772e-03+0.00673466j,  1.14526531e-02-0.02653297j,  1.60574644e-02+0.0009815j,  5.84829604e-04+0.00466978j,  6.24456004e-03-0.06521174j, 2.53783404e-02+0.00723711j, -6.69095097e-02-0.00087687j,  1.89606367e-02-0.03759679j,  4.20148577e-02-0.01618053j,  1.93680844e-03+0.00216819j, -9.09004152e-02-0.02661845j,  3.42368290e-02-0.0295189j,  -2.04642164e-03-0.01637477j,  9.10285841e-03-0.04916554j,  1.42887501e-02-0.05671721j,  2.57615745e-02+0.02203169j, -2.02453156e-02-0.08213027j,  1.10326531e-02+0.01249601j, -5.43123991e-02+0.11767019j, -2.86517631e-02-0.05157906j , 1.78951756e-02+0.01324597j])
norm_best_final_state = np.linalg.norm(best_final_state)
# if norm_best_final_state != 0:
#     best_final_state = best_final_state / norm_best_final_state
# best_final_rho = np.tensordot(np.conjugate(best_final_state), best_final_state, axes=0)
# print(best_final_state.shape)

target_state = np.array([-1.84835037e-02+0.0357884j,  -9.38539996e-02-0.10072194j, -3.17142940e-02+0.02168831j ,-7.35545208e-02-0.05063489j,  6.02953783e-03+0.02080841j , 1.13297953e-01-0.02563372j,-5.64698630e-02-0.0870938j , -1.89484229e-02+0.03469516j, -1.37987056e-02+0.03168261j  ,4.82819454e-02-0.06849051j, -1.74248779e-03-0.07199181j ,-1.59307772e-01-0.07434383j, -2.77665658e-03-0.00550342j , 2.03885479e-01-0.08788568j, -1.09863247e-01-0.07607515j , 2.03718133e-02-0.04872855j,-1.25408592e-02-0.07568858j,  1.13283093e-01-0.03198461j,  3.00823499e-02+0.05431869j , 1.96736693e-01-0.04338324j, -4.59415781e-02-0.01757068j ,-3.20089574e-01+0.4353613j,  2.91426688e-01+0.04800088j , 9.29057785e-03+0.05002886j, -1.29205962e-02+0.01646069j , 1.70025401e-01-0.05705094j, -6.44243065e-02-0.13738767j , 1.62314703e-01-0.04113906j, -5.79683103e-02+0.02292555j , 6.44053276e-02+0.00921671j,  2.76553452e-02-0.02524478j , 8.45136243e-02-0.04301539j,-2.63154312e-02+0.03249099j, -7.19979981e-02-0.09705468j, -8.56160094e-03-0.0230571j,  -1.31455756e-01-0.06425887j, -2.02901270e-04+0.03166966j,  3.02625783e-01+0.02012253j, -7.32906541e-02-0.15335121j,  6.97811803e-03-0.04036564j, -1.22440399e-02+0.01202771j, -9.32337604e-02-0.15241523j, -7.69369018e-02+0.0617007j,  -1.08984119e-02-0.12200334j,  4.57288783e-03+0.0545976j,   8.78522343e-02+0.05431292j,  2.02750318e-03-0.09065463j, -1.77129409e-02-0.02535297j,  9.39190851e-03-0.0101203j,  -4.50259793e-02+0.02761871j,  9.30590771e-03+0.05869269j , 1.09736657e-02+0.0016745j,  1.42993985e-02-0.00537461j, -9.94041256e-02+0.03199844j,  2.62426637e-02+0.03062096j, -1.68202172e-02+0.02212902j, -4.39931452e-03+0.01385749j,  4.65745421e-02+0.0013195j,  4.91007885e-03-0.06073608j, -4.80273308e-04+0.0135952j, -1.68881996e-02-0.00303131j , 8.64480767e-02-0.01886735j, -2.86103117e-02-0.02330132j , 3.19839362e-02-0.01021015j])
norm_target_state = np.linalg.norm(target_state)
# if norm_target_state != 0:
#     target_state = target_state / norm_target_state
# target_rho = np.tensordot(np.conjugate(best_final_state), best_final_state, axes=0)

print("|psi><psi|:{}".format(np.tensordot(np.conjugate(target_state), target_state, axes=0)))
#print("target_state:{}".format(target_state))

reward_reference = local_reward(target_rho,best_final_rho)

print("reward_reference:{}".format(reward_reference))

