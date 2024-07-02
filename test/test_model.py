#%%
import pytest
import tdqc
import matplotlib
import matplotlib.pyplot as plt
import cmath
import math
import numpy as np
import copy 
from tdqc.numerics.ed.models_ed import *
from tdqc.numerics.ed.exact_diagonalisation import *
from tdqc.solver.ed import EDSolver

@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.fast
def test_model_structure():
    from tdqc.numerics.ed.models_ed import Model
    # A model must have a Hamiltonian (required for EDSolver.py)
    # A model must have methods to get hamiltonian, eig_values,eig_vectors,ground_state (required for EDSolver.py)

@pytest.mark.fast
def test_lri_model():
    # Models
    model = lri_model
    L = 6
    J = 1
    m_x = 2
    m_z = 2
    alpha = 3
    model.parametrize_hamiltonian(*[L,J,alpha,m_x,m_z])
    gs_per_site_list = []
    site_list = [l for l in range(1,L,1)]
    # The initial state is a uniform superposition.
    initial_state = np.ones([2**L], dtype='complex128')/2**(L-1)
    psi_t_n = State(initial_state)
    H = model.hamiltonian
    eig_values,eig_vectors = np.linalg.eig(H)
    t_initial = 0
    t_final = 1
    step = 0.1
    exact_diagonalization = ExactDiagonalization(model,L,initial_state,t_final,t_initial,step)
    ground_state = exact_diagonalization.get_ground_state()
    print("gs:{}".format(ground_state))
    print("psi_t_0:{}".format(psi_t_n._density_mat))
    psi_t_n.time_step_ed( model, delta_t = step, imaginary=False)
    print("psi_t_1:{}".format(psi_t_n._density_mat))
    t_list = [t for t in np.arange(t_initial,t_final,step)]
    pass

@pytest.mark.fast
def test_ed_solver_structure():
    from tdqc.solver.ed import EDSolver
    from tdqc.numerics.ed.models_ed import State
    from tdqc.numerics.ed.models_ed import xxz_model
    settings = dict()
    L = 4
    J = 1
    m_x = 2
    m_z = 2
    alpha = 3
    model = lri_model
    model.parametrize_hamiltonian(*[L,J,alpha,m_x,m_z])
    
    settings["model"] = model
    init_vec_state = np.zeros([2**L],dtype='complex128')
    init_vec_state[0] = 1
    settings["state"] = State(init_vec_state)
    settings["t_initial"] = 0.0
    settings["t_final"] = 1.0
    settings["n_steps"] = int(1/0.001)

    solver = EDSolver()
    solver.load_settings(settings)
    assert callable(getattr(solver, 'solve', None)), "EDSolver has a method solve"
    # It must be possible to get the list of amplitudes obtained from solved.
    assert hasattr(solver, 'time_evolution'), "EDSolver has an attribut time_evolution"
    solver.solve()
    assert isinstance(getattr(solver, 'time_evolution', None), np.ndarray), "EDSolver method 'solve' returns an array"
    rho_target = solver.get_rho_target()
    assert isinstance(rho_target,np.ndarray), "EDSolver method 'get_target_state' returns an array"

@pytest.mark.slow
def test_lri_model_solve():
    from tdqc.solver.ed import EDSolver
    from tdqc.numerics.ed.models_ed import State
    from tdqc.numerics.ed.models_ed import xxz_model
    settings = dict()
    L = 4
    J = 2
    m_x = 1
    m_z = 1
    alpha = 3
    model = lri_model
    model.parametrize_hamiltonian(*[L,J,alpha,m_x,m_z])
    settings["model"] = model
    H = model.hamiltonian
    # The initial state is a uniform superposition.
    #initial_state = np.ones([2**L], dtype='complex128')
    #initial_state = initial_state/np.linalg.norm(initial_state)
    
    initial_state= np.zeros([2**L],dtype='complex128')
    initial_state[0] = 1.+1j*0
    settings["state"] = State(initial_state)
    settings["t_initial"] = 0.0
    settings["t_final"] = 1.*4.*math.pi
    settings["n_steps"] = int(1/0.001)
    solver = EDSolver()
    solver.load_settings(settings)
    #print("solver.__model.hamiltonian:{}".format(solver._EDSolver__model.hamiltonian))
    solver.solve()
    #print('final_state:{}'.format(solver.final_state._vec_state))
    site_list = [l for l in range(1,L)]
    amplitudes = solver.time_evolution
    step = (settings["t_final"]-settings["t_initial"])/settings["n_steps"]
    t_list = [t for t in np.arange(settings["t_initial"],settings["t_final"],step)]
    # Plot the time evolution of the state in the computational basis.
    legend_list = []
    for i in range(0,2**L):
        plt.plot(t_list,np.square(np.abs(amplitudes[:,i])),'-')
        legend_list.append(str("|")+bin(i)[2:]+str(">"))#+str(i)+str(">"))
    plt.xlabel('time t')
    plt.ylabel('Probability')
    #plt.title('Probabilities of the states in the basis {|0>,|1>,...,|2**L-1>}')
    plt.legend(legend_list)
    plt.show()
    plt.savefig('test.png')
    #gs = ground_states(eig_values,eig_vectors)

@pytest.mark.fast
def test_fermionic_system():
    site=0
    L=1
    f_dagger_fct = f_dagger(site, L)
    f_dagger_theory = np.array([[0,1], [0,0]])
    assertion = (f_dagger_fct==f_dagger_theory)
    assert assertion.all()

    f_fct = f(site, L)
    f_theory = np.array([[0,0], [-1,0]])
    assertion = (f_fct==f_theory)
    assert assertion.all()

def test_tb_model():
    tb_2quanti = copy.deepcopy(tb_second_quantization)
    g = 1
    L = 2
    tb_2quanti.parametrize_hamiltonian(*[L,g])
    gs_per_site_list = []
    site_list = [l for l in range(1, L, 1)]
    # The initial state is a uniform superposition.
    initial_state = np.ones([2**L], dtype='complex128')/2**(L-1)
    psi_t_n = State(initial_state)
    H = tb_2quanti.hamiltonian
    eig_values, eig_vectors = np.linalg.eigh(H)
    #print("eig_values:{}".format(eig_values))
    #print("eig_vectors:{}".format(eig_vectors))
    for i, eif_v in enumerate(eig_vectors):
        print("eigen_vector:{} with eigenval:{} ".format(eig_vectors[:, i],eig_values[i]))

    t_initial = 0
    t_final = 1
    step = 0.1
    exact_diagonalization = ExactDiagonalization(tb_2quanti, L, initial_state,t_final,t_initial,step)
    ground_state = exact_diagonalization.get_ground_state()
    #print("gs:{}".format(ground_state))
    #print("psi_t_0:{}".format(psi_t_n._density_mat))
    psi_t_n.time_step_ed( tb_2quanti, delta_t = step, imaginary=False)
    #print("psi_t_1:{}".format(psi_t_n._density_mat))
    t_list = [t for t in np.arange(t_initial,t_final,step)]
    return eig_values
    # pass

def test_anderson_model():
    anderson_instance = copy.deepcopy(anderson_impurity_model)
    L = 4 # L=2 <=> 2 impurity sites; L=4 <=> 2 impurity sites and 2 spin sites
    E_k = np.array([0])
    V_k = np.array([1])
    E = 0
    U = 0
    anderson_instance.parametrize_hamiltonian(*[L, E_k, V_k, E, U])
    gs_per_site_list = []
    site_list = [l for l in range(1, L, 1)]
    # The initial state is a uniform superposition.
    initial_state = np.ones([2**L], dtype='complex128')/2**(L-1)
    psi_t_n = State(initial_state)
    H = anderson_instance.hamiltonian
    print("H:{}".format(H))
    eig_values, eig_vectors = np.linalg.eigh(H)
    #print("eig_values:{}".format(eig_values))
    #print("eig_vectors:{}".format(eig_vectors))
    for i, eif_v in enumerate(eig_vectors):
        print("eigen_vector:{} with eigenval:{} ".format(eig_vectors[:, i],eig_values[i]))

    t_initial = 0
    t_final = 1
    step = 0.1
    exact_diagonalization = ExactDiagonalization(anderson_instance, L, initial_state,t_final,t_initial,step)
    ground_state = exact_diagonalization.get_ground_state()
    #print("gs:{}".format(ground_state))
    #print("psi_t_0:{}".format(psi_t_n._density_mat))
    psi_t_n.time_step_ed( anderson_instance, delta_t = step, imaginary=False)
    #print("psi_t_1:{}".format(psi_t_n._density_mat))
    t_list = [t for t in np.arange(t_initial,t_final,step)]
    return eig_values
 

def number_of_particle_anderson():
    anderson_instance = copy.deepcopy(anderson_impurity_model_tridiagonal)
    L = 8 # L=2 <=> 2 impurity sites; L=4 <=> 2 impurity sites and 2 spin sites
    E_k = np.array([-1.353633082708697533,-3.886596945793371893e-02,1.237242218373216351])
    V_k = np.array([9.518621810161435881e-2,1.109370682087390536e-1,9.833858307543467958e-2])
    E = 0
    U = 8
    anderson_instance.parametrize_hamiltonian(*[L, E_k, V_k, E, U])
    H = anderson_instance.hamiltonian
    Q = anderson_instance.hessenberg_unitary
    gs_per_site_list = []
    site_list = [l for l in range(1, L, 1)]
    # Build the particle number operator.
    number_operator = np.zeros((2**(L), 2**(L)), dtype='complex128')
    for site in range(0, L):
        number_operator += np.dot(creator(site, L), annihilator(site, L))
    print("number_operator:{}".format(number_operator))
    number_operator = np.dot(Q.conj().T, np.dot(number_operator, Q))
    eig_values, eig_vectors = np.linalg.eigh(H)
    #print("eig_values:{}".format(eig_values))
    #print("eig_vectors:{}".format(eig_vectors))
    for i, eif_v in enumerate(eig_vectors):
        print("eigen_vector:{} with eigenval:{} ".format(eig_vectors[:, i],eig_values[i]))
    ground_state = anderson_instance.ground_state
    number_of_particle = np.dot(ground_state.conj().T, np.dot(number_operator, ground_state))
    print(number_of_particle)
    return number_of_particle

n = number_of_particle_anderson()
'''
    #print("Eigenvectors:{}".format(eig_vectors))
    print(eig_values)
    print(gs)
    print(abs(amplitudes[-1,:])**2)
    #print("Eigenvalues:{}".format(eig_values))
    # Plot the time evolution of the projection of the state of the system onto the each of the eigenstates 
    # which are not a ground state.
    legend_list = []
    min_index = np.argmin(eig_values)
    min_energy = eig_values[min_index]
    fig = plt.figure()
    #ax = fig.add_subplot(111)
    #ax.plot([1,2,3])

    for index, state in enumerate(eig_vectors.T):
        projection_list = []
        # If it is not the ground state
        if abs(eig_values[index]-min_energy)>10**(-12):
            for t_n,_ in enumerate(t_list):
                projection_list.append(abs(np.dot(state.T.conj(),amplitudes[t_n,:]))**2)
                #print("np.dot(gs.T,amplitudes[t_n,:]:{}".format(np.dot(gs.T,amplitudes[t_n,:])))
            plt.plot(t_list,projection_list,'-',)
            legend_list.append(index)#str(eig_values[index])+", state:"+str(eig_vectors[index]))
            print("Eigenvectors number: {}".format(index))
        else:
            print("Eigenvectors number: {}: is a ground state".format(index))
    plt.xlabel('time t')
    plt.ylabel('|<eigenvector|Psi(t)>|**2')
    plt.title('Projection of the state on the eigenvectors different from the ground states.')
    plt.legend(legend_list)
    #print(legend_list)
    plt.show()
    fig.savefig('test.png')
'''


'''
    # Plot the time evolution of the projection of the state of the system onto the ground state(s).
    projection_list = []
    #print("amplitudes:{}".format(amplitudes))
    gs = ground_states(eig_values,eig_vectors)
    print(gs)
    for t_n,_ in enumerate(t_list):
        projection_list.append(abs(np.dot(gs.T.conj(),amplitudes[t_n,:]))**2)
        #print("np.dot(gs.T,amplitudes[t_n,:]:{}".format(np.dot(gs.T.conj(),amplitudes[t_n,:])))
    plt.plot(t_list,projection_list,'-',)
    plt.xlabel('time t')
    #plt.ylabel('amplitudes')
    plt.ylabel('sum(abs(gs-amplitudes(t)))')
    plt.title('Time evolution of the projection of the state of the system onto the ground state(s)')
    plt.show()
'''

# %%

# %%
