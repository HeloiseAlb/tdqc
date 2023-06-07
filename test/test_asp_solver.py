#%%
import pytest
import tdqc
import numpy as np
import sys 
import cmath
import math
from tdqc.numerics.ed.exact_diagonalisation import *

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

#%%
@pytest.fixture(scope="session", autouse=True)
def setup_logger(record_testsuite_property):
    import logging
    globals()['LOGGER'] = logging.getLogger(__name__)
    LOGGER.debug((' ' +__name__ + ' ').center(100,'-'))

@pytest.mark.fast
def test_asp_solver_load_settings():
    from tdqc.solver.asp import AdiaStatePrepa
    from tdqc.numerics.asp.parameters_asp import parameters
    solver = AdiaStatePrepa()
    solver.load_settings(parameters)
    
@pytest.mark.fast
def test_asp_solver_structure():
    from tdqc.solver.asp import AdiaStatePrepa
    from tdqc.numerics.asp.parameters_asp import parameters
    solver = AdiaStatePrepa()
    solver.load_settings(parameters)
    
    assert callable(getattr(solver, 'solve', None)), "ASPSolver has a method solve"
    solver.solve()
    # It must be possible to get the list of amplitudes obtained from solve.
    assert hasattr(solver, 'time_evolution'), "ASPSolver has an attribut time_evolution"
    
    assert isinstance(getattr(solver, 'time_evolution', None), np.ndarray), "ASPSolver method 'solve' returns an array"
    rho_target = solver.get_rho_target()
    assert isinstance(rho_target,np.ndarray), "ASPSolver method 'get_target_state' returns an array"
    print(rho_target)
    print(solver.time_evolution)
    

@pytest.mark.fast
def test_fidelity_and_evolution_amplitude():
    from tdqc.solver.asp import AdiaStatePrepa
    from tdqc.numerics.ed.models_ed import State
    from tdqc.numerics.ed.models_ed import trans_ising_model
    from tdqc.numerics.asp.parameters_asp import parameters
    t_list = [t for t in np.linspace(parameters["t_initial"],parameters["t_final"],parameters["n_steps"]+1)]    
    L = parameters["n_sites"]
    solver = AdiaStatePrepa()
    solver.load_settings(parameters)
    solver.solve()
    rho_target = solver.get_rho_target()
    amplitudes = solver.time_evolution
    # Plot the time evolution of the state in the computational basis.
    legend_list = []
    for i in range(2**L):
        plt.plot(t_list,abs(amplitudes[:,i])**2,'-')
        legend_list.append(str("|")+bin(i)[2:]+str(">"))
    plt.xlabel('time t')
    plt.ylabel('Probability')
    plt.title('Probabilities of the states in the basis {|0>,|1>,...,|2**L-1>}')
    plt.legend(legend_list)
    plt.savefig('my_plot.png')
    H_f = parameters["model_f"]
    H_0 = parameters["model_0"]
    ground_state_h_f = H_f.ground_states
    size = ground_state_h_f.shape
    random_state = np.random.randn(*size) + 1j * np.random.randn(*size)
    random_state /= np.linalg.norm(random_state)
    final_state = solver.get_state_target()
    print("Fidelity with the ground state:{}".format(abs(np.vdot(np.conj(ground_state_h_f),final_state))))
    print("Fidelity with a random state:{}".format(abs(np.vdot(np.conj(random_state),final_state))))


#%%


#@pytest.mark.fast
#def test_fidelity_evolution():
from tdqc.solver.asp import AdiaStatePrepa
from tdqc.numerics.ed.models_ed import State
from tdqc.numerics.ed.models_ed import trans_ising_model
from tdqc.numerics.asp.parameters_asp import parameters
t_list = [t for t in np.linspace(parameters["t_initial"],parameters["t_final"],parameters["n_steps"])]    
L = parameters["n_sites"]
solver = AdiaStatePrepa()
solver.load_settings(parameters)
solver.solve()

fidelities = solver.list_fidelities
gaps = solver.list_gaps

#%% 
# Plot the time evolution of the fidelities between the ground state of H_t_n 
# and the state of the system.
print("average fidelity: {}".format(np.average(abs(fidelities[:]))))
plt.plot(t_list,abs(fidelities[:]))
plt.xlabel('time t')
plt.ylabel('Fidelities')
plt.title('Absolute value of the fidelity between state \n of the system at time t and the ground state \n of the Hamiltonian at time t: H(t)')
plt.savefig('my_plot_fidelities.png')

# Is T long enough?
delta_s_H = -parameters['model_0'].hamiltonian + parameters['model_f'].hamiltonian
_, singular_values, _ = np.linalg.svd(delta_s_H)

# Print the largest singular value
largest_singular_value = singular_values[0]
print("Largest singular value: {}".format(largest_singular_value))
delta_s = (parameters["t_final"]-parameters["t_initial"])/parameters["n_steps"]
print("T limit:{}".format(largest_singular_value/delta_s**2))
#test_fidelity_evolution()

# %%

# %%
