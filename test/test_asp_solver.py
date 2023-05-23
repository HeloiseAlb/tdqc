#%%
import pytest
import tdqc
import numpy as np
import sys 
import cmath
import math
from tdqc.numerics.ed.exact_diagonalisation import *

import matplotlib
matplotlib.use('Agg')
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

#%%
@pytest.mark.fast
def test_ed_solver_structure():
    from tdqc.solver.asp import AdiaStatePrepa
    from tdqc.numerics.asp.parameters_asp import parameters
    solver = AdiaStatePrepa()
    solver.load_settings(parameters)

    assert callable(getattr(solver, 'solve', None)), "EDSolver has a method solve"
    # It must be possible to get the list of amplitudes obtained from solve.
    assert hasattr(solver, 'time_evolution'), "EDSolver has an attribut time_evolution"
    solver.solve()
    assert isinstance(getattr(solver, 'time_evolution', None), np.ndarray), "EDSolver method 'solve' returns an array"
    rho_target = solver.get_rho_target()
    assert isinstance(rho_target,np.ndarray), "EDSolver method 'get_target_state' returns an array"
test_ed_solver_structure()

#%%
@pytest.mark.slow
def test_ed_solver_solve():
    from tdqc.solver.ed import EDSolver
    from tdqc.numerics.ed.models_ed import State
    from tdqc.numerics.ed.models_ed import xxz_model
    settings = dict()
     
    L = 10
    Jzz = 1.0
    Jxy = 1.0
    model = xxz_model
    model.parametrize_hamiltonian(*[L,Jxy,Jzz])
    settings["model"] = model
    init_vec_state = np.zeros([2**L],dtype='complex128')
    init_vec_state[0] = 1
    settings["state"] = State(init_vec_state)
    settings["t_initial"] = 0.0
    settings["t_final"] = 1.0
    settings["n_steps"] = int(1/0.1)
    
    solver = EDSolver()
    solver.load_settings(settings)
    # assert callable(getattr(solver, 'solve', None)), "EDSolver has a method solve"
    # It must be possible to get the list of amplitudes obtained from solved.
    # assert hasattr(solver, 'time_evolution'), "EDSolver has an attribut time_evolution"
    solver.solve()
    assert isinstance(getattr(solver, 'time_evolution', None), np.ndarray), "EDSolver method 'solve' returns an array"
    rho_target = solver.get_rho_target()
    assert isinstance(rho_target,np.ndarray), "EDSolver method 'get_target_state' returns an array"
    time_evo = solver.time_evolution
    print('time_evolution:{}:'.format(time_evo))
    print('rho_target:{}'.format(rho_target))

@pytest.mark.fast
def test_ed_solver_solve_trans_ising():
    from tdqc.solver.ed import EDSolver
    from tdqc.numerics.ed.models_ed import State
    from tdqc.numerics.ed.models_ed import trans_ising_model
    settings = dict()
    L, J, alpha, h = 10,1,2,1
    model = trans_ising_model
    model.parametrize_hamiltonian(*[L, J, alpha, h])
    settings["model"] = model
    init_vec_state = np.zeros([2**L],dtype='complex128')
    init_vec_state[0] = 1
    settings["state"] = State(init_vec_state)
    settings["t_initial"] = 0.0
    settings["t_final"] = 1.0
    settings["n_steps"] = int(1/0.1)
    
    solver = EDSolver()
    solver.load_settings(settings)
    # assert callable(getattr(solver, 'solve', None)), "EDSolver has a method solve"
    # It must be possible to get the list of amplitudes obtained from solved.
    # assert hasattr(solver, 'time_evolution'), "EDSolver has an attribut time_evolution"
    solver.solve()
    assert isinstance(getattr(solver, 'time_evolution', None), np.ndarray), "EDSolver method 'solve' returns an array"
    rho_target = solver.get_rho_target()
    assert isinstance(rho_target,np.ndarray), "EDSolver method 'get_target_state' returns an array"
    time_evo = solver.time_evolution
    print('time_evolution:{}:'.format(time_evo))
    print('rho_target:{}'.format(rho_target))

test_ed_solver_solve_trans_ising()
#%%
from tdqc.solver.ed import EDSolver
from tdqc.numerics.ed.models_ed import State
from tdqc.numerics.ed.models_ed import trans_ising_model

# CONVENTIONS
# spin down: [1,0]
# spin up: [0,1]
h_bar = 1 # 1.054571817*10**(-34) # in J.s


# Models
settings = dict()
L, J, alpha, h = 4,1,10,1
model = trans_ising_model
model.parametrize_hamiltonian(*[L, J, alpha, h])
settings["model"] = model
init_vec_state = np.zeros([2**L],dtype='complex128')
init_vec_state[0] = 1
settings["state"] = State(init_vec_state)
settings["t_initial"] = 0.0
settings["t_final"] = 10.0
settings["n_steps"] = 100
t_list = [t for t in np.linspace(settings["t_initial"],settings["t_final"],settings["n_steps"])]    

gs_per_site_list = []
solver = EDSolver()
solver.load_settings(settings)
solver.solve()
rho_target = solver.get_rho_target()
amplitudes = solver.time_evolution
# Plot the time evolution of the state in the computational basis.
#legend_list = []
plt.plot()
for i in range(2**L):
    plt.plot(t_list,abs(amplitudes[:,i])**2,'-')
    #legend_list.append(str("|")+bin(i)[2:]+str(">"))
plt.xlabel('time t')
plt.ylabel('Probability')
plt.title('Probabilities of the states in the basis {|0>,|1>,...,|2**L-1>}')
#plt.legend(legend_list)

