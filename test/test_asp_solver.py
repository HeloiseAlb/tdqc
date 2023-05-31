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
#%%
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

test_asp_solver_structure()

#%%
from tdqc.solver.asp import AdiaStatePrepa
from tdqc.numerics.ed.models_ed import State
from tdqc.numerics.ed.models_ed import trans_ising_model
from tdqc.numerics.asp.parameters_asp import parameters

# CONVENTIONS
# spin down: [1,0]
# spin up: [0,1]
h_bar = 1 # 1.054571817*10**(-34) # in J.s

t_list = [t for t in np.linspace(parameters["t_initial"],parameters["t_final"],parameters["n_steps"]+1)]    
L = parameters["n_sites"]
solver = AdiaStatePrepa()
solver.load_settings(parameters)

assert callable(getattr(solver, 'solve', None)), "ASPSolver has a method solve"
solver.solve()
# It must be possible to get the list of amplitudes obtained from solve.

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
plt.show()
# %%
# Plot the time evolution of the state in the computational basis.
import matplotlib.pyplot as plt

legend_list = []

for i in range(2**L):
    plt.plot(t_list, abs(amplitudes[:,i])**2, '-')
    # legend_list.append(str("|")+bin(i)[2:]+str(">"))

plt.xlabel('time t')
plt.ylabel('Probability')
plt.title('Probabilities of the states in the basis {|0>,|1>,...,|2**L-1>}')
# plt.legend(legend_list)
plt.show()


# %%
