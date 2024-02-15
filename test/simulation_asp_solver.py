#%%
import numpy as np
from math import pi

import matplotlib.pyplot as plt
from tdqc.numerics.ed.exact_diagonalisation import *
from tdqc.solver.asp import AdiaStatePrepa
# from tdqc.numerics.ed.models_ed import State
from tdqc.numerics.asp.parameters_asp_lrti import parameters

#%%
def plot_eigenvalues_evolution():
    ham_params = parameters["ham_params"]
    h =  ham_params['h']
    ferro_angle =  format(parameters['ferro_angle']/pi, '.1f')  
    n_steps = parameters["n_steps"]
    t_list = [t for t in np.linspace(parameters["t_initial"], parameters["t_final"], parameters["n_steps"])]    
    L = parameters["n_sites"]
    solver = AdiaStatePrepa()
    solver.load_settings(parameters)
    solver.solve()

    fidelities = solver.list_fidelities
    gaps = solver.list_gaps
    list_difference_energy_with_gs_hamiltonian = solver.list_difference_energy_with_gs_hamiltonian
    list_eigenvalues = solver.list_eigenvalues
    eigenvalues_time_evolution = np.transpose(list_eigenvalues)

    # Build the x-axis according to the dimension of the list_eigenvalues
    t_initial = parameters['t_initial']
    t_final = parameters['t_final']
    n_steps = parameters['n_steps']

    t_list = [t for t in np.linspace(t_initial, t_final, n_steps)]

    # Plot each eigenvalue evolution
    for i, line in enumerate(eigenvalues_time_evolution):
        plt.plot(t_list, line, label=f'E_ {i+1}')

    # Set labels and title
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.title(f'Evolution of each eigenvalues for h={h}')
    #plt.legend()

    # Save the plot
    plt.savefig(f'evolution_eigenvalues_N{L}_h{h}_ferro_angle{ferro_angle}_n_steps{parameters["n_steps"]}_lrtIsing.png')

plot_eigenvalues_evolution()

#%%
#def plot_time_evolution(plot_fidelities = True, plot_amplitudes = True, plot_eigenvector_probabilities = True):
"""
Plot the time evolution of the selected variables among:
- the fidelities between the ground state of H_t_n and the state of the system,
- the difference between the ground state energy of H_t_n and the energy state of the system,
- the probability amplitudes of the different state of the computational basis.
"""
plot_fidelities = True
plot_amplitudes = True
plot_eigenvector_probabilities = True
generate_files = False
parameters["n_steps"] = 3
g = 1.2
h = g
ferro_angle = 0.1
ham_params={'J': 1.0, 'g': g, 'h': h, 'alpha': int(2)}
parameters["ham_params"]= ham_params
parameters['ferro_angle'] = ferro_angle
t_list = [t for t in np.linspace(parameters["t_initial"], parameters["t_final"], parameters["n_steps"])]   
L = parameters["n_sites"]
solver = AdiaStatePrepa()
solver.load_settings(parameters)
solver.solve(ED = False)
fidelities = solver.list_fidelities
time_evolution = solver.time_evolution
list_ground_state_h_t_n = solver.list_ground_state_h_t_n

#%%
list_projections_on_gs = np.zeros(len(t_list))
for t, state_t_n in enumerate(time_evolution[:-1]):
    ground_state_h_t_n = list_ground_state_h_t_n[t]
    #print("ground_state_h_t_n:{}".format(ground_state_h_t_n))
    list_projections_on_gs[t] = abs(np.vdot(np.conj(ground_state_h_t_n), state_t_n))
    print("list_projections_on_gs[t]:{}".format(list_projections_on_gs[t]))
plt.plot(t_list, abs(list_projections_on_gs))
plt.xlabel('time t')
plt.ylabel('Fidelities')
plt.title('Absolute value of the fidelity between state \n of the system at time t and the ground state \n of the Hamiltonian at time t: H(t)')
plt.savefig(f'my_plot_projections_on_gs_N{L}_system{solver.system_class}_nsteps{solver.n_steps}_tfinal{solver.t_final}_g{g}_ferro_angle{ferro_angle}.png')
plt.show(block=False)


#%%

gaps = solver.list_gaps
list_difference_energy_with_gs_hamiltonian = solver.list_difference_energy_with_gs_hamiltonian
#print("average fidelity: {}".format(np.average(abs(fidelities[:]))))
#if plot_fidelities:
#fig1 = plt.figure() 
plt.plot(t_list, abs(fidelities[:]))
plt.xlabel('time t')
plt.ylabel('Fidelities')
plt.title('Absolute value of the fidelity between state \n of the system at time t and the ground state \n of the Hamiltonian at time t: H(t)')
#plt.savefig(f'my_plot_fidelities_N{L}_system{solver.system_class}_nsteps{solver.n_steps}_tfinal{solver.t_final}.png')
plt.show(block=False)

#%%
# Is T long enough?
delta_s_H = -parameters['model_0'].hamiltonian + parameters['model_f'].hamiltonian
_, singular_values, _ = np.linalg.svd(delta_s_H)

# Print the largest singular value
largest_singular_value = singular_values[0]
print("Largest singular value: {}".format(largest_singular_value))
delta_s = (parameters["t_final"]-parameters["t_initial"])/parameters["n_steps"]
t_limit = largest_singular_value/delta_s**2
print("T limit:{}".format(t_limit))
if t_limit < parameters["t_initial"]- parameters["t_final"]:
    print("The simulation time is not long enough to apply the adiabatic therorem.")

fig2 = plt.figure() 
plt.plot(t_list, list_difference_energy_with_gs_hamiltonian.real)
plt.xlabel('time t')
#plt.ylabel(r'$\langle$ $\psi$(t)|H(t)|$\psi$(t)$\rangle$ - E_0(t)')
#plt.title('Difference between energy of the system at time t \n and the ground state energy \n of the Hamiltonian at time t: H(t)')
#plt.savefig('my_plot_list_difference_energy_with_gs_hamiltonian.png')

plt.ylabel(r'$\langle$ $\psi$(t)|H(t)|$\psi$(t)$\rangle$')
plt.title('Energy of the system at time t')
if generate_files:
    plt.savefig(f'list_energies_N{L}_system{solver.system_class}_nsteps{solver.n_steps}_tfinal{solver.t_final}.png')
else:
    plt.show()

amplitudes = solver.time_evolution
if plot_amplitudes:
    fig3 = plt.figure()
    # Plot the time evolution of the state in the computational basis.
    legend_list = []
    for i in range(2**L):
        plt.plot(t_list,abs(amplitudes[:,i])**2,'-')
        legend_list.append(str("|")+bin(i)[2:]+str(">"))
    plt.xlabel('time t')
    plt.ylabel('Probability')
    plt.title('Probabilities of the states in the basis {|0>,|1>,...,|2**L-1>}')
    plt.legend(legend_list)
    if generate_files:
        plt.savefig(f'amplitudes_N{L}_system{solver.system_class}_nsteps{solver.n_steps}_tfinal{solver.t_final}.png')
    else:
        plt.show()

if plot_eigenvector_probabilities:
    fig4 = plt.figure()
    dim = 2**L
    list_eigenvectors = solver.list_eigenvectors
    # Initialize an array to store the projections
    projections = np.zeros((solver.n_steps, dim))
    for index, amplitude in enumerate(amplitudes):
        for index2, eigenvector in enumerate(list_eigenvectors[index,:,:]):
                    projections[index, index2] = abs(np.dot(amplitude, eigenvector))**2

    legend_list = []
    projections = np.transpose(projections)
    for index, line in enumerate(projections):
        plt.plot(t_list, line, label=f'P(E_{index})')
    plt.xlabel('time t')
    plt.ylabel('P_{E}')
    plt.title('Probabilities of the states in the eigenbasis')
    plt.legend(legend_list)
    if generate_files:
        plt.savefig(f'eigenvector_proba_N{L}_system{solver.system_class}_nsteps{solver.n_steps}_tfinal{solver.t_final}.png')
    else:
        plt.show()
    
if generate_files:
    solver.generate_data_files()
    

#plot_time_evolution(plot_fidelities = False, plot_amplitudes = False, plot_eigenvector_probabilities = True)


# %%
