#%%
import pytest
import tdqc
import numpy as np
import sys 
import cmath
import math

import matplotlib.pyplot as plt
from tdqc.numerics.ed.exact_diagonalisation import *

from tdqc.solver.asp import AdiaStatePrepa
from tdqc.numerics.ed.models_ed import State
from tdqc.numerics.ed.models_ed import trans_ising_model
from tdqc.numerics.asp.parameters_asp import parameters

#%%
def plot_eigenvalues_evolution():
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
    plt.title('Evolution of each eigenvalues')
    plt.legend()

    # Save the plot
    plt.savefig(f'evolution_eigenvalues_N{L}.png')

# plot_eigenvalues_evolution()

#%%
def plot_time_evolution(plot_fidelities = True,plot_amplitudes = True  ):
    """
    Plot the time evolution of the selected variables among:
    - the fidelities between the ground state of H_t_n and the state of the system,
    - the difference between the ground state energy of H_t_n and the energy state of the system,
    """
    t_list = [t for t in np.linspace(parameters["t_initial"], parameters["t_final"], parameters["n_steps"])]   

    L = parameters["n_sites"]
    solver = AdiaStatePrepa()
    solver.load_settings(parameters)
    solver.solve(ED = False)

    fidelities = solver.list_fidelities
    amplitudes = solver.time_evolution
    gaps = solver.list_gaps
    list_difference_energy_with_gs_hamiltonian = solver.list_difference_energy_with_gs_hamiltonian
    print("average fidelity: {}".format(np.average(abs(fidelities[:]))))
    
    if plot_fidelities:
        fig1 = plt.figure() 
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
    t_limit = largest_singular_value/delta_s**2
    print("T limit:{}".format(t_limit))
    if t_limit < parameters["t_initial"]- parameters["t_final"]:
        print("The simulation time is not long enough to apply the adiabatic therorem.")
        

    fig2 = plt.figure() 
    plt.plot(t_list,list_difference_energy_with_gs_hamiltonian.real)
    plt.xlabel('time t')
    #plt.ylabel(r'$\langle$ $\psi$(t)|H(t)|$\psi$(t)$\rangle$ - E_0(t)')
    #plt.title('Difference between energy of the system at time t \n and the ground state energy \n of the Hamiltonian at time t: H(t)')
    #plt.savefig('my_plot_list_difference_energy_with_gs_hamiltonian.png')

    plt.ylabel(r'$\langle$ $\psi$(t)|H(t)|$\psi$(t)$\rangle$')
    plt.title('Energy of the system at time t')
    plt.savefig('my_plot_list_energies.png')

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
        plt.savefig('my_plot_amplitudes.png')




    


plot_time_evolution()





    