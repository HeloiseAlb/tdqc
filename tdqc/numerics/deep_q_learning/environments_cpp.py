"""Model defining the environement used in deep_q_learning.py

This environement uses physical systems defined in system_cpp, coded in c++.
Only the derived class DynamicalEvolution is usable.
(There used to be more)

States are stored as list of previous actions. Because the initial wave function is fixed, this completely defines to the actual state,
the quantum wave (albeit maybe not uniquely).
The way the states are used to feed the neural networks depends on the actual
NN model used, coded in models.py.
"""

import sys
#for p in sys.path:
#    print(str('p'),p)
import numpy as np
import cmath
from math import pi, log2, sqrt
import tdqc.numerics.deep_q_learning.system_cpp.system.cpp as sy

class QuantumEnv():
    """ Quantum environment using QuDyn (cpp) for time evolution.
    """
    def __init__(self,
                 n_sites,
                 n_steps,
                 n_directions,
                 gate_order,
                 system_class,
                 ham_params,
                 time_segment,
                 initial_state,
                 seed_initial_state,
                 range_one,
                 range_all,
                 measurement,
                 bulk_size=0,
                 entangling_gates_dir='jx',
                 #  weighted_average=False,
                 average_exponent=1.0,
                 periodic_boundary_conditions=False,
                 **other_params):
        
        """Define the model of the system. The Hamiltonian of a system is defined in the system.cpp..
        """
        
        if system_class == 'LongRangeIsing':
            self.system = sy.LongRangeIsing(False)
            self.system.set_system(
                n_sites=n_sites,
                n_steps=n_steps,
                jx=ham_params['J'],
                hx=ham_params['g'],
                hz=ham_params['h'],
                alpha=ham_params['alpha'],
                time_segment=time_segment,
                gate_order=gate_order,
                entangling_gates_dir=entangling_gates_dir,
                average_exponent=average_exponent,
                periodic_boundary_conditions=periodic_boundary_conditions
            )
        
        elif system_class == 'Schwinger':
            self.system = sy.Schwinger(False)
            self.system.set_system(
                n_sites=n_sites,
                n_steps=n_steps,
                m=ham_params['m_c'],
                w=ham_params['w_c'],
                j=ham_params['j_c'],
                alpha=ham_params['alpha'],
                time_segment=time_segment,
                gate_order=gate_order,
                entangling_gates_dir=entangling_gates_dir,
                average_exponent=average_exponent,
            )
        else:
            raise ValueError('The system_class is not implemented')

        
        self.ham_params = ham_params
        self.system_class = system_class
        self.time_segment = time_segment

        self.n_steps = n_steps
        self.n_sites = n_sites
        self.n_directions = n_directions
        self.range_one = range_one
        self.range_all = range_all

        self.action_dim = 2 * n_sites + 1

        if self.n_directions != 2:
            raise NotImplementedError(f'not implemented for n_directions = '
                                      f'{self.n_directions}.')
        
        self.measurement = measurement
        self.set_initial_state(seed_initial_state,
                               initial_state)
        self.reset()

    def get_action_dim(self):
        return self.action_dim

    def get_n_sites(self):
        return self.n_sites

    def set_initial_state(self, seed, initial_state):
        if initial_state == 'random_product_state':
            np.random.seed(seed)
            #  randomly directed vector on the unit sphere
            #  f(theta)g(phi)dtheta dphi = sin(theta)/4pi
            #  g(phi) = 1/2pi, f(theta) = sin(theta)/2
            #  -> F(theta) = (1-cos(theta))/2
            #  F^-1(u) = arccos(1 - 2u)
            phis = [2*pi*np.random.rand() for _ in range(self.n_sites)]
            thetas = [np.arccos(1-2*np.random.rand()) for _ in
                      range(self.n_sites)]
            spinors = [np.array([np.cos(theta/2),
                                 np.exp(1j*phi)*np.sin(theta/2)])
                       for phi, theta in zip(phis, thetas)]
            rstate = tensor_prod(*spinors)
            self.state_real = rstate.real
            self.state_imag = rstate.imag
        elif initial_state == 'ferro':
            self.state_real = np.zeros(2**self.n_sites)
            self.state_real[0] = 1.0
            self.state_imag = np.zeros(2**self.n_sites)
        elif initial_state == 'antiferro':
            self.state_imag = np.zeros(2**self.n_sites)
            spinors = [np.array([1.0, 0.0]) if _ % 2 == 0
                       else np.array([0.0, 1.0]) for _ in range(self.n_sites)]
            self.state_real = tensor_prod(*spinors)
        else:
            raise NotImplementedError(f'Initial state of type {initial_state} '
                                      'not implemented.')
        
        #  the flip is used to be consitent with how states are encoded in
        #  the QuDyn library.
        state_real = np.flip(self.state_real, axis=0)
        state_imag = np.flip(self.state_imag, axis=0)
        
        # I NEED TO CHECK AND MODIFY THE CODE TO SUPPRESS THE NEXT FEW LINES.
        '''
        if calculate_target_state:
            set_rho_target = (self.measurement == 'trace_distance' or
                              self.measurement == 'relative_entropy')
        '''
        #    self.system.set_target_state(set_rho_target=set_rho_target)


    def decode_action_sequence(self, action_sequence):
        """ Given [..., ai, ...], the list of actions, return list of gates jx, [.., hx_i, ...],
        [..., hz_i, ...]"""
        if self.n_directions != 2:  # and self.n_directions != 3:
            raise ValueError('decode_action_sequence is only implemented '
                             'for n_directions = 2.')
        jx_gates = []
        hz_gates = []
        hx_gates = []
        assert len(action_sequence) == self.n_steps
        assert len(action_sequence[0]) == self.action_dim
        n = self.n_sites
        for action in action_sequence:
            jx_gates.append(action[0])
            hz_gates.append(action[1:n + 1])
            hx_gates.append(action[n + 1:2 * n + 1])

        jx_gates = np.array(jx_gates) * self.range_all
        hz_gates = np.array(hz_gates) * self.range_one
        hx_gates = np.array(hx_gates) * self.range_one
        return jx_gates, hx_gates, hz_gates

    def render(self, action_sequence, outfile=sys.stdout):
        jx_gates, hx_gates, hz_gates = \
            self.decode_action_sequence(action_sequence)
        outfile.write('\n')
        for n in range(self.n_steps):
            outfile.write(f'Step {n}:\n')
            outfile.write(f'   jx gates: {jx_gates[n]:.2f}\n')
            outfile.write(
                '   hz gates: [' + ', '.join(
                    [f'{g:.2f}' for g in hz_gates[n]]
                ) + ']\n'
            )
            outfile.write(
                '   hx gates: [' + ', '.join(
                    [f'{g:.2f}' for g in hx_gates[n]]
                ) + ']\n'
            )
            outfile.write('\n')

    def step(self, action):
        self.current_state, done = self.get_transition(self.current_state,
                                                       action)
        if done:
            reward = self.reward(self.current_state,rho_target=self.__rho_target)
        else:
            reward = 0.0
        return (self.current_state, reward, done, {})

    def reset(self):
        self.current_state = []
        return self.current_state

    def get_transition(self, state, action):
        state.append(action)
        done = (len(state) == self.n_steps)
        return (state, done)

    def random_action(self):
        return np.random.uniform(-1, 1, size=self.action_dim)




class DynamicalEvolution(QuantumEnv):

    def __init__(self, measurement, **other_params):
        super().__init__(measurement=measurement,
                         calculate_target_state=True,
                         **other_params)

    def reward(self,action_sequence,rho_target):
        n_qubits = self.n_sites
        j_gates, hx_gates, hz_gates = \
            self.decode_action_sequence(action_sequence)
        # self.system.set_gates(j_gates, hx_gates, hz_gates)
        # No sure about the following line!! Checked the nature of the object
        print("self.current_state:{}".format(self.current_state))
        self.system.set_gates(j_gates, hx_gates, hz_gates)
        rho_DQS_real, rho_DQS_im = self.current_state
        rho_DQS = rho_DQS_real + 1j*rho_DQS_im
        print(rho_DQS) 
        return local_reward(rho_DQS,rho_target,n_qubits)

    def measurement_from_gates(self, jx_gates, hx_gates, hz_gates,
                               measurement=None):

        self.system.set_gates(jx_gates, hx_gates, hz_gates)
        if measurement is None:
            measurement = self.measurement
        #  if measurement == "energy_contributions":
        #      return self.system.get_energy_contributions()
        #  elif measurement == "local_energy":
        #      return self.system.get_local_energy()

        meas = self.system.start(measurement=measurement)
        #  if measurement == "spin_correlations":
        #      return self.system.get_spin_correlations()
        #  elif measurement == "local_fluctuations_zz":
        #      return self.system.get_local_fluctuations_zz()
        return meas

    def measurement_target_state(self, measurement=None):
        if measurement is None:
            measurement = self.measurement
        #  if measurement == "energy_contributions":
        #      return self.system.get_target_energy_contributions()
        #  elif measurement == "local_energy":
        #      return self.system.get_target_local_energy()

        meas = self.system.measurement_target_state(measurement)
        #  if measurement == "spin_correlations":
        #      return self.system.get_target_spin_correlations()
        #  elif measurement == "local_fluctuations_zz":
        #      return self.system.get_target_local_fluctuations_zz()
        return meas

    def measurement_trotter(self, measurement=None):
        if self.system_class != 'LongRangeIsing':
            return None
            #  raise ValueError('Trotter decomposition only defined for LRI.')
        trotter_action_sequence = self.initial_action_sequence()
        jx_gates, hx_gates, hz_gates = \
            self.decode_action_sequence(trotter_action_sequence)

        return self.measurement_from_gates(jx_gates=jx_gates,
                                           hx_gates=hx_gates,
                                           hz_gates=hz_gates,
                                           measurement=measurement)

    def get_ground_state_energy(self, return_eigenvectors=False):
        """Return the ground state energy."""
        energy = self.system.get_ground_state_energy()
        return energy

    def initial_action_sequence(self, all_zeros=False):
        """Return an initial list of actions for each step.

        When the Trotter decomposition exists, this is
        = [[a_1, a_2, ...]]*n_steps
        reminder: each a_j correspond to a gate (single qubit or entangling)

        parameters:
            all_zeros (int): if True, all a_j = 0.0.

        Returns: np.array with shape=(n_steps, action_dim)
        """

        if self.system_class == 'LongRangeIsing':
            if self.n_directions == 2:
                a_all = self.ham_params['J'] * self.time_segment \
                    / self.n_steps / self.range_all
                a_onex = self.ham_params['g'] * self.time_segment \
                    / self.n_steps / self.range_one
                a_onez = self.ham_params['h'] * self.time_segment \
                    / self.n_steps / self.range_one
                action = [a_all] + [a_onez] * self.n_sites \
                    + [a_onex] * self.n_sites
                return [action] * self.n_steps
            else:
                raise NotImplementedError('Trotter sequence only implemented'
                                          ' for n_directions = 2.')
        else:
            if all_zeros:
                return np.zeros(shape=(self.n_steps, self.action_dim))
            return np.random.uniform(-1, 1, size=(self.n_steps,
                                                  self.action_dim))

def local_reward(rho1,rho2,n_qubits=None):
    if n_qubits == None:
        n_qubits = int(log2(rho1.shape[0]))
    sum_measures = 0
    for j in range(0,n_qubits-1):
        for k in range(j+1,n_qubits):
            sum_measures += cmath.sqrt(relative_entropy(reduced_density_matrix(rho1,j,k),reduced_density_matrix(rho2,j,k)))
    r_local = 1 - 2/(n_qubits*(n_qubits-1))*sum_measures
    return r_local

def reduced_density_matrix(rho_init,site1,site2,n_qubits=None):
    # Return the reduced density matrix of the subsystem made of sites site1 and site2 for rho. 
    rho = rho_init
    if n_qubits == None:
        n_qubits = int(log2(rho.shape[0]))
    n = n_qubits
    if site1>site2:
        site1,site2 = site2,site1
    if site1>0:
        n1,n2 =int(2**(site1)), int(2**(n-site1))
        rho = rho.reshape([n1,n2,n1,n2])
        rho = np.trace(rho,axis1=0,axis2=2)
        n -= site1
        site2 -= site1
    if site2>1:
        n1,n2 = int(2**(site2-1)),int(2**(n-site2))
        rho = rho.reshape([2,n1,n2,2,n1,n2])
        rho = np.trace(rho,axis1=1,axis2=4)
        n -= site2-1
    if n>2:
        n2 = int(2**(n-2))
        rho = rho.reshape([4,n2,4,n2])
        rho = np.trace(rho,axis1=1,axis2=3)
    rho = rho.reshape([4,4])
    return rho


def relative_entropy(rho1,rho2):
    return np.trace(rho1*(np.log(rho1)-np.log(rho2)))

def tensor_prod(*arg):
    """
    tensor_prod(a1, a2) = np.kron(a1, a2).
    tensor_prod(a1, a2, ..., an) = np.kron(tensor_prod(a1, ..., an-1), an)
    """
    res = arg[0]
    for i in range(1, len(arg)):
        res = np.kron(res, arg[i])
    #  res = arg[-1]
    #  for i in range(1, len(arg)):
    #      res = np.kron(res, arg[len(arg) - i - 1])
    return res
