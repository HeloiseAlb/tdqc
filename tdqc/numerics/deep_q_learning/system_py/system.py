import numpy as np
import subprocess


class SpinSystem():
    # Why was this object called by the Markus and Adrien SpinSystem? For me,
    # it is not linked to spin particularly. 
    def __init__(self,):
        pass

    def set_system(self, n_sites,
                n_steps,
                t_initial,
                t_final,gate_order,alpha):
        self.n_sites = n_sites
        self.n_steps = n_steps
        self.t_initial = t_initial
        self.t_final = t_final
        self.gate_order = gate_order
        self.alpha = alpha
        #self.set_coupling_matrix()

    def set_gates(self, jx_angle_list, hx_angle_list, hz_angle_list):
        # The function sets the attributs 'jx_gate_list', 'hx_gate_list' and 'hz_gate_list' to the object.
        
        # The arrays do not need to be reshaped and do not need to contain complex values.
        #jx_gates = np.array(jx_angle_list.reshape((self.n_steps)),dtype=complex) #,self.n_sites,self.n_sites], 
        #hz_gates = np.array(hz_angle_list.reshape((self.n_steps,self.n_sites)),dtype=complex)
        #hx_gates = np.array(hx_angle_list.reshape((self.n_steps,self.n_sites)),dtype=complex)
        self.jx_gate_list = jx_angle_list
        self.hx_gate_list = hx_angle_list
        self.hz_gate_list = hz_angle_list
        #print(' self.jx_gate_list = {} and jx_angle_list = {}'.format(self.jx_gate_list,jx_angle_list))
        #print(' self.hx_gate_list = {} and hx_angle_list = {}'.format(self.hx_gate_list, hx_angle_list))
        #print(' self.hz_gate_list = {} and hz_angle_list = {}'.format(self.hz_gate_list, hz_angle_list))

    def start(self, measurement):
        # Here we need to run the simulation of the gate sequence.
        # Return a measurement (a double precision floating-point values). 
        print("The function SPinSytem.start is used. It need to be done.")
        subprocess.call("echo 'Here we call the MPS code to simulate the gate sequence'", shell=True)
     

