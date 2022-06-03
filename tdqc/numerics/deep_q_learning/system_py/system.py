import numpy as np
import subprocess


class SpinSystem():
    # Why was this object called by the Markus and Adrien SPINSystem ?
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
        self.set_coupling_matrix()

    def set_gates(self, jx_angle_list, hx_angle_list, hz_angle_list):
        jx_gates = np.zeros([self.n_steps],dtype=complex)#,self.n_sites,self.n_sites], dtype=complex)
        hz_gates = np.zeros([self.n_steps,self.n_sites],dtype=complex)#,self.n_sites], dtype=complex)
        hx_gates = np.zeros([self.n_steps,self.n_sites],dtype=complex)#,self.n_sites], dtype=complex)
        for step in range(0, self.n_steps,1):
            jx_gates[step] = jx_angle_list[step]# * self.coupling_matrix
            vhx = np.zeros(self.n_sites)
            vhz = np.zeros(self.n_sites)
            '''
            for site in range(0,self.n_sites,1):
                vhx[site] = 2.0 *hx_angle_list[step][self.n_sites-1-site]
                vhz[site] = 2.0 *hz_angle_list[step][self.n_sites-1-site]
            hx_gates[step] = vhx 
            hz_gates[step] = vhz
            '''
        # The following np.arrays are as of size [n_steps, n_sites, n_sites]. They store the angles of the ie the parameters of the three kind of gates that are apply here. 
        self.jx_gate_list = jx_gates
        self.hx_gate_list = hx_gates
        self.hz_gate_list = hz_gates

    def set_coupling_matrix(self,):
        coupling_matrix = np.zeros((self.n_sites,self.n_sites))
        for i in range(0,self.n_sites,1):
            for j in range(i+1,self.n_sites,1):
                coupling_matrix[i,j] = 4.0/((j-i)**self.alpha)
                # Not needed but to be sure.
                coupling_matrix[j,i] = 4.0/((j-i)**self.alpha)
        self.coupling_matrix = coupling_matrix

    def start(self, measurement):
        # Here we need to run the simulation of the gate sequence.
        # Return a measurement (a double precision floating-point values). 
        print("The function SPinSytem.start is used. It need to be done.")
        subprocess.call("echo 'Here we call the MPS code to simulate the gate sequence'", shell=True)
     

