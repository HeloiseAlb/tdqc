import subprocess

class LongRangeIsing():

    def __init__(self, dummyBool):

        pass

    def set_system(self, n_sites, n_steps,
                         jx, hx, hz, alpha, time_segment,
                         gate_order, entangling_gates_dir,
                         average_exponent, periodic_boundary_conditions):

        self.n_sites = n_sites
        self.n_steps = n_steps
        self.jx = jx
        self.hx = hx
        self.hz = hz
        self.alpha = alpha
        self.t0 = time_segment

    def set_initial_state(self, state_real, state_imag):

        pass

    def set_gates(self, jx_gate_list, hx_gate_list, hz_gate_list):
        self.jx_angles = jx_gate_list
        self.hx_angles = hx_gate_list
        self.hz_angles = hz_gate_list


    def start(self, measurement):
        # Here we need to run the simulation of the gate sequence
        subprocess.call("echo 'Here we call the MPS code to simulate the gate sequence'", shell=True)

    def get_ground_state_energy(self):

        return 0.


    def set_target_state(self, set_rho_target):
        # set_rho_target is a boolean saying whether to compute reduced density matrices

        # Here we need to run the simulation of the physical dynamics up to time t=time_segment
        subprocess.call("echo 'Here we call the MPS code to compute the target state'", shell=True)


    def measurement_target_state(self, measurement):

        pass

if __name__ == "__main__":

    system = LongRangeIsing(False)
    system.set_system(10, 3, 1., 0.2, 0.3, 2., 1.2, "xz", "?", "?", False)
    system.set_target_state(True)
    system.start("rdm")
