#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "hamiltonians.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(system_cpp, m) {
    py::class_<LongRangeIsing>(m, "LongRangeIsing")
        .def(py::init<const bool &>())
        .def("set_system", &LongRangeIsing::setSystem,
             "n_sites"_a, "n_steps"_a, "jx"_a, "hx"_a, "hz"_a, "alpha"_a,
             "time_segment"_a, "gate_order"_a, "entangling_gates_dir"_a,
             "average_exponent"_a, "periodic_boundary_conditions"_a)
        .def("set_initial_state", &LongRangeIsing::setInitialState,
             "state_real"_a, "state_imag"_a)
        .def("set_gates", &LongRangeIsing::setGates,
             "jx_gate_list"_a, "hx_gate_list"_a, "hz_gate_list"_a)
        .def("start", &LongRangeIsing::start, "measurement"_a)
        .def("get_ground_state_energy", &LongRangeIsing::getGroundStateEnergy)
        .def("set_target_state",
                &LongRangeIsing::setTargetState, "set_rho_target"_a)
        .def("measurement_target_state", &LongRangeIsing::measurementTargetState,
                "measurement"_a);


    py::class_<Schwinger>(m, "Schwinger")
        .def(py::init<const bool &>())
        .def("set_system", &Schwinger::setSystem,
             "n_sites"_a, "n_steps"_a, "m"_a, "w"_a, "j"_a, "alpha"_a,
             "time_segment"_a, "gate_order"_a, "entangling_gates_dir"_a,
             "average_exponent"_a)
        .def("set_initial_state", &Schwinger::setInitialState,
             "state_real"_a, "state_imag"_a)
        .def("set_gates", &Schwinger::setGates,
             "jx_gate_list"_a, "hx_gate_list"_a, "hz_gate_list"_a)
        .def("start", &Schwinger::start, "measurement"_a)
        .def("measurement_target_state", &Schwinger::measurementTargetState,
                "measurement"_a)
        .def("get_ground_state_energy", &Schwinger::getGroundStateEnergy)
        .def("set_target_state", &Schwinger::setTargetState, "set_rho_target"_a);
};
