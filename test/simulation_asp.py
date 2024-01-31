
#%%
import tdqc
from tdqc.solver.asp import AdiaStatePrepa
from tdqc.numerics.asp.parameters_asp import parameters

solver = AdiaStatePrepa()
solver.load_settings(parameters)
solver.solve()
solver.compute_property_lists()
solver.generate_data_files()
solver.save_gate_sequence()



# %%
