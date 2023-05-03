import pytest
import tdqc
import numpy as np
from tdqc.numerics.ed.models_ed import State
from tdqc.numerics.ed.models_ed import xxz_model


range_t_final = np.array([9.0])

for t_final_local in range_t_final:
    from tdqc.solver.deep_q_learning import DeepQLearning, DQLWithReplayMemory
    from tdqc.numerics.deep_q_learning.parameters_lri import parameters, parameters_replay_memory
    print(t_final_local)
    solver = DQLWithReplayMemory()
    parameters.update({"t_final":t_final_local})
    solver.load_settings(settings=parameters)
    solver.load_seetings_replay_memory(**parameters_replay_memory)
    solver.solve()


