# run_experiments.py
import numpy as np
import matplotlib.pyplot as plt
from channel_methods import channel


def run_experiment(params):
    ch = channel(**params)
    ch.cool_and_flow_iter(100)  # Run the experiment for a number of iterations
    return ch.T_panel_matrix, ch.T_fluid_matrix


def save_results(filename, T_panel_matrix, T_fluid_matrix):
    np.savez(filename, T_panel_matrix=T_panel_matrix, T_fluid_matrix=T_fluid_matrix)


if __name__ == "__main__":
    experiments = [
        {"T_fluid_i": 290, "mass_flow_rate": 0.001},
        {"T_fluid_i": 300, "mass_flow_rate": 0.002},
        # Add more experiment parameters here
    ]

    for i, params in enumerate(experiments):
        T_panel_matrix, T_fluid_matrix = run_experiment(params)
        save_results(f"experiment_{i}.npz", T_panel_matrix, T_fluid_matrix)
