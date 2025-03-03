import pandas as pd
import numpy as np
from channel_methods import channel

# Load Mojave dataset
df = pd.read_csv("mojave_summer_clear_days.csv")

# Define constants & parameter ranges
Tcoolant_range = (20, 30)  # Coolant inlet temperature (°C)
# will need to redefine methods for volumetric flowrate if not using water
mass_flowrate_range = np.linspace(
    0.0001, 0.01, 50
)  # Test flow rates from 0.0001 kg/s to 0.01 kg/s (assuming water)
PV_EFFICIENCY_REF = 0.20  # 20% efficiency (upper bound) at STC conditions
TEMP_COEFF = -0.0045  # -0.45% per °C efficiency loss
T_REF = 25  # Reference temperature (°C)


def assess_efficiency_increase(T_panel_no_cooling, T_panel_with_cooling):
    """Computes efficiency improvement due to cooling."""
    eta_no_cooling = PV_EFFICIENCY_REF * (1 + TEMP_COEFF * (T_REF - T_panel_no_cooling))
    eta_with_cooling = PV_EFFICIENCY_REF * (
        1 + TEMP_COEFF * (T_REF - T_panel_with_cooling)
    )
    efficiency_gain = eta_with_cooling - eta_no_cooling
    return efficiency_gain


def run_experiment(Tamb, I, Tcoolant_in, mass_flowrate, cooling):
    """Runs a single cooling experiment and returns the temperature profile."""

    print(
        f"Running experiment with Tamb={Tamb}°C, I={I} W/m^2, Tcoolant_in={Tcoolant_in}°C, mass_flowrate={mass_flowrate} kg/s, cooling={'On' if cooling else 'Off'}"
    )

    # Redefine experimental variables in appropriate units
    Tamb = Tamb + 273.15  # Convert to Kelvin
    Tcoolant_in = Tcoolant_in + 273.15  # Convert to Kelvin
    # irradiance already in W/m^2
    # mass flow rate already in kg/s

    panel_experiment = channel(
        T_ambient=Tamb,
        T_fluid_i=Tcoolant_in,
        intensity=I,  # irradiance
        mass_flow_rate=mass_flowrate,
    )

    # Run the experiment and let hit steady state
    if cooling:
        panel_experiment.cool_and_flow_iter(1000)
    else:  # No cooling
        panel_experiment.cool_and_flow_iter(0)

    # Return the temperature profile
    return panel_experiment.T_panel_matrix


def find_optimal_flowrate(Tamb, I, Tcoolant_in):
    """Runs cooling simulation at multiple flow rates and finds the optimal one."""
    best_flowrate = 0
    prev_efficiency_gain = 0
    efficiency_threshold = 0.05  # Require at least 5% efficiency gain

    for mass_flowrate in mass_flowrate_range:
        print(f"Testing mass flowrate: {mass_flowrate} kg/s")

        # simulate no cooling and cooling
        T_panel_no_cooling = run_experiment(
            Tamb, I, Tcoolant_in, mass_flowrate, cooling=False
        )
        T_panel_with_cooling = run_experiment(
            Tamb, I, Tcoolant_in, mass_flowrate, cooling=True
        )

        # for ease, use mean temperature of panel
        efficiency_gain = assess_efficiency_increase(
            np.mean(T_panel_no_cooling), np.mean(T_panel_with_cooling)
        )  # Compute efficiency improvement

        print(f"Efficiency gain: {efficiency_gain}")

        # if efficiency increase sufficient or leads to decrease, return flowrate
        if (
            (efficiency_gain >= efficiency_threshold)
            | (efficiency_gain < prev_efficiency_gain)
            | (efficiency_gain < 0)
        ):
            best_flowrate = mass_flowrate
            break  # Stop at the first flowrate that meets the requirement

        prev_efficiency_gain = efficiency_gain

    return best_flowrate


# Generate dataset
num_samples = len(df)  # Use all available environmental data

# Storage for results
data = []

# Limit to the first few rows for testing
num_samples_to_test = 5

for index, row in df.iterrows():
    if index >= num_samples_to_test:
        break

    Tamb = row["air_temperature"]
    I = row["ghi"]
    Tcoolant_in = np.random.uniform(
        *Tcoolant_range
    )  # Randomly sample coolant inlet temp

    # Log progress
    print(f"Processing sample {index + 1}/{num_samples_to_test}")

    mass_flowrate_optimal = find_optimal_flowrate(Tamb, I, Tcoolant_in)

    data.append([Tamb, I, Tcoolant_in, mass_flowrate_optimal])

# Convert to DataFrame and save to CSV
final_df = pd.DataFrame(
    data, columns=["Tamb", "I", "Tcoolant_in", "mass_flowrate_optimal"]
)
final_df.to_csv("solar_cooling_training_data_test.csv", index=False)

print("Test dataset generation complete. Saved as solar_cooling_training_data_test.csv")
