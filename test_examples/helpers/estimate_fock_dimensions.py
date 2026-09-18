"""Estimate Fock dimensions for a local pseudomode environment defined by arrays."""

import numpy as np

from dampyf import LocalPseudomodes, estimate_fock_dimensions, units

environment = LocalPseudomodes(
    frequencies=np.array([400.0, 500.0, 600.0, 700.0, 800.0]),
    damping_rates=np.array([10.0, 20.0, 15.0, 30.0, 10.0]),
    couplings=np.array([20.0, 40.0, 70.0, 70.0, 50.0]),
    thermal_energies=np.full(5, 20.0),
    fock_dimensions=np.full(5, 4),  # Existing cutoffs; the estimator determines new ones.
)

time = 100.0 * units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME
time_step = 1.0 * units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME
population_threshold = 1.0e-6
average_population_threshold = 1.0e-6
minimum_dimension = 2
maximum_dimension = 30
safety_padding = 0


if __name__ == "__main__":
    estimate = estimate_fock_dimensions(
        environment,
        time,
        time_step,
        population_threshold=population_threshold,
        average_population_threshold=average_population_threshold,
        minimum_dimension=minimum_dimension,
        maximum_dimension=maximum_dimension,
        safety_padding=safety_padding,
    )
    for result in estimate["mode_results"]:
        print(
            f"mode {result['mode']}: fock_dim={result['recommended_dimension']}, "
            f"max cutoff population={result['maximum_highest_fock_state_population']:.4e}, "
            f"average cutoff population={result['average_highest_fock_state_population']:.4e}"
        )
    print("Recommended fock_dimensions:", estimate["fock_dimensions"])
