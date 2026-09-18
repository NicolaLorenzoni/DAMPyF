"""Estimate memory using the model arrays and simulation settings defined below."""

import numpy as np

from dampyf import DampfConfig, LocalPseudomodes, estimate_required_memory, units

hamiltonian = np.array([[1000.0, 100.0], [100.0, 1200.0]])
# A single LocalPseudomodes object specifies the same local environment for both sites.
pseudomodes = LocalPseudomodes(
    frequencies=np.array([400.0, 500.0, 600.0, 700.0, 800.0]),
    damping_rates=np.array([10.0, 20.0, 15.0, 30.0, 10.0]),
    couplings=np.array([20.0, 40.0, 70.0, 70.0, 50.0]),
    thermal_energies=np.full(5, 20.0),
    fock_dimensions=np.full(5, 4),
)
config = DampfConfig(
    simulation_mode="energy_transfer",
    time=1000.0 * units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME,
    time_step=1.0 * units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME,
    data_time_step=1.0 * units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME,
    maximum_bond_dimension=20,
)


if __name__ == "__main__":
    estimate = estimate_required_memory(config, hamiltonian, pseudomodes)
    print("DAMPyF memory estimate")
    print("======================")
    for name, value in estimate.items():
        label = name.replace("_", " ")
        if name.endswith("_gib"):
            print(f"{label:<38}: {value:12.6f} GiB")
        else:
            print(f"{label:<38}: {value}")
