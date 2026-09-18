"""Run dimer energy transfer and compare with the stored BD = 20 reference."""

from pathlib import Path
from runpy import run_path

import numpy as np

from dampyf import DampfConfig, LocalPseudomodes, eigenstate_density_matrix, run_dampf, units

EXAMPLE_FOLDER = Path(__file__).resolve().parent
hamiltonian = np.array([[1000.0, 100.0], [100.0, 1200.0]])
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
    checkpoint_interval_seconds=None,
    execution_mode="local",  # Use "slurm" when submitting the Slurm launcher.
)

initial_eigenstate = 1
output_directory = EXAMPLE_FOLDER / "output_data"
output_identifier = f"dimer_bd{config.maximum_bond_dimension}"
reference_bond_dimension = 20
reference_file = EXAMPLE_FOLDER / "reference_data" / f"rho_system_bd{reference_bond_dimension}.npz"
show_plots = config.execution_mode == "local"
verbose = True


if __name__ == "__main__":
    import matplotlib

    if not show_plots:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    initial_density_matrix = eigenstate_density_matrix(hamiltonian, initial_eigenstate)
    result = run_dampf(
        hamiltonian,
        pseudomodes,
        config,
        initial_density_matrix=initial_density_matrix,
        output_directory=output_directory,
        output_identifier=output_identifier,
        verbose=verbose,
    )
    analysis = run_path(str(EXAMPLE_FOLDER / "analysis_tools" / "plot_system_dynamics.py"))
    analysis["main"](show_plots=False)
    if show_plots:
        plt.show()
    else:
        plt.close("all")
