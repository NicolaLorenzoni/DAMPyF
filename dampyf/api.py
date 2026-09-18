"""
    Public DAMPyF API using in-memory model arrays.
"""

from pathlib import Path

from .configuration import DampfConfig
from .parameters import create_pseudomode_parameters, create_system_parameters
from .structures import DampfResult


def run_dampf(
    hamiltonian,
    pseudomodes,
    config: DampfConfig,
    *,
    initial_density_matrix=None,
    electric_dipoles=None,
    magnetic_dipoles=None,
    restart_from=None,
    output_directory=None,
    output_identifier="simulation",
    verbose=True,
) -> DampfResult:
    """
        Run one DAMPyF calculation from model arrays and simulation settings.

    Parameters
    ----------
    hamiltonian : array_like, shape (N, N)
        Hermitian system Hamiltonian in the site basis, where N is the number of sites.

    pseudomodes : LocalPseudomodes or sequence
        One LocalPseudomodes instance applies the same local environment to every site.
        Alternatively, provide one LocalPseudomodes instance or None per site.
        None indicates that the corresponding site has no local pseudomodes.
        At least one site must have pseudomodes.

        Each LocalPseudomodes instance requires the following arrays, all with shape (Q,),
        where Q is the number of pseudomodes in that local environment:
        - frequencies: pseudomode frequencies.
        - damping_rates: positive pseudomode damping rates.
        - couplings: system-pseudomode coupling strengths; complex values are supported.
        - thermal_energies: thermal energies k_B T, with zero corresponding to zero temperature.
        - fock_dimensions: positive integer Fock-space dimensions.

    config : DampfConfig
        Numerical and execution settings.

        Required fields:
        - simulation_mode: "energy_transfer" or "linear_spectra".
        - time: total propagation duration, or additional duration when restarting.
        - time_step: integration timestep.
        - data_time_step: interval between recorded outputs.
        - maximum_bond_dimension: maximum retained MPS bond dimension.

        Optional fields:
        - compression_tolerance=1.0e-8: relative tolerance used in MPS compression.
        - compression_svd_method="eig": SVD implementation; "standard", "qr", or "eig".
        - system_update_coefficient_tolerance=1.0e-8: discard mixing coefficients at or below this magnitude.
        - trotter_order="second": Trotter decomposition order; "first" or "second".
        - checkpoint_interval_seconds=None: checkpoint interval in wall-clock seconds; None disables saving.
        - execution_mode="local": obtain resource settings from the local machine or a "slurm" allocation.
        - ray_max_parallel_system_updates=None: upper limit on simultaneous system-update tasks.
          None selects the limit automatically from the system size and estimated available memory.

        data_time_step must be a positive integer multiple of time_step.
        time must be a non-negative integer multiple of data_time_step.

    initial_density_matrix : array_like, shape (N, N), optional
        Initial system density matrix in the site basis.
        Required for energy_transfer when starting a new simulation.
        Leave as None for linear_spectra or when supplying restart_from.
        Default: None.

    electric_dipoles : array_like, shape (N, 3), optional
        Electric transition dipoles, with one row per site and columns ordered as x, y, z.
        Default: None.

    magnetic_dipoles : array_like, shape (N, 3), optional
        Magnetic transition dipoles, with one row per site and columns ordered as x, y, z.
        Default: None.

    restart_from : str or Path, optional
        Directory containing a saved DAMPyF MPS checkpoint.
        Use matching model parameters and leave initial_density_matrix as None.
        The returned time axis starts at zero at the checkpoint.
        config.time specifies the additional propagation duration.
        Default: None.

    output_directory : str or Path, optional
        Directory for simulation results, parameter records, and checkpoints.
        None returns results in memory without writing output files.
        Required when config.checkpoint_interval_seconds is specified.
        Default: None.

    output_identifier : str, optional
        Identifier appended to output filenames; must be non-empty and contain no path separators.
        Default: "simulation".

    verbose : bool, optional
        Print setup summaries, propagation progress, and elapsed run time.
        Default: True.

    Returns
    -------
    DampfResult
        Result object containing:
        - times: recorded simulation times, with shape (number_of_times,).
        - rho_system: reduced density matrices, with shape (number_of_times, N, N), for energy_transfer.
        - optical_coherence: coherence matrices, with shape (number_of_times, N, N), for linear_spectra.
        - system_output_file: path to the saved numerical results, or None.
        - simulation_data_file: path to the saved parameter and run record, or None.
        - elapsed_time_seconds: measured run time in wall-clock seconds.
        - final_maximum_bond_dimension: largest bond dimension in the final MPS collection.
        - maximum_bond_dimension_reached: largest recorded bond dimension during propagation.

        rho_system or optical_coherence is None when it does not apply to the selected mode.

    Notes
    -----
    Hamiltonian entries, frequencies, damping rates, couplings, and thermal energies must
    use consistent units. Simulation times must use the corresponding units with hbar = 1.
    For energies in cm^-1, convert times from femtoseconds using:
        time_fs * dampyf.units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME

    checkpoint_interval_seconds and elapsed_time_seconds refer to wall-clock seconds.

    This function starts and stops its own local Ray runtime.
    An already initialized Ray runtime causes a RuntimeError and is left untouched.
    In executable scripts, place the call inside if __name__ == "__main__":.
    """
    if not isinstance(config, DampfConfig):
        raise TypeError("config must be a DampfConfig instance.")

    system = create_system_parameters(hamiltonian, electric_dipoles, magnetic_dipoles)
    pseudomode_parameters = create_pseudomode_parameters(pseudomodes, system.N)

    from .core import run_dampf as _run_dampf

    return _run_dampf(
        config,
        system,
        pseudomode_parameters,
        initial_density_matrix=initial_density_matrix,
        restart_from=None if restart_from is None else Path(restart_from),
        output_directory=None if output_directory is None else Path(output_directory),
        output_identifier=output_identifier,
        verbose=verbose,
    )
