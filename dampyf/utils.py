"""
    Generic utilities and validation for DAMPyF.
"""

import numpy as np
from .structures import SystemIndexMaps

BYTES_PER_GIB = 1024**3


def estimate_memory_requirements(config, pseudomodes):
    """
     Return shared estimates for the helper and runtime scheduler, in GiB.
    """
    number_of_sites = pseudomodes.N
    dimension_factor = sum(int(pseudomodes.fock_dim[site][mode]) ** 2 for site in range(number_of_sites) for mode in range(pseudomodes.Q[site]))
    mpo_dimension_factor = sum(int(pseudomodes.fock_dim[site][mode]) ** 4 for site in range(number_of_sites) for mode in range(pseudomodes.Q[site]))
    
    one_mps_gib = 16 * int(config.maximum_bond_dimension) ** 2 * dimension_factor / BYTES_PER_GIB
    one_rank_one_mpo_gib = 16 * mpo_dimension_factor / BYTES_PER_GIB
    
    stored_components = (number_of_sites * (number_of_sites + 1) // 2 if config.simulation_mode == "energy_transfer" else number_of_sites**2)
    local_mpos = stored_components if config.simulation_mode == "energy_transfer" else number_of_sites
    if config.trotter_order == "second":
        local_mpos *= 2
        
    stored_state_gib = stored_components * one_mps_gib
    mpos_gib = (local_mpos + 1) * one_rank_one_mpo_gib
    one_local_task_gib = 2.0 * one_mps_gib + one_rank_one_mpo_gib
    all_local_tasks_gib = stored_components * one_local_task_gib
    one_system_task_gib = 3.0 * number_of_sites * one_mps_gib
    all_system_tasks_gib = number_of_sites * one_system_task_gib
    peak_tasks_gib = max(all_local_tasks_gib, all_system_tasks_gib)
    
    return {
        "total_gib": stored_state_gib + mpos_gib + peak_tasks_gib,
        "one_mps_gib": one_mps_gib,
        "stored_state_gib": stored_state_gib,
        "one_rank_one_mpo_gib": one_rank_one_mpo_gib,
        "mpos_gib": mpos_gib,
        "one_local_update_task_gib": one_local_task_gib,
        "all_local_update_tasks_gib": all_local_tasks_gib,
        "one_system_update_task_gib": one_system_task_gib,
        "all_system_update_tasks_gib": all_system_tasks_gib,
        "peak_parallel_tasks_gib": peak_tasks_gib,
        "stored_components": stored_components,
        "local_update_mpos": local_mpos,
    }


def boltzmann_occupation(pseudomodes, n, q):
    """
        Return the thermal occupation of one pseudomode.
    """
    thermal_energy = pseudomodes.thermal_energy[n][q]
    if thermal_energy <= 0.0:
        return 0.0
    exponent = pseudomodes.Omega_osc[n][q] / thermal_energy
    return 0.0 if exponent > 700.0 else 1.0 / np.expm1(exponent)




def generate_system_index_maps(N):
    """
        Generate maps for the diagonal and upper-triangular system entries.
    """
    N = int(N)

    # Consistency check.
    if N <= 0:
        raise ValueError("N must be positive.")

    single_to_double_indices = []
    double_to_single_index = [[None for _ in range(N)] for _ in range(N)]

    for m in range(N):
        for n in range(m, N):
            ind = len(single_to_double_indices)
            single_to_double_indices.append((m, n))
            double_to_single_index[m][n] = ind

    return SystemIndexMaps(
        N=N,
        single_to_double_indices=single_to_double_indices,
        double_to_single_index=double_to_single_index,
    )



def check_user_parameters(config, system, initial_density_matrix, restart_from, output_directory, output_identifier):
    """
        Validate parameters.
    """
    if config.simulation_mode not in ("energy_transfer", "linear_spectra"):
        raise ValueError("simulation_mode must be either 'energy_transfer' or 'linear_spectra'.")
    if config.time < 0.0:
        raise ValueError("time must be non-negative.")
    if config.time_step <= 0.0:
        raise ValueError("time_step must be positive.")
    if config.data_time_step <= 0.0:
        raise ValueError("data_time_step must be positive.")
    if abs(config.data_time_step / config.time_step - round(config.data_time_step / config.time_step)) > 1.0e-12:
        raise ValueError("data_time_step must be a positive integer multiple of time_step.")
    if abs(config.time / config.data_time_step - round(config.time / config.data_time_step)) > 1.0e-12:
        raise ValueError("time must be an integer multiple of data_time_step.")
    if config.trotter_order not in ("first", "second"):
        raise ValueError("trotter_order must be either 'first' or 'second'.")
    if (
        isinstance(config.maximum_bond_dimension, (bool, np.bool_))
        or int(config.maximum_bond_dimension) != config.maximum_bond_dimension
        or int(config.maximum_bond_dimension) <= 0
    ):
        raise ValueError("maximum_bond_dimension must be a positive integer.")
    if config.compression_tolerance <= 0.0:
        raise ValueError("compression_tolerance must be positive.")
    if config.compression_svd_method not in ("standard", "qr", "eig"):
        raise ValueError("compression_svd_method must be 'standard', 'qr', or 'eig'.")
    if config.system_update_coefficient_tolerance < 0.0:
        raise ValueError("system_update_coefficient_tolerance must be non-negative.")

    checkpoint_interval = config.checkpoint_interval_seconds
    if checkpoint_interval is not None and checkpoint_interval <= 0.0:
        raise ValueError("checkpoint_interval_seconds must be positive or None.")
    if checkpoint_interval is not None and output_directory is None:
        raise ValueError("checkpointing requires output_directory.")

    if restart_from is not None and initial_density_matrix is not None:
        raise ValueError("initial_density_matrix and restart_from are mutually exclusive.")
    if restart_from is not None and not restart_from.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {restart_from}")

    if config.simulation_mode == "energy_transfer" and restart_from is None:
        if initial_density_matrix is None:
            raise ValueError("energy_transfer requires initial_density_matrix or restart_from.")
        density_matrix = np.asarray(initial_density_matrix, dtype=complex)
        if density_matrix.shape != (system.N, system.N):
            raise ValueError(f"initial_density_matrix must have shape ({system.N}, {system.N}).")
        if not np.all(np.isfinite(density_matrix)):
            raise ValueError("initial_density_matrix contains non-finite values.")
        if not np.allclose(density_matrix, density_matrix.T.conj(), atol=1.0e-10):
            raise ValueError("initial_density_matrix must be Hermitian.")
        if abs(np.trace(density_matrix)) == 0.0:
            raise ValueError("initial_density_matrix must have nonzero trace.")
    elif config.simulation_mode == "linear_spectra" and initial_density_matrix is not None:
        raise ValueError("initial_density_matrix is not used in linear_spectra mode.")

    if not isinstance(output_identifier, str) or output_identifier.strip() == "":
        raise ValueError("output_identifier must be a non-empty string.")
    if "/" in output_identifier or "\\" in output_identifier:
        raise ValueError("output_identifier must not contain path separators.")
    if config.execution_mode not in ("local", "slurm"):
        raise ValueError("execution_mode must be either 'local' or 'slurm'.")
    if config.ray_max_parallel_system_updates is not None:
        limit = config.ray_max_parallel_system_updates
        if isinstance(limit, (str, bool, np.bool_)) or int(limit) != limit or int(limit) <= 0:
            raise ValueError("ray_max_parallel_system_updates must be None or a positive integer.")


def print_summaries(config, system, pseudomodes, system_index_maps, rho_sys_initial, restart_from, parallelization_summary):
    """
        Print the main setup summaries.
    """
    from .initial_state import print_initial_state_summary
    from .parameters import print_pseudomode_summary
    from .model import print_system_summary
    from .parallelization import print_parallelization_summary

    print_system_summary(system, system_index_maps)
    print()
    print_pseudomode_summary(pseudomodes)
    print()
    print_initial_state_summary(config, system, rho_sys_initial, restart_from)
    print()
    print_parallelization_summary(config, parallelization_summary)
    print()
