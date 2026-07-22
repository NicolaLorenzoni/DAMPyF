"""
Small generic helper functions for DAMPF.
"""

import numpy as np



def boltzmann_occupation(pseudomodes, n, q):
    """
        Return the thermal occupation of the input pseudomode.
    """
    thermal_energy = pseudomodes.thermal_energy[n][q]
    if thermal_energy <= 0.0:
        return 0.0
    exponent = pseudomodes.Omega_osc[n][q] / thermal_energy
    return 0.0 if exponent > 700.0 else 1.0 / np.expm1(exponent)


def check_user_parameters(params, system, system_index_maps):
    """
        Consistency checks for the user-editable simulation parameters.
    """
    # Consistency check.
    if params.simulation_mode not in ("energy_transfer", "linear_spectra"):
        raise ValueError("simulation_mode must be either 'energy_transfer' or 'linear_spectra'.")

    # Consistency check.
    if params.time < 0.0:
        raise ValueError("time must be non-negative.")

    # Consistency check.
    if params.dt <= 0.0:
        raise ValueError("dt must be positive.")

    # Consistency check.
    if params.dtdata <= 0.0:
        raise ValueError("dtdata must be positive.")

    # Consistency check.
    if abs(params.dtdata / params.dt - round(params.dtdata / params.dt)) > 1.0e-12:
        raise ValueError("dtdata must be a positive integer multiple of dt.")

    # Consistency check.
    if abs(params.time / params.dtdata - round(params.time / params.dtdata)) > 1.0e-12:
        raise ValueError("time must be an integer multiple of dtdata.")

    # Consistency check.
    if params.trotter_order not in ("first", "second"):
        raise ValueError("trotter_order must be either 'first' or 'second'.")

    # Consistency check.
    if int(params.BD) != params.BD or int(params.BD) <= 0:
        raise ValueError("BD must be a positive integer.")

    # Consistency check.
    if params.compression_tol <= 0.0:
        raise ValueError("compression_tol must be positive.")

    # Consistency check.
    if params.compression_svd_method not in ("standard", "qr", "eig"):
        raise ValueError("compression_svd_method must be 'standard', 'qr', or 'eig'.")

    # Consistency check.
    if params.system_update_coeff_tol < 0.0:
        raise ValueError("system_update_coeff_tol must be non-negative.")

    # Consistency check.
    if params.backuptime_seconds <= 0.0:
        raise ValueError("backuptime_seconds must be positive.")

    # Consistency check.
    if not isinstance(params.same_local_environment, bool):
        raise ValueError("same_local_environment must be True or False.")

    # Consistency check.
    if params.initial_state_type is None:
        if params.simulation_mode != "linear_spectra":
            raise ValueError(
                "initial_state_type may be None only when simulation_mode is 'linear_spectra'."
            )
    elif params.initial_state_type not in ("site", "eigenstate", "polarized_pulse", "backup"):
        raise ValueError(
            "initial_state_type must be None, 'site', 'eigenstate', 'polarized_pulse', or 'backup'."
        )

    if params.initial_state_type == "backup":
        # Consistency check.
        if not isinstance(params.backup_identifier, str):
            raise ValueError("backup_identifier must be a string when initial_state_type is 'backup'.")

        # Consistency check.
        if params.backup_identifier.strip() == "":
            raise ValueError("backup_identifier must not be empty when initial_state_type is 'backup'.")

        # Consistency check.
        if "/" in params.backup_identifier or "\\" in params.backup_identifier:
            raise ValueError("backup_identifier must be a checkpoint folder name, not a path.")

    if params.simulation_mode == "energy_transfer":
        if params.initial_state_type == "site":
            # Consistency check.
            if int(params.initial_site) != params.initial_site or not (0 <= int(params.initial_site) < system_index_maps.N):
                raise ValueError(f"initial_site must be an integer between 0 and {system_index_maps.N - 1}.")

        if params.initial_state_type == "eigenstate":
            # Consistency check.
            if int(params.initial_eigenstate) != params.initial_eigenstate or not (0 <= int(params.initial_eigenstate) < system_index_maps.N):
                raise ValueError(f"initial_eigenstate must be an integer between 0 and {system_index_maps.N - 1}.")

        if params.initial_state_type == "polarized_pulse":
            light_polarization = np.asarray(params.light_polarization, dtype=float)

            # Consistency check.
            if light_polarization.shape != (3,):
                raise ValueError("light_polarization must have shape (3,).")

            # Consistency check.
            if np.linalg.norm(light_polarization) == 0.0:
                raise ValueError("light_polarization must have non-zero norm.")

            # Consistency check.
            if getattr(params, "electric_dipoles_file", getattr(params, "dipoles_file", None)) is None:
                raise ValueError("polarized_pulse initial states require electric_dipoles_file.")

            # Consistency check.
            polarization = light_polarization / np.linalg.norm(light_polarization)
            if np.sum(np.abs(system.electric_dipoles @ polarization) ** 2) == 0.0:
                raise ValueError("The pulse polarization is orthogonal to all site dipoles.")


    # Consistency check.
    if not isinstance(params.output_identifier, str):
        raise ValueError("output_identifier must be a string.")

    # Consistency check.
    if params.output_identifier.strip() == "":
        raise ValueError("output_identifier must not be empty.")

    # Consistency check.
    if "/" in params.output_identifier or "\\" in params.output_identifier:
        raise ValueError("output_identifier must not contain path separators.")

    # Consistency check.
    if params.execution_mode not in ("local", "slurm"):
        raise ValueError("execution_mode must be either 'local' or 'slurm'.")

    # Consistency check.
    if params.ray_max_parallel_system_updates is not None:
        if isinstance(params.ray_max_parallel_system_updates, str):
            raise ValueError("ray_max_parallel_system_updates must be None or a positive integer.")
        if int(params.ray_max_parallel_system_updates) != params.ray_max_parallel_system_updates or int(params.ray_max_parallel_system_updates) <= 0:
            raise ValueError("ray_max_parallel_system_updates must be None or a positive integer.")

    return



def print_summaries(params, system, pseudomodes, system_index_maps, operator_refs, rho_sys_initial, parallelization_summary):
    """
        Print the main setup summaries.
    """
    from .dampf_initial_state import print_initial_state_summary
    from .dampf_load_data import print_pseudomode_summary
    from .dampf_model import print_system_summary
    from .dampf_parallelization import print_parallelization_summary

    print_system_summary(system, system_index_maps)
    print()
    print_pseudomode_summary(pseudomodes)
    print()
    print_initial_state_summary(params, system, rho_sys_initial)

    print()
    print_parallelization_summary(params, parallelization_summary)
    print()

    return
