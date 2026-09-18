"""
    Data and checkpoint storage routines for DAMPyF.
"""

from pathlib import Path

import numpy as np
import ray

from . import mps_operations as mp


def ensure_output_folder(output_directory):
    """
        Create and return the output directory
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def output_filename(prefix, output_directory, output_identifier, extension):
    """
        Return an output filename containing the user-defined identifier
    """
    return Path(output_directory) / f"{prefix}_{output_identifier}.{extension}"


def save_system_data(times, system_data_list, config, output_directory, output_identifier):
    """
        Save the time axis and reduced-system or optical-coherence output.
    """
    ensure_output_folder(output_directory)
    times = np.asarray(times, dtype=float)
    system_data = np.asarray(system_data_list, dtype=complex)

    if config.simulation_mode == "energy_transfer":
        output_file = output_filename("rho_system", output_directory, output_identifier, "npz")
        np.savez(output_file, times=times, rho_system=system_data)
    else:
        output_file = output_filename("optical_coherence", output_directory, output_identifier, "npz")
        np.savez(output_file, times=times, optical_coherence=system_data)
    return output_file


def _array_to_string(array):
    return np.array2string(np.asarray(array), precision=12, suppress_small=False)


def _write_pseudomode_parameters(file_handle, pseudomodes):
    file_handle.write("pseudomode parameters\n")
    file_handle.write(f"same_local_environment={pseudomodes.same_local_environment}\n")
    file_handle.write("columns: Omega_osc drate coupling thermal_energy fock_dim\n")

    for site in range(pseudomodes.N):
        file_handle.write(f"site {site}\n")
        if pseudomodes.Q[site] == 0:
            file_handle.write("None\n")
            continue
        for mode in range(pseudomodes.Q[site]):
            omega = float(pseudomodes.Omega_osc[site][mode])
            damping_rate = float(pseudomodes.drate[site][mode])
            coupling = complex(pseudomodes.coupling[site][mode])
            coupling_string = f"{coupling.real:.12g}{coupling.imag:+.12g}j"
            thermal_energy = float(pseudomodes.thermal_energy[site][mode])
            fock_dimension = int(pseudomodes.fock_dim[site][mode])
            file_handle.write(f"{omega:.12g} {damping_rate:.12g} {coupling_string} " f"{thermal_energy:.12g} {fock_dimension}\n")


@ray.remote(num_cpus=1)
def _maximum_bond_dimension_worker(density_mps):
    """
        Return the maximum bond dimension of one MPS.
    """
    return max(density_mps.bond_dimensions, default=1)


def maximum_state_bond_dimension(state_refs, config):
    """
        Return the maximum bond dimension in the current MPS collection.
    """
    if config.simulation_mode == "energy_transfer":
        density_mps_refs = list(state_refs)
    else:
        density_mps_refs = [density_mps_ref for state_row in state_refs for density_mps_ref in state_row]
    values = ray.get([_maximum_bond_dimension_worker.remote(ref) for ref in density_mps_refs])
    return max(values, default=1)


def save_simulation_data(
    config,
    system,
    pseudomodes,
    initial_density_matrix,
    restart_from,
    system_output_file,
    elapsed_time_seconds,
    final_maximum_bond_dimension,
    maximum_bond_dimension_reached,
    output_directory,
    output_identifier):
    """
        Save a readable record of one simulation and return its path.
    """
    ensure_output_folder(output_directory)
    output_file = output_filename("simulation_data", output_directory, output_identifier, "txt")

    with open(output_file, "w") as file_handle:
        file_handle.write(f"output_file={None if system_output_file is None else Path(system_output_file).name}\n")
        file_handle.write(f"simulation_mode={config.simulation_mode}\n")
        file_handle.write(f"time={config.time}\n")
        file_handle.write(f"time_step={config.time_step}\n")
        file_handle.write(f"data_time_step={config.data_time_step}\n")
        file_handle.write(f"trotter_order={config.trotter_order}\n")
        file_handle.write(f"maximum_bond_dimension={config.maximum_bond_dimension}\n")
        file_handle.write(f"compression_tolerance={config.compression_tolerance}\n")
        file_handle.write(f"compression_svd_method={config.compression_svd_method}\n")
        file_handle.write(f"system_update_coefficient_tolerance={config.system_update_coefficient_tolerance}\n")
        file_handle.write(f"checkpoint_interval_seconds={config.checkpoint_interval_seconds}\n")
        file_handle.write(f"execution_mode={config.execution_mode}\n")
        file_handle.write(f"ray_max_parallel_system_updates={config.ray_max_parallel_system_updates}\n")
        file_handle.write(f"elapsed_time_seconds={elapsed_time_seconds:.6f}\n")
        file_handle.write(f"final_maximum_bond_dimension={final_maximum_bond_dimension}\n")
        file_handle.write(f"maximum_bond_dimension_reached={maximum_bond_dimension_reached}\n")
        file_handle.write(f"restart_from={restart_from}\n")
        file_handle.write("\ninitial_system_density_matrix=\n")
        initial_density_string = "None" if initial_density_matrix is None else _array_to_string(initial_density_matrix)
        file_handle.write(initial_density_string + "\n")
        file_handle.write("\nsystem_hamiltonian=\n")
        file_handle.write(_array_to_string(system.Hs) + "\n")
        file_handle.write("electric_dipoles=\n")
        electric_dipoles_string = ("None" if system.electric_dipoles is None else _array_to_string(system.electric_dipoles))
        file_handle.write(electric_dipoles_string + "\n")
        file_handle.write("magnetic_dipoles=\n")
        magnetic_dipoles_string = ("None" if system.magnetic_dipoles is None else _array_to_string(system.magnetic_dipoles))
        file_handle.write(magnetic_dipoles_string + "\n")
        file_handle.write("\n")
        _write_pseudomode_parameters(file_handle, pseudomodes)
    return output_file


def save_checkpoint(state_refs, step_index, config, output_directory, output_identifier):
    """
        Save the current MPS state in the selected output directory.
    """
    ensure_output_folder(output_directory)
    checkpoint_folder = Path(output_directory) / f"checkpoint_{output_identifier}_step_{step_index:08d}"
    checkpoint_folder.mkdir(parents=True, exist_ok=True)

    if config.simulation_mode == "energy_transfer":
        for index, density_mps_ref in enumerate(state_refs):
            ray.get(density_mps_ref).save(checkpoint_folder / f"state_{index:05d}.hdf5")
    else:
        for initial_index, state_row in enumerate(state_refs):
            for current_index, density_mps_ref in enumerate(state_row):
                ray.get(density_mps_ref).save(checkpoint_folder / f"state_{initial_index:05d}_{current_index:05d}.hdf5")
    return checkpoint_folder


def _expected_checkpoint_physical_dimensions(pseudomodes):
    dimensions = []
    for site in range(pseudomodes.N):
        for mode in range(pseudomodes.Q[site]):
            dimensions.append((int(pseudomodes.fock_dim[site][mode]) ** 2,))
    return tuple(dimensions)


def _load_checkpoint_mps(checkpoint_file, expected_physical_dimensions):
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Missing checkpoint state file: {checkpoint_file}")
    density_mps = mp.MatrixProductObject.load(checkpoint_file)
    if density_mps.local_physical_dimensions != expected_physical_dimensions:
        raise ValueError(
            f"Checkpoint state {checkpoint_file} has local physical dimensions "
            f"{density_mps.local_physical_dimensions}, but the current pseudomode configuration requires "
            f"{expected_physical_dimensions}."
        )
    return ray.put(density_mps)


def load_checkpoint(checkpoint_folder, config, pseudomodes, system_index_maps):
    """Load all MPS components from an explicitly supplied checkpoint path."""
    checkpoint_folder = Path(checkpoint_folder)
    if not checkpoint_folder.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_folder}")
    expected_dimensions = _expected_checkpoint_physical_dimensions(pseudomodes)

    if config.simulation_mode == "energy_transfer":
        return [
            _load_checkpoint_mps(checkpoint_folder / f"state_{index:05d}.hdf5", expected_dimensions)
            for index in range(system_index_maps.num_stored_indices)
        ]

    state_refs = []
    for initial_index in range(system_index_maps.N):
        state_refs.append(
            [
                _load_checkpoint_mps(
                    checkpoint_folder / f"state_{initial_index:05d}_{current_index:05d}.hdf5",
                    expected_dimensions,
                )
                for current_index in range(system_index_maps.N)
            ]
        )
    return state_refs
