"""
Data storage routines for DAMPF.
"""

from pathlib import Path
import numpy as np
import ray

from . import dampf_mps_operations as mp


OUTPUT_FOLDER = Path("output_data")


def ensure_output_folder():
    """
        Create the output folder if needed.
    """
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    return


def output_filename(prefix, params, extension):
    """
        Return an output filename containing the user-defined identifier.
    """
    return OUTPUT_FOLDER / f"{prefix}_{params.output_identifier}.{extension}"


def save_system_data(times, system_data_list, params):
    """
        Save the system output data and the corresponding time axis in one file.
    """
    ensure_output_folder()

    times = np.asarray(times, dtype=float)
    system_data = np.asarray(system_data_list, dtype=complex)

    if params.simulation_mode == "energy_transfer":
        output_file = output_filename("rho_system", params, "npz")
        np.savez(output_file, times=times, rho_system=system_data)
    else:
        output_file = output_filename("optical_coherence", params, "npz")
        np.savez(output_file, times=times, optical_coherence=system_data)

    return output_file


def _array_to_string(array):
    """
        Convert an array to a readable string for text output.
    """
    return np.array2string(np.asarray(array), precision=12, suppress_small=False)


def _write_pseudomode_parameters(file_handle, pseudomodes):
    """
        Write pseudomode parameters in five-column format.
    """
    file_handle.write("pseudomode parameters\n")
    file_handle.write(f"same_local_environment={pseudomodes.same_local_environment}\n")
    file_handle.write("columns: Omega_osc drate coupling thermal_energy fock_dim\n")

    for n in range(pseudomodes.N):
        file_handle.write(f"site {n}\n")

        if pseudomodes.Q[n] == 0:
            file_handle.write("None\n")
            continue

        for q in range(pseudomodes.Q[n]):
            omega = float(pseudomodes.Omega_osc[n][q])
            drate = float(pseudomodes.drate[n][q])
            coupling = complex(pseudomodes.coupling[n][q])
            coupling_string = f"{coupling.real:.12g}{coupling.imag:+.12g}j"
            thermal_energy = float(pseudomodes.thermal_energy[n][q])
            fock_dim = int(pseudomodes.fock_dim[n][q])
            file_handle.write(f"{omega:.12g} {drate:.12g} {coupling_string} {thermal_energy:.12g} {fock_dim}\n")

    return


@ray.remote(num_cpus=1)
def _maximum_bond_dimension_worker(density_mps):
    """
        Return the maximum bond dimension of one MPS.
    """
    return max(density_mps.bond_dimensions, default=1)


def maximum_state_bond_dimension(state_refs, params):
    """
        Return the maximum bond dimension in the current MPS collection.
    """
    if params.simulation_mode == "energy_transfer":
        density_mps_refs = list(state_refs)
    else:
        density_mps_refs = [density_mps_ref for state_row in state_refs for density_mps_ref in state_row]

    bond_dimensions = ray.get([_maximum_bond_dimension_worker.remote(density_mps_ref) for density_mps_ref in density_mps_refs])
    return max(bond_dimensions, default=1)


def save_simulation_data(
    params,
    system,
    pseudomodes,
    rho_sys_initial,
    system_output_file,
    elapsed_time_seconds,
    final_maximum_bond_dimension,
    maximum_bond_dimension_reached,
):
    """
        Save a text file with simulation input data and configuration parameters.
    """
    ensure_output_folder()

    output_file = output_filename("simulation_data", params, "txt")

    with open(output_file, "w") as file_handle:
        file_handle.write(f"output file name: {Path(system_output_file).name}\n")
        file_handle.write(f"time={params.time}\n")
        file_handle.write(f"dt={params.dt}\n")
        file_handle.write(f"dtdata={params.dtdata}\n")
        file_handle.write(f"trotter_order={params.trotter_order}\n")
        file_handle.write(f"BD={params.BD}\n")
        file_handle.write(f"elapsed_time_seconds={elapsed_time_seconds:.6f}\n")
        file_handle.write(f"final_maximum_bond_dimension={final_maximum_bond_dimension}\n")
        file_handle.write(f"maximum_bond_dimension_reached={maximum_bond_dimension_reached}\n")
        file_handle.write(f"initial_state_type={params.initial_state_type}\n")
        if params.initial_state_type == "backup":
            file_handle.write(f"backup_identifier={params.backup_identifier}\n")

        file_handle.write("\n")
        file_handle.write("initial state\n")
        file_handle.write(f"initial_state_type={params.initial_state_type}\n")
        if params.initial_state_type == "backup":
            file_handle.write(f"backup_identifier={params.backup_identifier}\n")
        file_handle.write("initial_system_density_matrix=\n")
        if rho_sys_initial is None:
            file_handle.write("None\n")
        else:
            file_handle.write(_array_to_string(rho_sys_initial))
            file_handle.write("\n")

        file_handle.write("\n")
        file_handle.write("system parameters\n")
        file_handle.write("system_hamiltonian=\n")
        file_handle.write(_array_to_string(system.Hs))
        file_handle.write("\n")
        file_handle.write("electric_dipoles=\n")
        if system.electric_dipoles is None:
            file_handle.write("None\n")
        else:
            file_handle.write(_array_to_string(system.electric_dipoles))
            file_handle.write("\n")
        file_handle.write("magnetic_dipoles=\n")
        if system.magnetic_dipoles is None:
            file_handle.write("None\n")
        else:
            file_handle.write(_array_to_string(system.magnetic_dipoles))
            file_handle.write("\n")

        file_handle.write("\n")
        _write_pseudomode_parameters(file_handle, pseudomodes)

    return


def save_checkpoint(state_refs, step_index, params):
    """
        Save the current MPS state.
    """
    ensure_output_folder()

    checkpoint_folder = OUTPUT_FOLDER / f"checkpoint_{params.output_identifier}_step_{step_index:08d}"
    checkpoint_folder.mkdir(parents=True, exist_ok=True)

    if params.simulation_mode == "energy_transfer":
        for ind, density_mps_ref in enumerate(state_refs):
            density_mps = ray.get(density_mps_ref)
            density_mps.save(str(checkpoint_folder / f"state_{ind:05d}.hdf5"))
            del density_mps

    else:
        for initial_index, state_row in enumerate(state_refs):
            for current_index, density_mps_ref in enumerate(state_row):
                density_mps = ray.get(density_mps_ref)
                density_mps.save(str(checkpoint_folder / f"state_{initial_index:05d}_{current_index:05d}.hdf5"))
                del density_mps

    return


def _expected_checkpoint_physical_dimensions(pseudomodes):
    """
        Return the expected local physical dimensions of a saved DAMPF MPS.
    """
    dimensions = []
    for n in range(pseudomodes.N):
        for q in range(pseudomodes.Q[n]):
            dimensions.append((int(pseudomodes.fock_dim[n][q]) ** 2,))
    return tuple(dimensions)


def _load_checkpoint_mps(checkpoint_file, expected_physical_dimensions):
    """
        Load one checkpoint MPS, validate its local dimensions, and place it in Ray.
    """
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Missing checkpoint state file: {checkpoint_file}")

    density_mps = mp.MatrixProductObject.load(checkpoint_file)

    if density_mps.local_physical_dimensions != expected_physical_dimensions:
        raise ValueError(
            f"Checkpoint state {checkpoint_file} has local physical dimensions "
            f"{density_mps.local_physical_dimensions}, but the current pseudomode "
            f"configuration requires {expected_physical_dimensions}."
        )

    return ray.put(density_mps)


def load_checkpoint(params, pseudomodes, system_index_maps):
    """
        Load all MPS components from the checkpoint selected by backup_identifier.

        The loaded state is used as a new initial state. The propagation time axis
        therefore starts again from zero.
    """
    checkpoint_folder = OUTPUT_FOLDER / params.backup_identifier

    if not checkpoint_folder.is_dir():
        raise FileNotFoundError(f"Checkpoint folder not found: {checkpoint_folder}")

    expected_dimensions = _expected_checkpoint_physical_dimensions(pseudomodes)

    if params.simulation_mode == "energy_transfer":
        state_refs = []
        for ind in range(system_index_maps.num_stored_indices):
            checkpoint_file = checkpoint_folder / f"state_{ind:05d}.hdf5"
            state_refs.append(_load_checkpoint_mps(checkpoint_file, expected_dimensions))
        return state_refs

    state_refs = []
    for initial_index in range(system_index_maps.N):
        state_row = []
        for current_index in range(system_index_maps.N):
            checkpoint_file = checkpoint_folder / f"state_{initial_index:05d}_{current_index:05d}.hdf5"
            state_row.append(_load_checkpoint_mps(checkpoint_file, expected_dimensions))
        state_refs.append(state_row)

    return state_refs

