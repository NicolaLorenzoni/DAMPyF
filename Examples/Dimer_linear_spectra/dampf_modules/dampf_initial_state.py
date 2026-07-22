"""
Initial-state construction for DAMPF.
"""

import numpy as np
import ray

from . import dampf_mps_operations as mp
from .dampf_storage import load_checkpoint


@ray.remote(num_cpus=1)
def pseudomode_thermal_state_worker(pseudomodes, basis_data):
    """
        Build the product thermal pseudomode state in IdTr basis.
    """
    if basis_data is None or basis_data.fock_to_idtr is None:
        raise ValueError("Pseudomode thermal states require the Fock-to-IdTr transformation.")

    local_tensors = []

    for n in range(pseudomodes.N):
        for q in range(pseudomodes.Q[n]):
            dim = int(pseudomodes.fock_dim[n][q])
            dim2 = dim * dim
            fock_vector = np.zeros(dim2, dtype=complex)

            if pseudomodes.thermal_energy[n][q] < 0.0:
                raise ValueError("Pseudomode thermal energy must be non-negative.")

            if pseudomodes.thermal_energy[n][q] <= 0.0:
                fock_vector[0] = 1.0
            else:
                beta = 1.0 / pseudomodes.thermal_energy[n][q]
                partition = 0.0

                for k in range(100 * dim):
                    partition += np.exp(-beta * pseudomodes.Omega_osc[n][q] * k)

                for k in range(dim):
                    fock_vector[dim * k + k] = np.exp(-beta * pseudomodes.Omega_osc[n][q] * k) / partition

            tensor = np.zeros((1, dim2, 1), dtype=complex)
            tensor[0, :, 0] = fock_vector @ basis_data.fock_to_idtr[n][q]
            local_tensors.append(tensor)

    return mp.MatrixProductObject(local_tensors)


@ray.remote(num_cpus=1)
def initial_state_component_worker(vib_state, coeff):
    """
        Build one initial MPS component.
    """
    return coeff * vib_state.copy()


def system_initial_density(params, system):
    """
        Return the initial system density matrix in the site basis.
    """
    if params.initial_state_type == "site":
        rho = np.zeros((system.N, system.N), dtype=complex)
        rho[int(params.initial_site), int(params.initial_site)] = 1.0
        return rho

    if params.initial_state_type == "eigenstate":
        amplitudes = system.eig_transf_matrix[:, int(params.initial_eigenstate)]
        return np.outer(amplitudes, np.conjugate(amplitudes))

    if params.initial_state_type == "polarized_pulse":
        polarization = params.light_polarization / np.linalg.norm(params.light_polarization)
        amplitudes = system.electric_dipoles @ polarization
        weight = np.sum(np.abs(amplitudes) ** 2)

        amplitudes = amplitudes / np.sqrt(weight)
        return np.outer(amplitudes, np.conjugate(amplitudes))

    raise RuntimeError("Unknown initial_state_type after parameter validation.")


def initialize_state_refs(params, system, pseudomodes, system_index_maps, basis_data):
    """
        Build the initial MPS state directly in the Ray object store.
    """
    if params.initial_state_type == "backup":
        state_refs = load_checkpoint(params, pseudomodes, system_index_maps)
        return state_refs, None

    vib_state_ref = pseudomode_thermal_state_worker.remote(pseudomodes, basis_data)

    if params.simulation_mode == "energy_transfer":
        rho_sys = system_initial_density(params, system)
        state_refs = []

        for m, n in system_index_maps.single_to_double_indices:
            state_refs.append(initial_state_component_worker.remote(vib_state_ref, rho_sys[m, n]))

        return state_refs, rho_sys

    zero_state_ref = initial_state_component_worker.remote(vib_state_ref, 0.0)
    state_refs = []

    for initial_index in range(system.N):
        state_row = []
        for current_index in range(system.N):
            if current_index == initial_index:
                state_row.append(vib_state_ref)
            else:
                state_row.append(zero_state_ref)
        state_refs.append(state_row)

    return state_refs, None


def print_initial_state_summary(params, system, rho_sys):
    """
        Print the initial-state summary.
    """
    print("Initial-state summary")
    print("---------------------")

    if params.initial_state_type == "backup":
        print("initial_state_type : backup")
        print(f"backup_identifier  : {params.backup_identifier}")
        if params.simulation_mode == "energy_transfer":
            print("state components   : loaded electronic-density MPS components")
        else:
            print("state components   : loaded N x N optical-coherence MPS matrix")
        return

    if params.simulation_mode == "energy_transfer":
        print(f"initial_state_type : {params.initial_state_type}")
        print("rho_sys components  :")

        for m in range(rho_sys.shape[0]):
            for n in range(rho_sys.shape[1]):
                if abs(rho_sys[m, n]) > 0.0:
                    print(f"    rho_sys[{m},{n}] = {rho_sys[m, n]}")

    else:
        print("simulation_mode    : linear_spectra")
        print("state components   : N x N optical-coherence MPS matrix")
        print(f"initial coherences : diagonal entries for {system.N} excited states")

    return
