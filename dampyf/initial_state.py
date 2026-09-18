"""
    Initial-state construction for DAMPF.
"""

import numpy as np
import ray

from . import mps_operations as mp
from .storage import load_checkpoint


@ray.remote(num_cpus=1)
def pseudomode_thermal_state_worker(pseudomodes, basis_data):
    """
        Build the product thermal pseudomode state in IdTr (identity + traceless matrices) basis.
    """
    local_tensors = []

    for n in range(pseudomodes.N):
        for q in range(pseudomodes.Q[n]):
            dim = int(pseudomodes.fock_dim[n][q])
            dim2 = dim * dim
            fock_vector = np.zeros(dim2, dtype=complex)

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


def initialize_state_refs(config, system, pseudomodes, system_index_maps, basis_data, initial_density_matrix, restart_from):
    """
        Build the initial MPS state directly in the Ray object store.
    """
    if restart_from is not None:
        state_refs = load_checkpoint(restart_from, config, pseudomodes, system_index_maps)
        return state_refs, None

    vib_state_ref = pseudomode_thermal_state_worker.remote(pseudomodes, basis_data)

    if config.simulation_mode == "energy_transfer":
        rho_sys = np.asarray(initial_density_matrix, dtype=complex).copy()
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


def print_initial_state_summary(config, system, rho_sys, restart_from):
    """
        Print the initial-state summary.
    """
    print("Initial-state summary")
    print("---------------------")

    if restart_from is not None:
        print(f"checkpoint         : {restart_from}")
        if config.simulation_mode == "energy_transfer":
            print("state components   : loaded electronic-density MPS components")
        else:
            print("state components   : loaded N x N optical-coherence MPS matrix")
        return

    if config.simulation_mode == "energy_transfer":
        print("initial state      : supplied density matrix")
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
