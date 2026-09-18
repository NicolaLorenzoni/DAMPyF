"""
    Observable and reduced-density-matrix routines for DAMPF.
"""

import numpy as np
import ray


@ray.remote(num_cpus=1)
def density_trace_worker(density_mps, factor):
    """
        Trace one density component in a Ray task.
    """
    prod = np.array([1.0 + 0.0j])
    for site in range(len(density_mps)):
        prod = prod @ density_mps.local_tensors[site][:, 0, :]

    return factor * prod[0]


def reduced_system_density_matrix(state, pseudomodes, system_index_maps):
    """
        Return the reduced system density matrix.
    """
    rho_sys = np.zeros((system_index_maps.N, system_index_maps.N), dtype=complex)
    factor = 1.0
    task_refs = []

    for n in range(pseudomodes.N):
        for q in range(pseudomodes.Q[n]):
            factor *= np.sqrt(pseudomodes.fock_dim[n][q])

    for ind in range(system_index_maps.num_stored_indices):
        task_refs.append(density_trace_worker.remote(state[ind], factor))

    values = ray.get(task_refs)
    for value, (m, n) in zip(values, system_index_maps.single_to_double_indices):
        rho_sys[m, n] = value
        if m < n:
            rho_sys[n, m] = np.conjugate(value)

    return rho_sys


def optical_coherence_matrix(state_refs, pseudomodes):
    """
        Return the optical-coherence matrix for linear-spectra mode.
    """
    optical_coherence = np.zeros((pseudomodes.N, pseudomodes.N), dtype=complex)
    factor = 1.0
    task_refs = []

    for n in range(pseudomodes.N):
        for q in range(pseudomodes.Q[n]):
            factor *= np.sqrt(pseudomodes.fock_dim[n][q])

    for initial_index in range(pseudomodes.N):
        for current_index in range(pseudomodes.N):
            task_refs.append(density_trace_worker.remote(state_refs[initial_index][current_index], factor))

    values = ray.get(task_refs)
    counter = 0
    for initial_index in range(pseudomodes.N):
        for current_index in range(pseudomodes.N):
            optical_coherence[initial_index, current_index] = values[counter]
            counter += 1

    return optical_coherence
