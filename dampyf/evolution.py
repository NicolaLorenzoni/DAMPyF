"""
    Time evolution routines for DAMPF.
"""

import numpy as np
import ray

from .mps_operations import apply_mpo

SYSTEM_UPDATE_RESOURCE = "system_update_slot"


def compressed_linear_combination(terms, zero_template, compression_tolerance, maximum_bond_dimension, compression_svd_method):
    """
        Return a compressed linear combination of MPS components.
    """
    result = None

    for coeff, density_mps in terms:
        if abs(coeff) == 0.0:
            continue
        if isinstance(density_mps, ray.ObjectRef):
            density_mps = ray.get(density_mps)
        if result is None:
            result = coeff * density_mps.copy()
        else:
            result += coeff * density_mps
        result.compress(relerr=compression_tolerance, rank=maximum_bond_dimension, svd_method=compression_svd_method)

    if result is None:
        return 0.0 * zero_template.copy()

    return result


@ray.remote(num_cpus=1)
def mps_update_worker(density_mps, update_mpo, normalization=1.0, conjugate_result=False, return_bond_dimension=False):
    """
        Apply one MPO to one MPS component.
    """
    if abs(normalization) == 0.0:
        raise ValueError("Cannot propagate with zero normalization.")

    updated_mps = apply_mpo(update_mpo, density_mps)

    if normalization != 1.0:
        updated_mps = updated_mps / normalization

    if conjugate_result:
        updated_mps = updated_mps.conj()

    if return_bond_dimension:
        return updated_mps, max(updated_mps.bond_dimensions, default=1)

    return updated_mps


@ray.remote(num_cpus=1)
def system_update_worker(
    row_index,
    state_matrix,
    Us,
    compression_tolerance,
    maximum_bond_dimension,
    compression_svd_method,
    system_update_coefficient_tolerance,
    simulation_mode,
):
    """
        Compute one system-update row.
    """
    if simulation_mode == "energy_transfer":
        full_mps_ref_matrix = state_matrix
        zero_template = full_mps_ref_matrix[0][0]
        if isinstance(zero_template, ray.ObjectRef):
            zero_template = ray.get(zero_template)

        left_mixed_row = []
        m = row_index

        for n in range(Us.shape[0]):
            terms = []
            for k in range(Us.shape[0]):
                coeff = Us[m, k]
                if abs(coeff) > system_update_coefficient_tolerance:
                    terms.append((coeff, full_mps_ref_matrix[k][n]))
            left_mixed_row.append(compressed_linear_combination(terms, zero_template, compression_tolerance, maximum_bond_dimension, compression_svd_method))

        updated_row = []

        for n in range(m, Us.shape[0]):
            terms = []
            for k in range(Us.shape[0]):
                coeff = np.conjugate(Us[n, k])
                if abs(coeff) > system_update_coefficient_tolerance:
                    terms.append((coeff, left_mixed_row[k]))
            updated_row.append(compressed_linear_combination(terms, zero_template, compression_tolerance, maximum_bond_dimension, compression_svd_method))

    else:
        state_ref_row = state_matrix
        zero_template = state_ref_row[0]
        if isinstance(zero_template, ray.ObjectRef):
            zero_template = ray.get(zero_template)

        updated_row = []

        for n in range(Us.shape[0]):
            terms = []
            for k in range(Us.shape[0]):
                coeff = Us[k, n]
                if abs(coeff) > system_update_coefficient_tolerance:
                    terms.append((coeff, state_ref_row[k]))
            updated_row.append(compressed_linear_combination(terms, zero_template, compression_tolerance, maximum_bond_dimension, compression_svd_method))

    if len(updated_row) == 1:
        return updated_row[0]

    return updated_row


def prepare_full_mps_ref_matrix(state_refs, system_index_maps, adjoint_mpo_ref):
    """
        Return the full system MPS reference matrix.
    """
    full_mps_ref_matrix = [[None for _ in range(system_index_maps.N)] for _ in range(system_index_maps.N)]

    for ind, (m, n) in enumerate(system_index_maps.single_to_double_indices):
        full_mps_ref_matrix[m][n] = state_refs[ind]

    for m in range(system_index_maps.N):
        for n in range(m):
            ind = system_index_maps.pair_to_stored_index(n, m)
            full_mps_ref_matrix[m][n] = mps_update_worker.remote(state_refs[ind], adjoint_mpo_ref, 1.0, True)

    return full_mps_ref_matrix


def launch_system_updates(state_matrix, operator_refs, config, system_index_maps):
    """
        Launch the system update workers.
    """
    if config.simulation_mode == "energy_transfer":
        updated_state_refs = [None for _ in range(system_index_maps.num_stored_indices)]

        for m in range(system_index_maps.N):
            row_length = system_index_maps.N - m
            worker = system_update_worker.options(num_returns=row_length, resources={SYSTEM_UPDATE_RESOURCE: 1})
            row_refs = worker.remote(
                m,
                state_matrix,
                operator_refs.Us,
                config.compression_tolerance,
                config.maximum_bond_dimension,
                config.compression_svd_method,
                config.system_update_coefficient_tolerance,
                config.simulation_mode,
            )
            if row_length == 1:
                row_refs = [row_refs]
            else:
                row_refs = list(row_refs)

            for n in range(m, system_index_maps.N):
                ind = system_index_maps.pair_to_stored_index(m, n)
                updated_state_refs[ind] = row_refs[n - m]

        return updated_state_refs

    updated_state_refs = []

    for initial_index in range(system_index_maps.N):
        worker = system_update_worker.options(num_returns=system_index_maps.N, resources={SYSTEM_UPDATE_RESOURCE: 1})
        row_refs = worker.remote(
            initial_index,
            state_matrix[initial_index],
            operator_refs.Us,
            config.compression_tolerance,
            config.maximum_bond_dimension,
            config.compression_svd_method,
            config.system_update_coefficient_tolerance,
            config.simulation_mode,
        )
        if system_index_maps.N == 1:
            row_refs = [row_refs]
        else:
            row_refs = list(row_refs)
        updated_state_refs.append(row_refs)

    return updated_state_refs


def system_update(state_refs, system_index_maps, operator_refs, config):
    """
        Apply the system update.
    """
    if config.simulation_mode == "energy_transfer":
        state_matrix = prepare_full_mps_ref_matrix(state_refs, system_index_maps, operator_refs.adjoint_mpo)
    else:
        state_matrix = state_refs

    return launch_system_updates(state_matrix, operator_refs, config, system_index_maps)


def evolve(state_refs, config, operator_refs, system_index_maps, normalization=1.0):
    """
        Propagate the state until the next output time.
    """
    current_normalization = normalization
    data_timesteps = int(round(config.data_time_step / config.time_step))
    bond_dimension_refs = []

    for _ in range(data_timesteps):
        if config.simulation_mode == "energy_transfer":
            for ind in range(system_index_maps.num_stored_indices):
                state_refs[ind], bond_dimension_ref = mps_update_worker.options(num_returns=2).remote(
                    state_refs[ind], operator_refs.local_update_mpo_refs_1[ind], current_normalization, False, True
                )
                bond_dimension_refs.append(bond_dimension_ref)
            current_normalization = 1.0
        else:
            for initial_index in range(system_index_maps.N):
                for current_index in range(system_index_maps.N):
                    worker = mps_update_worker.options(num_returns=2)
                    state_refs[initial_index][current_index], bond_dimension_ref = worker.remote(
                        state_refs[initial_index][current_index],
                        operator_refs.local_update_mpo_refs_1[current_index],
                        1.0,
                        False,
                        True,
                    )
                    bond_dimension_refs.append(bond_dimension_ref)

        state_refs = system_update(state_refs, system_index_maps, operator_refs, config)

        if config.trotter_order == "second":
            if config.simulation_mode == "energy_transfer":
                for ind in range(system_index_maps.num_stored_indices):
                    state_refs[ind], bond_dimension_ref = mps_update_worker.options(num_returns=2).remote(
                        state_refs[ind], operator_refs.local_update_mpo_refs_2[ind], 1.0, False, True
                    )
                    bond_dimension_refs.append(bond_dimension_ref)
            else:
                for initial_index in range(system_index_maps.N):
                    for current_index in range(system_index_maps.N):
                        worker = mps_update_worker.options(num_returns=2)
                        state_refs[initial_index][current_index], bond_dimension_ref = worker.remote(
                            state_refs[initial_index][current_index],
                            operator_refs.local_update_mpo_refs_2[current_index],
                            1.0,
                            False,
                            True,
                        )
                        bond_dimension_refs.append(bond_dimension_ref)

    maximum_bond_dimension_reached = max(ray.get(bond_dimension_refs), default=1)
    return state_refs, maximum_bond_dimension_reached
