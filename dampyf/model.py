"""
    Model construction for DAMPF.
"""

import numpy as np
import ray
from scipy.linalg import expm

from . import mps_operations as mp

from .structures import RayEvolutionOperatorRefs
from .utils import boltzmann_occupation


def truncated_matrix(matrix, target_dim, vectorized=False):
    """
        Restrict an extended local map to the target Fock dimension.
    """
    if vectorized:
        extended_dim = int(round(np.sqrt(matrix.shape[0])))

        # Consistency check.
        if extended_dim * extended_dim != matrix.shape[0]:
            raise ValueError("Vectorized map dimension is not a square Fock dimension.")

        reduction = np.zeros((extended_dim, extended_dim), dtype=float)
        for j in range(target_dim):
            reduction[j, j] = 1.0
        reduction = np.kron(reduction, reduction)
    else:
        reduction = np.zeros(matrix.shape, dtype=float)
        for j in range(target_dim):
            reduction[j, j] = 1.0

    zero_rows = np.where(~reduction.any(axis=1))[0]
    zero_columns = np.where(~reduction.any(axis=0))[0]

    reduced = np.delete(matrix, zero_rows, axis=0)
    reduced = np.delete(reduced, zero_columns, axis=1)

    return reduced


def local_change_basis_tensor(map_matrix, basis_site, mode):
    """
        Build a local map tensor in the working basis.
    """
    dim2 = basis_site.shape[0]
    tensor = np.zeros((1, dim2, dim2, 1), dtype=complex)

    # Consistency check.
    if mode not in ("left", "right", "both", "vectorized"):
        raise ValueError("mode must be 'left', 'right', 'both', or 'vectorized'.")

    for k in range(dim2):
        for j in range(dim2):
            if mode == "left":
                tensor[0, k, j, 0] = np.trace(np.conjugate(basis_site[k].T) @ map_matrix @ basis_site[j])
            elif mode == "right":
                tensor[0, k, j, 0] = np.trace(np.conjugate(basis_site[k].T) @ basis_site[j] @ np.conjugate(map_matrix.T))
            elif mode == "both":
                left_map, right_map = map_matrix
                tensor[0, k, j, 0] = np.trace(np.conjugate(basis_site[k].T) @ left_map @ basis_site[j] @ np.conjugate(right_map.T))
            else:
                tensor[0, k, j, 0] = np.transpose(basis_site[k]) @ map_matrix @ basis_site[j]

    return tensor


def build_local_tensors(config, pseudomodes, basis_data):
    """
        Build all one-site tensors used in the propagation.
    """
    time_step = config.time_step

    if config.trotter_order == "first":
        local_dt = time_step
    else:
        local_dt = time_step / 2.0

    site_iden = []
    site_Uv = []
    site_Diss = []
    site_Usv_left = []
    site_Usv_right = []
    site_Usv_both = []

    for n in range(pseudomodes.N):
        site_iden.append([])
        site_Uv.append([])
        site_Diss.append([])
        site_Usv_left.append([])
        site_Usv_right.append([])
        site_Usv_both.append([])

        for q in range(pseudomodes.Q[n]):
            dim = int(pseudomodes.fock_dim[n][q])
            extended_dim = int(3 * dim)
            dim2 = dim**2

            aqc = np.diag(np.sqrt(np.arange(1, extended_dim, dtype=float)), k=-1)
            aqa = np.diag(np.sqrt(np.arange(1, extended_dim, dtype=float)), k=1)
            ident = np.identity(extended_dim)

            Hv_ext = pseudomodes.Omega_osc[n][q] * (aqc @ aqa)
            coupling = pseudomodes.coupling[n][q]
            Hsv_ext = coupling * aqc + np.conjugate(coupling) * aqa
            nbar = boltzmann_occupation(pseudomodes, n, q)

            D_loss = np.kron(aqa, aqa) - 0.5 * (np.kron(aqc @ aqa, ident) + np.kron(ident, aqc @ aqa))
            D_gain = np.kron(aqc, aqc) - 0.5 * (np.kron(aqa @ aqc, ident) + np.kron(ident, aqa @ aqc))
            Diss_ext = pseudomodes.drate[n][q] * (nbar + 1.0) * D_loss + pseudomodes.drate[n][q] * nbar * D_gain

            Uv_ext = expm(-1.0j * Hv_ext * time_step)
            Usv_ext = expm(-1.0j * Hsv_ext * local_dt)
            UDiss_ext = expm(Diss_ext * local_dt)

            Uv = truncated_matrix(Uv_ext, dim, vectorized=False)
            Usv = truncated_matrix(Usv_ext, dim, vectorized=False)
            UDiss = truncated_matrix(UDiss_ext, dim, vectorized=True)

            site_iden[n].append(np.eye(dim2, dtype=complex).reshape(1, dim2, dim2, 1))
            site_Uv[n].append(local_change_basis_tensor((Uv, Uv), basis_data.basis[n][q], mode="both"))
            site_Diss[n].append(local_change_basis_tensor(UDiss, basis_data.vecbasis[n][q], mode="vectorized"))
            site_Usv_left[n].append(local_change_basis_tensor(Usv, basis_data.basis[n][q], mode="left"))
            site_Usv_right[n].append(local_change_basis_tensor(Usv, basis_data.basis[n][q], mode="right"))
            site_Usv_both[n].append(local_change_basis_tensor((Usv, Usv), basis_data.basis[n][q], mode="both"))

    return site_iden, site_Uv, site_Diss, site_Usv_left, site_Usv_right, site_Usv_both


@ray.remote(num_cpus=1)
def build_propagator_mpos(config, pseudomodes, basis_data, system_index_maps):
    """
        Build the local propagation MPOs used in one time step.
    """
    site_iden, site_Uv, site_Diss, site_Usv_left, site_Usv_right, site_Usv_both = build_local_tensors(config, pseudomodes, basis_data)

    Uv_chain = []
    Diss_chain = []
    for site in range(pseudomodes.N):
        Uv_chain += site_Uv[site]
        Diss_chain += site_Diss[site]

    UvMPO = mp.MatrixProductObject(Uv_chain)
    DissMPO = mp.MatrixProductObject(Diss_chain)

    if config.simulation_mode == "energy_transfer":
        num_local_mpos = system_index_maps.num_stored_indices
    else:
        num_local_mpos = pseudomodes.N

    UsvMPO = []

    for ind in range(num_local_mpos):
        Usv_chain = []

        if config.simulation_mode == "energy_transfer":
            m, n = system_index_maps.single_to_double_indices[ind]
            for site in range(pseudomodes.N):
                if m == n and site == m:
                    Usv_chain += site_Usv_both[site]
                elif m != n and site == m:
                    Usv_chain += site_Usv_left[site]
                elif m != n and site == n:
                    Usv_chain += site_Usv_right[site]
                else:
                    Usv_chain += site_iden[site]
        else:
            active_site = ind
            for site in range(pseudomodes.N):
                if site == active_site:
                    Usv_chain += site_Usv_left[site]
                else:
                    Usv_chain += site_iden[site]

        UsvMPO.append(mp.MatrixProductObject(Usv_chain))

    local_update_mpos_1 = []

    if config.trotter_order == "first":
        local_update_mpos_2 = None
    else:
        local_update_mpos_2 = []

    for ind in range(num_local_mpos):
        local_step = mp.apply_mpo(UsvMPO[ind], DissMPO)
        local_update_mpos_1.append(mp.apply_mpo(UvMPO, local_step))

        if config.trotter_order == "second":
            local_update_mpos_2.append(mp.apply_mpo(DissMPO, UsvMPO[ind]))

    if local_update_mpos_2 is None:
        if len(local_update_mpos_1) == 1:
            return local_update_mpos_1[0]
        return local_update_mpos_1

    local_mpos = list(local_update_mpos_1) + list(local_update_mpos_2)

    if len(local_mpos) == 1:
        return local_mpos[0]

    return local_mpos


@ray.remote(num_cpus=1)
def build_adjoint_mpo(pseudomodes):
    """
        Build the MPO implementing the vibrational adjoint map.
    """
    local_tensors = []

    for n in range(pseudomodes.N):
        for q in range(pseudomodes.Q[n]):
            dim = int(pseudomodes.fock_dim[n][q])
            tensor = np.zeros((1, dim * dim, dim * dim, 1), dtype=complex)

            for i in range(dim):
                tensor[0, i, i, 0] = 1.0

            for i in range(dim, dim * dim):
                if ((i - dim) % 2) == 0:
                    tensor[0, i, i, 0] = 1.0
                else:
                    tensor[0, i, i, 0] = -1.0

            local_tensors.append(tensor)

    return mp.MatrixProductObject(local_tensors)


def prepare_ray_operator_refs(config, system, pseudomodes, basis_data, system_index_maps):
    """
        Build fixed propagation objects directly in the Ray object store.
    """
    # Number of relevant local evolution mpos
    if config.simulation_mode == "energy_transfer":
        num_local_mpos = system_index_maps.num_stored_indices
    else:
        num_local_mpos = system.N

    if config.trotter_order == "first":
        num_local_mpo_returns = num_local_mpos
    else:
        num_local_mpo_returns = 2 * num_local_mpos

    # Build mpo as Ray object references
    local_mpo_refs = build_propagator_mpos.options(num_returns=num_local_mpo_returns).remote(config, pseudomodes, basis_data, system_index_maps)
    if num_local_mpo_returns == 1:
        local_mpo_refs = [local_mpo_refs]
    else:
        local_mpo_refs = list(local_mpo_refs)

    if config.trotter_order == "first":
        local_update_mpo_refs_1 = local_mpo_refs
        local_update_mpo_refs_2 = None
    else:
        local_update_mpo_refs_1 = local_mpo_refs[:num_local_mpos]
        local_update_mpo_refs_2 = local_mpo_refs[num_local_mpos:]

    return RayEvolutionOperatorRefs(
        Us=ray.put(expm(-1.0j * system.Hs * config.time_step)),
        adjoint_mpo=build_adjoint_mpo.remote(pseudomodes),
        local_update_mpo_refs_1=local_update_mpo_refs_1,
        local_update_mpo_refs_2=local_update_mpo_refs_2,
    )


def print_system_summary(system, system_index_maps):
    """
        Print a summary of the prepared system.
    """
    print("System summary")
    print("-------------------------")
    print(f"N                        : {system.N}")
    print(f"Hs shape                 : {system.Hs.shape}")
    print(f"electric dipoles loaded  : {system.electric_dipoles is not None}")
    print(f"magnetic dipoles loaded  : {system.magnetic_dipoles is not None}")
    print(f"eigenstate energies         : {system.eig_sys}")
    print(f"stored system indices: {system_index_maps.num_stored_indices}")
    preview_length = min(10, system_index_maps.num_stored_indices)
    print(f"first stored indices    : {system_index_maps.single_to_double_indices[:preview_length]}")

    return
