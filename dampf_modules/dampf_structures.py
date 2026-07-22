"""
Core data structures used by DAMPF.

This module contains simple containers, system-index maps, and small
helpers for creating lists of MPS/MPO objects.
"""

from dataclasses import dataclass
from typing import Any
import numpy as np

from . import dampf_mps_operations as mp


@dataclass
class SystemParameters:
    """
        System parameters dataclass.
    """

    N: int
    Hs: np.ndarray
    eig_sys: np.ndarray
    eig_transf_matrix: np.ndarray
    electric_dipoles: np.ndarray | None
    magnetic_dipoles: np.ndarray | None

    @property
    def dipoles(self):
        """
            Backward-compatible alias for electric transition dipoles.
        """
        return self.electric_dipoles


@dataclass
class PseudomodeParameters:
    """
        Pseudomode parameters dataclass.
    """

    N: int
    Q: list
    fock_dim: list
    Omega_osc: list
    coupling: list
    drate: list
    thermal_energy: list
    source_files: list
    same_local_environment: bool

    @property
    def num_osc(self):
        """
        Return the total number of pseudomodes.
        """
        return int(sum(self.Q))


@dataclass
class SystemIndexMaps:
    """
        Store maps between system density indices and MPS-list indices.
        Here, only the upper triangular part is considered, due to hermiticity.
    """

    N: int
    single_to_double_indices: list
    double_to_single_index: list

    @property
    def num_stored_indices(self):
        """
            Return the number of stored system indices.
        """
        return len(self.single_to_double_indices)

    def pair_to_stored_index(self, m, n):
        """
            Return the stored-list index corresponding to rho[m,n] or rho[n,m].
        """
        self._check_indices(m, n)
        if m <= n:
            return self.double_to_single_index[m][n]
        return self.double_to_single_index[n][m]

    def _check_indices(self, m, n):
        """
            Check that system density indices are inside the allowed range.
        """
        # Consistency check.
        if not (0 <= m < self.N and 0 <= n < self.N):
            raise IndexError(f"System density indices ({m},{n}) are outside the range 0,...,{self.N - 1}.")

        return


@dataclass
class BasisData:
    """
        Local pseudomode operator bases and basis transformation.
    """

    basis: list
    vecbasis: list
    fock_to_idtr: list



@dataclass
class RayEvolutionOperatorRefs:
    """
        Ray object-store references for evolution operators.
    """

    Us: Any = None
    adjoint_mpo: Any = None
    local_update_mpo_refs_1: Any = None
    local_update_mpo_refs_2: Any = None


def generate_system_index_maps(N):
    """
        Generate maps for the diagonal and upper-triangular system entries considered in simulations (to exploit hermiticity).
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

    return SystemIndexMaps(N=N, single_to_double_indices=single_to_double_indices, double_to_single_index=double_to_single_index)


def mps_list(pseudomodes, system_index_maps=None, rank=1, length=None):
    """
        Create a list of zero density-operator MPS components.
    """
    if length is None:
        if system_index_maps is None:
            length = 1
        else:
            length = system_index_maps.num_stored_indices

    dims = []
    for n in range(pseudomodes.N):
        for q in range(pseudomodes.Q[n]):
            dim = int(pseudomodes.fock_dim[n][q]) ** 2
            dims.append((dim,))

    prototype = mp.create_zero_matrix_product_object(sites=pseudomodes.num_osc, ldim=dims, rank=rank)
    return [prototype.copy() for _ in range(int(length))]


def mpo_list(pseudomodes, system_index_maps=None, rank=1, length=None):
    """
        Create a list of zero MPO objects.
    """
    if length is None:
        if system_index_maps is None:
            length = 1
        else:
            length = system_index_maps.num_stored_indices

    dims = []
    for n in range(pseudomodes.N):
        for q in range(pseudomodes.Q[n]):
            dim = int(pseudomodes.fock_dim[n][q]) ** 2
            dims.append((dim, dim))

    prototype = mp.create_zero_matrix_product_object(sites=pseudomodes.num_osc, ldim=dims, rank=rank)
    return [prototype.copy() for _ in range(int(length))]
