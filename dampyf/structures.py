"""
    Core data container used by DAMPyF.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np


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


@dataclass
class LocalPseudomodes:
    """
        Parameters of one site local pseudomode environment.
    """

    frequencies: np.ndarray
    damping_rates: np.ndarray
    couplings: np.ndarray
    thermal_energies: np.ndarray
    fock_dimensions: np.ndarray

    def __post_init__(self):
        self.frequencies = np.asarray(self.frequencies, dtype=float).copy()
        self.damping_rates = np.asarray(self.damping_rates, dtype=float).copy()
        self.couplings = np.asarray(self.couplings, dtype=complex).copy()
        self.thermal_energies = np.asarray(self.thermal_energies, dtype=float).copy()
        fock_dimensions = np.asarray(self.fock_dimensions)

        arrays = (
            self.frequencies,
            self.damping_rates,
            self.couplings,
            self.thermal_energies,
            fock_dimensions,
        )
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("Local pseudomode parameters must be one-dimensional arrays.")
        if len({len(array) for array in arrays}) != 1:
            raise ValueError("All local pseudomode parameter arrays must have the same length.")
        if len(self.frequencies) == 0:
            raise ValueError("A LocalPseudomodes object must contain at least one pseudomode.")
        if not np.all(np.isfinite(self.frequencies)) or np.any(self.frequencies == 0.0):
            raise ValueError("Pseudomode frequencies must be finite and nonzero.")
        if not np.all(np.isfinite(self.damping_rates)) or np.any(self.damping_rates <= 0.0):
            raise ValueError("Pseudomode damping rates must be finite and strictly positive.")
        if not np.all(np.isfinite(self.couplings)):
            raise ValueError("Pseudomode couplings must be finite.")
        if not np.all(np.isfinite(self.thermal_energies)) or np.any(self.thermal_energies < 0.0):
            raise ValueError("Pseudomode thermal energies must be finite and non-negative.")
        if np.any((self.frequencies < 0.0) & (self.thermal_energies > 0.0)):
            raise ValueError("Negative-frequency pseudomodes must have zero thermal energy.")
        if not np.all(np.isfinite(fock_dimensions)) or not np.all(fock_dimensions == np.round(fock_dimensions)):
            raise ValueError("Pseudomode Fock dimensions must be finite integers.")
        if np.any(fock_dimensions <= 0):
            raise ValueError("Pseudomode Fock dimensions must be positive.")
        self.fock_dimensions = fock_dimensions.astype(int)

    @property
    def number_of_modes(self):
        return len(self.frequencies)


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
    same_local_environment: bool

    @property
    def num_osc(self):
        """
            Return the total number of pseudomodes.
        """
        return int(sum(self.Q))


@dataclass
class DampfResult:
    """
        In-memory result and run data returned by run_dampf.
    """

    times: np.ndarray
    rho_system: np.ndarray | None
    optical_coherence: np.ndarray | None
    system_output_file: Path | None
    simulation_data_file: Path | None
    elapsed_time_seconds: float
    final_maximum_bond_dimension: int
    maximum_bond_dimension_reached: int

    @property
    def data(self):
        """
            Return the mode-dependent primary output array.
        """
        if self.rho_system is not None:
            return self.rho_system
        return self.optical_coherence


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
        if not (0 <= m < self.N and 0 <= n < self.N):
            raise IndexError(f"System density indices ({m},{n}) are outside the range 0,...,{self.N - 1}.")
        if m <= n:
            return self.double_to_single_index[m][n]
        return self.double_to_single_index[n][m]


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


