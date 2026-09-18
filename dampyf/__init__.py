"""Public interface for the DAMPyF package."""

from . import unit_conventions as units
from .structures import DampfResult, LocalPseudomodes

from .api import run_dampf
from .configuration import DampfConfig
from .helpers import (
    build_magnetic_dipoles,
    build_system_hamiltonian,
    eigenstate_density_matrix,
    estimate_fock_dimensions,
    estimate_required_memory,
    polarized_density_matrix,
    site_density_matrix,
)

__version__ = "1.0.0"

__all__ = [
    "DampfConfig",
    "DampfResult",
    "LocalPseudomodes",
    "build_magnetic_dipoles",
    "build_system_hamiltonian",
    "eigenstate_density_matrix",
    "estimate_fock_dimensions",
    "estimate_required_memory",
    "polarized_density_matrix",
    "run_dampf",
    "site_density_matrix",
    "units",
    "__version__",
]
