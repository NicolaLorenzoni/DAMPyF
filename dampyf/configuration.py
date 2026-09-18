"""
    Configuration object for DAMPyF simulations.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DampfConfig:
    """
        Numerical and execution settings used by :func:`dampyf.run_dampf`.
    """

    simulation_mode: str
    time: float
    time_step: float
    data_time_step: float
    maximum_bond_dimension: int
    compression_tolerance: float = 1.0e-8
    compression_svd_method: str = "eig"
    system_update_coefficient_tolerance: float = 1.0e-8
    trotter_order: str = "second"
    checkpoint_interval_seconds: float | None = None
    execution_mode: str = "local"
    ray_max_parallel_system_updates: int | None = None
