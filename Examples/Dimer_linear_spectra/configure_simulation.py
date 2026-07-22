"""
    User-editable DAMPF simulation parameters.

    All Hamiltonian parameters, damping rates, couplings, thermal energies,
    and times must be provided in one mutually consistent unit convention.
"""

import numpy as np
from dampf_modules import unit_conventions as units



# =============================================================================
# Simulation parameters
# =============================================================================
simulation_mode = "linear_spectra"  # Options: "energy_transfer", "linear_spectra"

time = 200.0*units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME    # Total simulation time
dt = 1.0*units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME        # Propagation timestep
dtdata = 1.0*dt                                         # Data collection timestep (has to be multiple of dt)
backuptime_seconds = 5*60*60                            # Backup time interval in seconds
trotter_order = "second"                                # Options: "first" (local error O(dt^2)), "second" (local error O(dt^3)).

BD = 15                              # Maximum bond dimension
compression_tol = 1.0e-8             # MPS single-tensor compression tolerance
compression_svd_method = "eig"       # Options: "standard", "qr", "eig". Eig does not properly capture singular values below sqrt(machine precision).
system_update_coeff_tol = 1.0e-8     # System evolution operator coefficients threshold


# =============================================================================
# System
# =============================================================================
system_hamiltonian_file = "dimer_system_hamiltonian.txt"  # Site-basis Hamiltonian
electric_dipoles_file = None   # Set to None if electric-dipole data are unused
magnetic_dipoles_file = None                    # Set to None if magnetic-dipole data are unused
# magnetic_dipoles_file = "dimer_magnetic_dipoles.txt"


# =============================================================================
# Pseudomode parameters
# =============================================================================
same_local_environment = True                         # True if local environments are equal for all sites
pseudomode_parameter_file = "Loc_env.txt"  # Provide text file for the local environment (used when same_local_environment = True)
pseudomode_parameter_files = None  # List containing one filename or None per site when same_local_environment = False


# =============================================================================
# Initial state
# =============================================================================
initial_state_type = None  # None for linear_spectra; options: "site", "eigenstate", "polarized_pulse", "backup"
backup_identifier = None          # Exact checkpoint folder name, e.g. "checkpoint_test_simulation_step_00000042" (used if initial_state_type is "backup")
initial_site = None                # Initially populated site (used if initial_state_type is "site")
initial_eigenstate = None             # Initially populated eigenstate, ordered by increasing energy (used if initial_state_type is "eigenstate")
light_polarization = None  # Pulse polarization (used if initial_state_type is "polarized_pulse")


# =============================================================================
# Output
# =============================================================================
output_identifier = "dimer_test_simulation"  # String identifier added to output filenames


# =============================================================================
# One-node Ray parallelisation
# =============================================================================
execution_mode                  = "local"  # Options: "local", "slurm"
ray_max_parallel_system_updates = None     # None or (advanced Ray option) positive integer.
