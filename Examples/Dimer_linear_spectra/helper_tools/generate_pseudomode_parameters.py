"""
    Generate a local pseudomode-parameter input file for DAMPyF.\
    Only modify the section marked "User input".
"""

from pathlib import Path

import numpy as np


# =============================================================================
# User input
# =============================================================================

Q = 5


# Output file name.
pseudomode_output_filename = "Loc_env.txt"


# Pseudomode frequencies. Values may be positive or negative, but not zero.
# Negative-frequency pseudomodes must have zero thermal energy.
pseudomode_frequencies = np.zeros(Q, dtype=float)

pseudomode_frequencies[0] = 400.0
pseudomode_frequencies[1] = 500.0
pseudomode_frequencies[2] = 600.0
pseudomode_frequencies[3] = 700.0
pseudomode_frequencies[4] = 800.0


# Pseudomode damping rates.
pseudomode_damping_rates = np.zeros(Q, dtype=float)

pseudomode_damping_rates[0] = 10.0
pseudomode_damping_rates[1] = 20.0
pseudomode_damping_rates[2] = 15.0
pseudomode_damping_rates[3] = 30.0
pseudomode_damping_rates[4] = 10.0


# Pseudomode couplings.
pseudomode_couplings = np.zeros(Q, dtype=complex)

pseudomode_couplings[0] = 20.0
pseudomode_couplings[1] = 40.0
pseudomode_couplings[2] = 70.0
pseudomode_couplings[3] = 70.0
pseudomode_couplings[4] = 50.0


# Pseudomode thermal energies, in the same units as the frequencies.
# Use 0.0 for zero-temperature and all negative-frequency pseudomodes.
pseudomode_thermal_energies = np.zeros(Q, dtype=float)

pseudomode_thermal_energies[0] = 20.0
pseudomode_thermal_energies[1] = 20.0
pseudomode_thermal_energies[2] = 20.0
pseudomode_thermal_energies[3] = 20.0
pseudomode_thermal_energies[4] = 20.0


# Local Fock-space dimensions.
pseudomode_fock_dimensions = np.zeros(Q, dtype=float)

pseudomode_fock_dimensions[0] = 4
pseudomode_fock_dimensions[1] = 4
pseudomode_fock_dimensions[2] = 4
pseudomode_fock_dimensions[3] = 4
pseudomode_fock_dimensions[4] = 4






# =============================================================================
# End of user input
# Do not modify below this line
# =============================================================================

script_folder = Path(__file__).resolve().parent
project_folder = script_folder.parent
output_folder = project_folder / "input_data" / "pseudomodes"
output_folder.mkdir(parents=True, exist_ok=True)

if Path(pseudomode_output_filename).name != pseudomode_output_filename:
    raise ValueError("pseudomode_output_filename should be a file name, not a path.")

pseudomode_output_file = output_folder / pseudomode_output_filename


# Consistency checks.
if Q <= 0:
    raise ValueError("Q must be positive.")

if pseudomode_frequencies.shape != (Q,):
    raise ValueError("pseudomode_frequencies must have shape (Q,).")

if pseudomode_damping_rates.shape != (Q,):
    raise ValueError("pseudomode_damping_rates must have shape (Q,).")

if pseudomode_couplings.shape != (Q,):
    raise ValueError("pseudomode_couplings must have shape (Q,).")

if pseudomode_thermal_energies.shape != (Q,):
    raise ValueError("pseudomode_thermal_energies must have shape (Q,).")

if pseudomode_fock_dimensions.shape != (Q,):
    raise ValueError("pseudomode_fock_dimensions must have shape (Q,).")

if not np.all(np.isfinite(pseudomode_frequencies)):
    raise ValueError("All pseudomode frequencies must be finite.")

if np.any(pseudomode_frequencies == 0.0):
    raise ValueError("All pseudomode frequencies must be nonzero.")

if not np.all(np.isfinite(pseudomode_damping_rates)):
    raise ValueError("All pseudomode damping rates must be finite.")

if np.any(pseudomode_damping_rates <= 0.0):
    raise ValueError("All pseudomode damping rates must be strictly positive.")

if not np.all(np.isfinite(pseudomode_couplings)):
    raise ValueError("All couplings must be finite.")

if not np.all(np.isfinite(pseudomode_thermal_energies)):
    raise ValueError("All pseudomode thermal energies must be finite.")

if np.any(pseudomode_thermal_energies < 0.0):
    raise ValueError("All pseudomode thermal energies must be non-negative.")

if np.any((pseudomode_frequencies < 0.0) & (pseudomode_thermal_energies > 0.0)):
    raise ValueError("Negative-frequency pseudomodes must have zero thermal energy.")

if not np.all(np.isfinite(pseudomode_fock_dimensions)):
    raise ValueError("All Fock-space dimensions must be finite.")

if np.any(pseudomode_fock_dimensions <= 0):
    raise ValueError("All Fock-space dimensions must be positive.")

if not np.allclose(pseudomode_fock_dimensions, np.round(pseudomode_fock_dimensions)):
    raise ValueError("All Fock-space dimensions must be integers.")


# Save pseudomode parameters.
pseudomode_header = "frequency damping_rate coupling thermal_energy fock_dim"
with open(pseudomode_output_file, "w") as file_handle:
    file_handle.write(f"# {pseudomode_header}\n")

    for q in range(Q):
        coupling = complex(pseudomode_couplings[q])
        coupling_string = f"{coupling.real:.10g}{coupling.imag:+.10g}j"
        file_handle.write(
            f"{pseudomode_frequencies[q]:10.4f} "
            f"{pseudomode_damping_rates[q]:10.4f} "
            f"{coupling_string:>24} "
            f"{pseudomode_thermal_energies[q]:10.4f} "
            f"{int(pseudomode_fock_dimensions[q]):5d}\n"
        )

print(f"Wrote {pseudomode_output_file.name} in {output_folder}")
print(f"Number of local pseudomodes: {Q}")
