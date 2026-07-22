"""
    Generate an system Hamiltonian input file for DAMPyF.
    Only modify the section marked "User input".
"""

from pathlib import Path

import numpy as np


# =============================================================================
# User input
# =============================================================================

N = 2


# Output file name.
hamiltonian_output_filename = "dimer_system_hamiltonian.txt"


# Site energies.
site_energies = np.zeros(N, dtype=float)

site_energies[0] = 1000.0
site_energies[1] = 1200.0


# Optional global energy shift.
energy_shift = 0.0


# System couplings. Only fill the upper triangular part, i < j.
system_couplings = np.zeros((N, N), dtype=float)

system_couplings[0, 1] = 100.0





# =============================================================================
# End of user input
# Do not modify below this line
# =============================================================================

script_folder = Path(__file__).resolve().parent
project_folder = script_folder.parent
output_folder = project_folder / "input_data" / "system"
output_folder.mkdir(parents=True, exist_ok=True)

hamiltonian_output_file = output_folder / hamiltonian_output_filename


# Consistency checks.
if site_energies.shape != (N,):
    raise ValueError("site_energies must have shape (N,).")

if system_couplings.shape != (N, N):
    raise ValueError("system_couplings must have shape (N, N).")

if not np.allclose(np.diag(system_couplings), 0.0, atol=1.0e-14):
    raise ValueError("The diagonal of system_couplings must be zero. Use site_energies for diagonal terms.")

if not np.allclose(np.tril(system_couplings), 0.0, atol=1.0e-14):
    raise ValueError("Only the upper triangular part of system_couplings should be filled.")


# Symmetrize the coupling matrix.
system_couplings = system_couplings + system_couplings.T


# Build the full system Hamiltonian.
Hs = np.diag(site_energies + energy_shift) + system_couplings

if Hs.shape != (N, N):
    raise ValueError("System Hamiltonian has inconsistent shape.")

if not np.allclose(Hs, Hs.T.conj(), atol=1.0e-10):
    raise ValueError("System Hamiltonian is not Hermitian/symmetric.")


hamiltonian_header = "system Hamiltonian in site basis"
np.savetxt(hamiltonian_output_file, Hs, fmt="% .10f", header=hamiltonian_header)

print(f"Wrote {hamiltonian_output_file.name} in {output_folder}")
print(f"Number of system sites: {N}")
