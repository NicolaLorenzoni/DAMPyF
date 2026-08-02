"""
Generate electric- and magnetic-dipole input files for DAMPyF.

The magnetic dipoles can either be specified directly or constructed from
electric dipoles and site positions.

Only modify the section marked "User input".
"""

from pathlib import Path

import numpy as np


# =============================================================================
# User input
# =============================================================================

N = 2


# Output file names.
electric_dipoles_output_filename = "dimer_electric_dipoles.txt"
magnetic_dipoles_output_filename = "dimer_magnetic_dipoles.txt"


# Electric transition dipoles.
electric_dipoles = np.zeros((N, 3), dtype=complex)

electric_dipoles[0] = [1.0, 0.0, 0.0]
electric_dipoles[1] = [0.0, 1.0, 0.0]


# Method used to define the magnetic transition dipoles.
# Options: "direct", "from_positions"
"""
  "direct": In this case, the user specifies directly to components
  "from_positions": In this case, the user specifies site position and the magnetic dipoles are
                    computed as magnetic_dipoles[n] = cross(site_positions[n], electric_dipoles[n]).
"""
magnetic_dipole_method = "direct"


# Magnetic transition dipoles used when magnetic_dipole_method = "direct".
magnetic_dipoles = np.zeros((N, 3), dtype=complex)

magnetic_dipoles[0] = [0.0, 0.0, 1.0]
magnetic_dipoles[1] = [0.0, 0.0, -1.0]


# Site positions used when magnetic_dipole_method = "from_positions".

site_positions = np.zeros((N, 3), dtype=float)

site_positions[0] = [0.0, -1.0, 0.0]
site_positions[1] = [-1.0, 0.0, 0.0]





# =============================================================================
# End of user input
# Do not modify below this line
# =============================================================================

script_folder = Path(__file__).resolve().parent
project_folder = script_folder.parent
output_folder = project_folder / "input_data" / "system"
output_folder.mkdir(parents=True, exist_ok=True)

electric_dipoles_output_file = output_folder / electric_dipoles_output_filename
magnetic_dipoles_output_file = output_folder / magnetic_dipoles_output_filename


def format_dipole_component(value):
    """
        Format one real or complex dipole component.
    """
    value = complex(value)

    if value.imag == 0.0:
        return f"{value.real:.10f}"

    return f"{value.real:.10f}{value.imag:+.10f}j"


def write_dipole_file(output_file, dipoles, header):
    """
        Write a three-column real or complex dipole file.
    """
    with open(output_file, "w") as file_handle:
        file_handle.write(f"# {header}\n")

        for dipole in dipoles:
            components = [format_dipole_component(component) for component in dipole]
            file_handle.write(" ".join(components) + "\n")

    return


# Consistency checks.
if electric_dipoles.shape != (N, 3):
    raise ValueError("electric_dipoles must have shape (N, 3).")

if magnetic_dipole_method not in ("direct", "from_positions"):
    raise ValueError('magnetic_dipole_method must be "direct" or "from_positions".')

if magnetic_dipole_method == "direct":
    if magnetic_dipoles.shape != (N, 3):
        raise ValueError("magnetic_dipoles must have shape (N, 3).")
else:
    if site_positions.shape != (N, 3):
        raise ValueError("site_positions must have shape (N, 3).")

    magnetic_dipoles = np.cross(site_positions, electric_dipoles)


if not np.all(np.isfinite(electric_dipoles)):
    raise ValueError("electric_dipoles contains non-finite values.")

if not np.all(np.isfinite(magnetic_dipoles)):
    raise ValueError("magnetic_dipoles contains non-finite values.")


# Write the dipole files.
electric_header = "x y z components of electric transition dipoles"
magnetic_header = "x y z components of magnetic transition dipoles"

write_dipole_file(electric_dipoles_output_file, electric_dipoles, electric_header)
write_dipole_file(magnetic_dipoles_output_file, magnetic_dipoles, magnetic_header)

print(f"Wrote {electric_dipoles_output_file.name} in {output_folder}")
print(f"Wrote {magnetic_dipoles_output_file.name} in {output_folder}")
print(f"Number of system sites: {N}")
print(f"Magnetic-dipole method: {magnetic_dipole_method}")
