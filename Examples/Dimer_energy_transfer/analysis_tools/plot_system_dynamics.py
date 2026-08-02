"""
Simple plot script for DAMPF system dynamics.

Run this script from any folder. It reads data from the output_data folder.
"""

from pathlib import Path
import sys

script_folder = Path(__file__).resolve().parent
project_folder = script_folder.parent
sys.path.insert(0, str(project_folder))

from dampf_modules import unit_conventions as units

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# User options
# =============================================================================
plot_basis = "site"  # Options: "site", "eigenstate"
rho_file = "rho_system_dimer_test_simulation.npz"
rho_file_2 = "rho_system_dimer_test_simulation_2.npz"

label_1 = "BD=5,  dt=1.0 fs"
label_2 = "BD=15, dt=0.5 fs"

# Needed only if plot_basis = "eigenstate".
system_hamiltonian_file = "dimer_system_hamiltonian.txt"

# None to plot all populations, or a list of indices.
population_indices = None

# None to plot all i < j coherences, or a list of (i, j) tuples.
coherence_pairs = None

# Global figure settings. All dimensions are in centimetres.
one_column_figure_width_cm = 7.5
two_column_figure_width_cm = 15.0
figure_row_height_cm = 5.0

# Global font settings.
axis_label_size = 9
tick_label_size = 8
legend_font_size = 8






# =============================================================================
# End of user options
# =============================================================================

# =============================================================================
# Plot settings
# =============================================================================

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": tick_label_size,
        "axes.titlesize": axis_label_size,
        "axes.labelsize": axis_label_size,
        "xtick.labelsize": tick_label_size,
        "ytick.labelsize": tick_label_size,
        "legend.fontsize": legend_font_size,
        "legend.framealpha": 1.0,
        "legend.facecolor": "white",
        "legend.edgecolor": "black",
        "legend.fancybox": False,
    }
)


# =============================================================================
# Helper functions
# =============================================================================
def load_density_matrix_data(output_folder, filename):
    """
    Load the reduced system density-matrix dynamics and its time axis.
    """
    if filename is None:
        raise ValueError("rho_file must be specified explicitly.")

    file_path = output_folder / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}")

    data = np.load(file_path)
    if "times" not in data or "rho_system" not in data:
        raise ValueError("rho_file must contain arrays named 'times' and 'rho_system'.")

    times = data["times"]
    rho_data = data["rho_system"]

    if rho_data.ndim != 3:
        raise ValueError("rho_system must have shape (time, system_index, system_index).")

    if rho_data.shape[1] != rho_data.shape[2]:
        raise ValueError("The last two axes of rho_system must have equal length.")

    if len(times) != rho_data.shape[0]:
        raise ValueError("times and rho_system have incompatible lengths.")

    print(f"Loaded {file_path}")
    return times, rho_data


def load_site_hamiltonian(system_input_folder, filename):
    """
    Load the site-basis system Hamiltonian.
    """
    file_path = system_input_folder / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}")

    return np.loadtxt(file_path, dtype=float)


def transform_to_eigenstate_basis(rho_data, hamiltonian):
    """
    Transform all density matrices from the site basis to the eigenbasis.
    """
    eigenstate_energies, eigenstate_vectors = np.linalg.eigh(hamiltonian)
    transformed_data = np.zeros_like(rho_data, dtype=complex)

    for time_index in range(rho_data.shape[0]):
        transformed_data[time_index] = (
            eigenstate_vectors.conj().T @ rho_data[time_index] @ eigenstate_vectors
        )

    return transformed_data, eigenstate_energies


def build_population_index_list(num_states, selected_indices):
    """
    Build the list of population indices to plot.
    """
    if selected_indices is None:
        return list(range(num_states))

    return selected_indices


def build_coherence_pair_list(num_states, selected_pairs):
    """
    Build the list of coherence pairs to plot.
    """
    if selected_pairs is None:
        pairs = []
        for row in range(num_states):
            for col in range(row + 1, num_states):
                pairs.append((row, col))
        return pairs

    return selected_pairs


def check_density_data_consistency(rho_data, rho_data_2):
    """
    Check that the density-matrix files describe systems of the same size.
    """
    if rho_data_2 is not None and rho_data_2.shape[1:] != rho_data.shape[1:]:
        raise ValueError("The two density-matrix files must have the same system dimensions.")


def plot_population_panel(
    ax,
    times,
    rho_data,
    times_2,
    rho_data_2,
    index,
    basis_label,
    label_1,
    label_2,
):
    """
    Compare one population for the two datasets.
    """
    population = np.real(rho_data[:, index, index])
    ax.plot(times, population, color="tab:blue", label=label_1)

    if rho_data_2 is not None:
        population_2 = np.real(rho_data_2[:, index, index])
        ax.plot(times_2, population_2, "--", color="tab:orange", label=label_2)

\
    if basis_label == "s":
        ax.set_title(fr"$\langle {index + 1}|\rho_s(t)|{index + 1}\rangle$")
    else:
        ax.set_title(
            fr"$\langle \epsilon_{{{index + 1}}}|\rho_s(t)|"
            fr"\epsilon_{{{index + 1}}}\rangle$"
        )

    ax.set_xlabel("Time [fs]")
    ax.grid(True)
    ax.legend()


def plot_coherence_panel(
    ax,
    times,
    rho_data,
    times_2,
    rho_data_2,
    row,
    col,
    component,
    basis_label,
    label_1,
    label_2,
):
    """
    Compare one real or imaginary coherence component for the two datasets.
    """
    coherence = rho_data[:, row, col]
    coherence_component = np.real(coherence) if component == "real" else np.imag(coherence)
    ax.plot(times, coherence_component, color="tab:blue", label=label_1)

    if rho_data_2 is not None:
        coherence_2 = rho_data_2[:, row, col]
        component_2 = np.real(coherence_2) if component == "real" else np.imag(coherence_2)
        ax.plot(times_2, component_2, "--", color="tab:orange", label=label_2)

\
    component_label = r"\mathrm{Re}" if component == "real" else r"\mathrm{Im}"

    if basis_label == "s":
        matrix_element = fr"\langle {row + 1}|\rho_s(t)|{col + 1}\rangle"
    else:
        matrix_element = (
            fr"\langle \epsilon_{{{row + 1}}}|\rho_s(t)|"
            fr"\epsilon_{{{col + 1}}}\rangle"
        )

    ax.set_title(fr"${component_label}\left\{{{matrix_element}\right\}}$")
    ax.set_xlabel("Time [fs]")
    ax.grid(True)
    ax.legend()


def plot_system_dynamics(
    times,
    rho_data,
    times_2,
    rho_data_2,
    population_indices,
    coherence_pairs,
    basis_label,
    label_1,
    label_2,
):
    """
    Plot populations first, followed by paired real and imaginary coherences.
    """
    number_of_population_rows = int(np.ceil(len(population_indices) / 2))
    number_of_coherence_rows = len(coherence_pairs)
    number_of_rows = number_of_population_rows + number_of_coherence_rows

    if number_of_rows == 0:
        raise ValueError("At least one population or coherence must be selected.")

    figure_size = (
        two_column_figure_width_cm / 2.54,
        figure_row_height_cm * number_of_rows / 2.54,
    )
    fig, axes = plt.subplots(number_of_rows, 2, figsize=figure_size)
    axes = np.atleast_2d(axes)

    for population_index, index in enumerate(population_indices):
        row_index = population_index // 2
        column_index = population_index % 2

        plot_population_panel(
            axes[row_index, column_index],
            times,
            rho_data,
            times_2,
            rho_data_2,
            index,
            basis_label,
            label_1,
            label_2,
        )

    if len(population_indices) % 2 == 1:
        axes[number_of_population_rows - 1, 1].axis("off")

    coherence_start_row = number_of_population_rows

    for coherence_index, (row, col) in enumerate(coherence_pairs):
        row_index = coherence_start_row + coherence_index

        plot_coherence_panel(
            axes[row_index, 0],
            times,
            rho_data,
            times_2,
            rho_data_2,
            row,
            col,
            "real",
            basis_label,
            label_1,
            label_2,
        )

        plot_coherence_panel(
            axes[row_index, 1],
            times,
            rho_data,
            times_2,
            rho_data_2,
            row,
            col,
            "imag",
            basis_label,
            label_1,
            label_2,
        )

    time_min = np.min(times)
    time_max = np.max(times)

    if times_2 is not None:
        time_min = min(time_min, np.min(times_2))
        time_max = max(time_max, np.max(times_2))

    for ax in axes.ravel():
        if ax.axison:
            ax.set_xlim(time_min, time_max)

    fig.tight_layout()
    fig.savefig("system_dynamics.pdf", bbox_inches="tight")


# =============================================================================
# Main script
# =============================================================================
project_folder = script_folder.parent
output_folder = project_folder / "output_data"
system_input_folder = project_folder / "input_data" / "system"

times, rho_data = load_density_matrix_data(output_folder, rho_file)

times_2 = None
rho_data_2 = None
if rho_file_2 is not None:
    times_2, rho_data_2 = load_density_matrix_data(output_folder, rho_file_2)

check_density_data_consistency(rho_data, rho_data_2)

if plot_basis == "site":
    basis_label = "s"
elif plot_basis == "eigenstate":
    basis_label = "e"
    site_hamiltonian = load_site_hamiltonian(system_input_folder, system_hamiltonian_file)
    rho_data, eigenstate_energies = transform_to_eigenstate_basis(rho_data, site_hamiltonian)

    if rho_data_2 is not None:
        rho_data_2, _ = transform_to_eigenstate_basis(rho_data_2, site_hamiltonian)

    print("Eigenstate energies:")
    print(eigenstate_energies)
else:
    raise ValueError("plot_basis must be either 'site' or 'eigenstate'.")

num_states = rho_data.shape[1]
selected_population_indices = build_population_index_list(num_states, population_indices)
selected_coherence_pairs = build_coherence_pair_list(num_states, coherence_pairs)

plot_system_dynamics(
    times/units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME,
    rho_data,
    times_2/units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME,
    rho_data_2,
    selected_population_indices,
    selected_coherence_pairs,
    basis_label,
    label_1,
    label_2,
)

plt.show()
