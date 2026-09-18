"""
Simple plot script for DAMPF system dynamics.

Model arrays are read from the example's run_simulation.py.
Run this script from any folder. It reads data from the output_data folder.
"""

from pathlib import Path
from runpy import run_path

import matplotlib.pyplot as plt
import numpy as np

from dampyf import units

script_folder = Path(__file__).resolve().parent

# =============================================================================
# User options
# =============================================================================
plot_basis = "eigenstate"  # Options: "site", "eigenstate"

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
        "text.usetex": False,
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

    with np.load(file_path, allow_pickle=False) as data:
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


def transform_to_eigenstate_basis(rho_data, hamiltonian):
    """
    Transform all density matrices from the site basis to the eigenbasis.
    """
    eigenstate_energies, eigenstate_vectors = np.linalg.eigh(hamiltonian)
    transformed_data = np.zeros_like(rho_data, dtype=complex)

    for time_index in range(rho_data.shape[0]):
        transformed_data[time_index] = eigenstate_vectors.conj().T @ rho_data[time_index] @ eigenstate_vectors

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

    if basis_label == "s":
        ax.set_title(rf"$\langle {index + 1}|\rho_s(t)|{index + 1}\rangle$")
    else:
        ax.set_title(rf"$\langle \epsilon_{{{index + 1}}}|\rho_s(t)|" rf"\epsilon_{{{index + 1}}}\rangle$")

    ax.set_xlabel("Time [fs]")
    ax.grid(True)


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

    component_label = r"\mathrm{Re}" if component == "real" else r"\mathrm{Im}"

    if basis_label == "s":
        matrix_element = rf"\langle {row + 1}|\rho_s(t)|{col + 1}\rangle"
    else:
        matrix_element = rf"\langle \epsilon_{{{row + 1}}}|\rho_s(t)|" rf"\epsilon_{{{col + 1}}}\rangle"

    ax.set_title(rf"${component_label}\left\{{{matrix_element}\right\}}$")
    ax.set_xlabel("Time [fs]")
    ax.grid(True)


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
    output_file,
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

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    fig.savefig(output_file, bbox_inches="tight")


def main(show_plots=True):
    """Compare the saved simulation with the reference selected in run_simulation.py."""
    model = run_path(str(script_folder.parent / "run_simulation.py"))
    output_folder = model["output_directory"]
    reference_file = model["reference_file"]
    times, rho_data = load_density_matrix_data(output_folder, f"rho_system_{model['output_identifier']}.npz")
    reference_times, reference_data = load_density_matrix_data(reference_file.parent, reference_file.name)
    check_density_data_consistency(rho_data, reference_data)

    if plot_basis == "eigenstate":
        hamiltonian = np.asarray(model["hamiltonian"], dtype=complex)
        rho_data, energies = transform_to_eigenstate_basis(rho_data, hamiltonian)
        reference_data, _ = transform_to_eigenstate_basis(reference_data, hamiltonian)
        print("Eigenstate energies [cm^-1]:", energies)
        basis_label = "e"
    elif plot_basis == "site":
        basis_label = "s"
    else:
        raise ValueError("plot_basis must be 'site' or 'eigenstate'.")

    num_states = rho_data.shape[1]
    plot_system_dynamics(
        times / units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME,
        rho_data,
        reference_times / units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME,
        reference_data,
        build_population_index_list(num_states, population_indices),
        build_coherence_pair_list(num_states, coherence_pairs),
        basis_label,
        f"Simulation: BD = {model['config'].maximum_bond_dimension}",
        f"Reference: BD = {model['reference_bond_dimension']}",
        output_folder / "system_dynamics.pdf",
    )
    print("Saved comparison:", output_folder / "system_dynamics.pdf")
    if show_plots:
        plt.show()


if __name__ == "__main__":
    main()
