"""
Simple post-processing script for DAMPF absorption spectra.

Model arrays are read from the example's run_simulation.py.
Run this script from any folder. It reads optical-coherence data from this
output_data folder, computes the dipole correlation, applies an optional
Gaussian time filter, and computes a one-sided Fourier transform.
"""

from pathlib import Path
from runpy import run_path

import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt

from dampyf import units

# =============================================================================
# User options
# =============================================================================
frequency_shift = 0.0  # Added to the Fourier frequency axis
frequency_min = 0.0  # Minimum Fourier frequency before frequency_shift
frequency_max = 2000.0  # Maximum Fourier frequency before frequency_shift
frequency_points = 6000

interpolation_factor = 5  # 1 means no interpolation; 5 gives five times more time points
apply_gaussian_filter = True
gaussian_sigma = 50.0 * units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME
normalize_spectrum = True

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

one_column_figure_size = (
    one_column_figure_width_cm / 2.54,
    figure_row_height_cm / 2.54,
)

two_column_figure_size = (
    two_column_figure_width_cm / 2.54,
    figure_row_height_cm / 2.54,
)


# =============================================================================
# Helper functions
# =============================================================================
def load_optical_coherence_data(output_folder, filename):
    """
    Load optical-coherence dynamics and its time axis.
    """
    if filename is None:
        raise ValueError("optical_coherence_file must be specified explicitly.")

    file_path = output_folder / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}")

    with np.load(file_path, allow_pickle=False) as data:
        times = data["times"]
        optical_coherence = data["optical_coherence"]

    if optical_coherence.ndim != 3:
        raise ValueError("optical_coherence must have shape (time, initial_index, final_index).")
    if optical_coherence.shape[1] != optical_coherence.shape[2]:
        raise ValueError("The last two axes of optical_coherence must have equal length.")
    if len(times) != optical_coherence.shape[0]:
        raise ValueError("times and optical_coherence have incompatible lengths.")

    print(f"Loaded {file_path}")
    return times, optical_coherence


def check_optical_coherence_consistency(optical_coherence, optical_coherence_2):
    """
    Check that the optical-coherence files describe systems of the same size.
    """
    if optical_coherence_2 is not None:
        if optical_coherence_2.shape[1:] != optical_coherence.shape[1:]:
            raise ValueError("The two optical-coherence files must have the same system dimensions.")

    return


def compute_dipole_correlation(optical_coherence, dipoles):
    """
    Compute the dipole correlation from optical-coherence dynamics.

    The isotropic rotational average is
    C(t) = (1/3) sum_{i,j} dot(conj(mu_i), mu_j) optical_coherence[t,i,j].
    """
    num_times = optical_coherence.shape[0]
    num_sites = optical_coherence.shape[1]
    dipole_correlation = np.zeros(num_times, dtype=complex)

    for time_index in range(num_times):
        value = 0.0 + 0.0j
        for initial_index in range(num_sites):
            for final_index in range(num_sites):
                factor = 0.0 + 0.0j
                for axis in range(3):
                    factor += np.conjugate(dipoles[initial_index, axis]) * dipoles[final_index, axis]
                value += factor * optical_coherence[time_index, initial_index, final_index]
        dipole_correlation[time_index] = value / 3.0

    return dipole_correlation


def interpolate_dipole_correlation(times, dipole_correlation, factor):
    """
    Interpolate the dipole correlation on a finer time grid.
    """
    if factor <= 1:
        return times, dipole_correlation

    num_interpolated_times = (len(times) - 1) * int(factor) + 1
    interpolated_times = np.linspace(times[0], times[-1], num_interpolated_times)

    real_spline = interpolate.CubicSpline(times, np.real(dipole_correlation))
    imag_spline = interpolate.CubicSpline(times, np.imag(dipole_correlation))

    interpolated_dipole_correlation = real_spline(interpolated_times) + 1j * imag_spline(interpolated_times)
    return interpolated_times, interpolated_dipole_correlation


def apply_gaussian_time_filter(times, dipole_correlation, sigma):
    """
    Apply a Gaussian time filter to the dipole correlation.
    """
    filtered_dipole_correlation = np.zeros_like(dipole_correlation, dtype=complex)

    for time_index in range(len(times)):
        time = times[time_index] - times[0]
        filter_value = np.exp(-(time**2) / (2.0 * sigma**2))
        filtered_dipole_correlation[time_index] = dipole_correlation[time_index] * filter_value

    return filtered_dipole_correlation


def compute_fourier_spectrum(times, dipole_correlation, frequency_min, frequency_max, num_frequencies):
    """
    Compute the real part of the one-sided Fourier transform.
    """
    frequencies = np.linspace(frequency_min, frequency_max, int(num_frequencies))
    spectrum = np.zeros(len(frequencies), dtype=float)
    for frequency_index in range(len(frequencies)):
        frequency = frequencies[frequency_index]
        integrand = np.exp(1j * frequency * times) * dipole_correlation
        spectrum[frequency_index] = np.real(np.trapezoid(integrand, x=times))

    return frequencies, spectrum


def normalize_absorption_spectrum(spectrum):
    """
    Normalize the absorption spectrum by its maximum value.
    """
    maximum = np.max(spectrum)

    if maximum == 0.0:
        return spectrum

    return spectrum / maximum


def process_optical_coherence(times, optical_coherence, dipoles):
    """
    Compute all post-processed absorption data for one optical-coherence dataset.
    """
    dipole_correlation = compute_dipole_correlation(optical_coherence, dipoles)
    interpolated_times, interpolated_dipole_correlation = interpolate_dipole_correlation(
        times, dipole_correlation, interpolation_factor
    )

    if apply_gaussian_filter:
        filtered_dipole_correlation = apply_gaussian_time_filter(
            interpolated_times, interpolated_dipole_correlation, gaussian_sigma
        )
        spectrum_input = filtered_dipole_correlation
    else:
        filtered_dipole_correlation = None
        spectrum_input = interpolated_dipole_correlation

    frequencies, spectrum = compute_fourier_spectrum(
        interpolated_times, spectrum_input, frequency_min, frequency_max, frequency_points
    )
    frequencies = frequencies + frequency_shift

    if normalize_spectrum:
        spectrum = normalize_absorption_spectrum(spectrum)

    return interpolated_times, interpolated_dipole_correlation, filtered_dipole_correlation, frequencies, spectrum


def plot_dipole_correlation(times, values, reference_times, reference_values, label_1, label_2, output_file):
    """Compare the absolute values of the unfiltered complex correlation functions."""
    fig, ax = plt.subplots(figsize=two_column_figure_size)
    times_fs = times / units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME
    reference_times_fs = reference_times / units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME
    ax.plot(times_fs, np.abs(values), label=label_1)
    ax.plot(reference_times_fs, np.abs(reference_values), "--", label=label_2)
    ax.set_xlim(min(times_fs[0], reference_times_fs[0]), max(times_fs[-1], reference_times_fs[-1]))
    ax.set_xlabel("Time [fs]")
    ax.set_ylabel(r"$|C(t)|$")
    ax.grid(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2)
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")


def plot_absorption_spectrum(
    frequencies, spectrum, reference_frequencies, reference_spectrum, label_1, label_2, output_file
):
    """Plot the computed and reference spectra with the same processing settings."""
    fig, ax = plt.subplots(figsize=one_column_figure_size)
    ax.plot(frequencies, spectrum, label=label_1)
    ax.plot(reference_frequencies, reference_spectrum, "--", label=label_2)
    ax.set_xlabel(r"Frequency [cm$^{-1}$]")
    ax.set_ylabel("Absorption [a.u.]")
    ax.set_xlim(frequency_min + frequency_shift, frequency_max + frequency_shift)
    ax.grid(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")


def main(show_plots=True):
    """Compute and compare spectra from the current run and the selected reference."""
    project_folder = Path(__file__).resolve().parent.parent
    model = run_path(str(project_folder / "run_simulation.py"))
    output_folder = model["output_directory"]
    reference_file = model["reference_file"]
    times, optical_coherence = load_optical_coherence_data(
        output_folder, f"optical_coherence_{model['output_identifier']}.npz"
    )
    reference_times, reference_coherence = load_optical_coherence_data(reference_file.parent, reference_file.name)
    check_optical_coherence_consistency(optical_coherence, reference_coherence)
    dipoles = np.asarray(model["electric_dipoles"], dtype=complex)
    if dipoles.shape != (optical_coherence.shape[1], 3) or not np.all(np.isfinite(dipoles)):
        raise ValueError("electric_dipoles must be a finite array with shape (number_of_sites, 3).")

    processed_times, correlation, _, frequencies, spectrum = process_optical_coherence(
        times, optical_coherence, dipoles
    )
    processed_reference_times, reference_correlation, _, reference_frequencies, reference_spectrum = (
        process_optical_coherence(reference_times, reference_coherence, dipoles)
    )
    np.save(output_folder / "absorption_spectrum.npy", np.column_stack((frequencies, spectrum)))
    np.save(
        output_folder / "absorption_spectrum_reference.npy",
        np.column_stack((reference_frequencies, reference_spectrum)),
    )
    label_1 = f"Simulation: BD = {model['config'].maximum_bond_dimension}"
    label_2 = f"Reference: BD = {model['reference_bond_dimension']}"
    plot_dipole_correlation(
        processed_times,
        correlation,
        processed_reference_times,
        reference_correlation,
        label_1,
        label_2,
        output_folder / "dipole_correlation.pdf",
    )
    plot_absorption_spectrum(
        frequencies,
        spectrum,
        reference_frequencies,
        reference_spectrum,
        label_1,
        label_2,
        output_folder / "absorption_spectrum.pdf",
    )
    print("Saved absorption comparison figures in", output_folder)
    if show_plots:
        plt.show()


if __name__ == "__main__":
    main()
