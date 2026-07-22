"""
Simple post-processing script for DAMPF absorption spectra.

Run this script from any folder. It reads optical-coherence data from this
output_data folder, computes the dipole correlation, applies an optional
Gaussian time filter, and computes a one-sided Fourier transform.
"""

from pathlib import Path
import sys

script_folder = Path(__file__).resolve().parent
project_folder = script_folder.parent
sys.path.insert(0, str(project_folder))

from dampf_modules import unit_conventions as units

import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt


# =============================================================================
# User options
# =============================================================================
optical_coherence_file = "optical_coherence_dimer_test_simulation.npz"  # Optical-coherence file inside output_data
optical_coherence_file_2 = "optical_coherence_dimer_test_simulation_2.npz"   # None for no comparison, or another optical-coherence file inside output_data

label_1 = "dataset 1"
label_2 = "dataset 2"

electric_dipoles_file = "dimer_electric_dipoles.txt"  # Electric-dipole file inside input_data/system

frequency_shift = 0.0       # Added to the Fourier frequency axis
frequency_min = 0.0      # Minimum Fourier frequency before frequency_shift
frequency_max = 2000.0   # Maximum Fourier frequency before frequency_shift
frequency_points = 6000

interpolation_factor = 5    # 1 means no interpolation; 5 gives five times more time points
apply_gaussian_filter = True
gaussian_sigma = 50.0*units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME
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

    data = np.load(file_path)
    if "times" not in data or "optical_coherence" not in data:
        raise ValueError("optical_coherence_file must contain arrays named 'times' and 'optical_coherence'.")

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


def load_dipoles(system_input_folder, filename, num_sites):
    """
    Load site dipoles from input_data/system.
    """
    if filename is None:
        raise ValueError("electric_dipoles_file must be specified explicitly.")

    file_path = system_input_folder / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}")

    dipoles = np.loadtxt(file_path, dtype=float)

    if dipoles.shape != (num_sites, 3):
        raise ValueError(f"Dipoles must have shape ({num_sites}, 3), got {dipoles.shape}.")

    print(f"Loaded {file_path}")
    return dipoles


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

    The convention follows the old absorption code:
    C(t) = sum_{i,j} dot(conj(mu_i), mu_j) optical_coherence[t,i,j].
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
        dipole_correlation[time_index] = value

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
        filter_value = np.exp(-time**2 / (2.0 * sigma**2))
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
    interpolated_times, interpolated_dipole_correlation = interpolate_dipole_correlation(times, dipole_correlation, interpolation_factor)

    if apply_gaussian_filter:
        filtered_dipole_correlation = apply_gaussian_time_filter(interpolated_times, interpolated_dipole_correlation, gaussian_sigma)
        spectrum_input = filtered_dipole_correlation
    else:
        filtered_dipole_correlation = None
        spectrum_input = interpolated_dipole_correlation

    frequencies, spectrum = compute_fourier_spectrum(interpolated_times, spectrum_input, frequency_min, frequency_max, frequency_points)
    frequencies = frequencies + frequency_shift

    if normalize_spectrum:
        spectrum = normalize_absorption_spectrum(spectrum)

    return interpolated_times, interpolated_dipole_correlation, filtered_dipole_correlation, frequencies, spectrum


def plot_dipole_correlation(times, dipole_correlation, filtered_dipole_correlation, times_2, dipole_correlation_2, filtered_dipole_correlation_2, label_1, label_2):
    """
    Plot the real and imaginary parts of the dipole correlation.
    """
    fig, axes = plt.subplots(1, 2, figsize=two_column_figure_size)
    ax_real, ax_imag = axes

    ax_real.plot(times, np.real(dipole_correlation), label=fr"{label_1}: $C(t)$")
    ax_imag.plot(times, np.imag(dipole_correlation), label=fr"{label_1}: $C(t)$")

    if filtered_dipole_correlation is not None:
        ax_real.plot(times, np.real(filtered_dipole_correlation), "--", label=fr"{label_1}: $C_{{\rm filt}}(t)$")
        ax_imag.plot(times, np.imag(filtered_dipole_correlation), "--", label=fr"{label_1}: $C_{{\rm filt}}(t)$")

    if dipole_correlation_2 is not None:
        ax_real.plot(times_2, np.real(dipole_correlation_2), ":", label=fr"{label_2}: $C(t)$")
        ax_imag.plot(times_2, np.imag(dipole_correlation_2), ":", label=fr"{label_2}: $C(t)$")

        if filtered_dipole_correlation_2 is not None:
            ax_real.plot(times_2, np.real(filtered_dipole_correlation_2), "-.", label=fr"{label_2}: $C_{{\rm filt}}(t)$")
            ax_imag.plot(times_2, np.imag(filtered_dipole_correlation_2), "-.", label=fr"{label_2}: $C_{{\rm filt}}(t)$")

    ax_real.set_title("Real part dipole correlation")
    ax_imag.set_title("Imaginary part dipole correlation")
    ax_real.set_xlabel("Time")
    ax_imag.set_xlabel("Time")

    time_min = np.min(times)
    time_max = np.max(times)

    if times_2 is not None:
        time_min = min(time_min, np.min(times_2))
        time_max = max(time_max, np.max(times_2))

    ax_real.set_xlim(time_min, time_max)
    ax_imag.set_xlim(time_min, time_max)

    ax_real.grid(True)
    ax_imag.grid(True)
    ax_real.legend()
    ax_imag.legend()
    fig.tight_layout()
    fig.savefig("dipole_correlation.pdf", bbox_inches="tight")
    return


def plot_absorption_spectrum(frequencies, spectrum, frequencies_2, spectrum_2, label_1, label_2):
    """
    Plot the absorption spectrum.
    """
    fig, ax = plt.subplots(figsize=one_column_figure_size)

    ax.plot(frequencies, spectrum, label=label_1)

    if spectrum_2 is not None:
        ax.plot(frequencies_2, spectrum_2, "--", label=label_2)

    ax.set_xlabel(r"$\omega$")
    ax.set_title(r"Absorption")
    ax.set_xlim(frequency_min, frequency_max)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("absorption_spectrum.pdf", bbox_inches="tight")
    return


def save_processed_data(
    output_folder,
    file_suffix,
    times,
    dipole_correlation,
    filtered_dipole_correlation,
    frequencies,
    spectrum,
):
    """
    Save the absorption spectrum.
    """
    spectrum_data = np.zeros((len(frequencies), 2), dtype=float)

    for frequency_index in range(len(frequencies)):
        spectrum_data[frequency_index, 0] = frequencies[frequency_index]
        spectrum_data[frequency_index, 1] = spectrum[frequency_index]

    np.save(
        output_folder / f"absorption_spectrum{file_suffix}.npy",
        spectrum_data,
    )

    return

# =============================================================================
# Main script
# =============================================================================
script_folder = Path(__file__).resolve().parent
project_folder = script_folder.parent
output_folder = project_folder / "output_data"
system_input_folder = project_folder / "input_data" / "system"

times, optical_coherence = load_optical_coherence_data(output_folder, optical_coherence_file)
times_2 = None
optical_coherence_2 = None
if optical_coherence_file_2 is not None:
    times_2, optical_coherence_2 = load_optical_coherence_data(output_folder, optical_coherence_file_2)

check_optical_coherence_consistency(optical_coherence, optical_coherence_2)

num_sites = optical_coherence.shape[1]
dipoles = load_dipoles(system_input_folder, electric_dipoles_file, num_sites)

interpolated_times, interpolated_dipole_correlation, filtered_dipole_correlation, frequencies, spectrum = process_optical_coherence(times, optical_coherence, dipoles)

interpolated_times_2 = None
interpolated_dipole_correlation_2 = None
filtered_dipole_correlation_2 = None
frequencies_2 = None
spectrum_2 = None
if optical_coherence_2 is not None:
    interpolated_times_2, interpolated_dipole_correlation_2, filtered_dipole_correlation_2, frequencies_2, spectrum_2 = process_optical_coherence(times_2, optical_coherence_2, dipoles)

save_processed_data(output_folder, "", interpolated_times, interpolated_dipole_correlation, filtered_dipole_correlation, frequencies, spectrum)
if optical_coherence_2 is not None:
    save_processed_data(output_folder, "_2", interpolated_times_2, interpolated_dipole_correlation_2, filtered_dipole_correlation_2, frequencies_2, spectrum_2)

plot_dipole_correlation(interpolated_times, interpolated_dipole_correlation, filtered_dipole_correlation, interpolated_times_2, interpolated_dipole_correlation_2, filtered_dipole_correlation_2, label_1, label_2)
plot_absorption_spectrum(frequencies, spectrum, frequencies_2, spectrum_2, label_1, label_2)

plt.show()
