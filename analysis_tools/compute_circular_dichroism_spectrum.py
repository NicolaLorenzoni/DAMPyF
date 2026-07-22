"""
Simple post-processing script for DAMPF circular dichroism spectra.

Run this script from any folder. It reads optical-coherence data from the
output_data folder, computes the electric-magnetic dipole correlation, applies
an optional Gaussian time filter, and computes a one-sided Fourier transform.
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
optical_coherence_file = "optical_coherence_test_simulation.npz"  # Optical-coherence file inside output_data
optical_coherence_file_2 = "optical_coherence_test_simulation_2.npz"   # None for no comparison, or another optical-coherence file inside output_data


label_1 = "dataset 1"
label_2 = "dataset 2"

electric_dipoles_file = "dimer_electric_dipoles.txt"    # Electric-dipole file inside input_data/system
magnetic_dipoles_file = "dimer_magnetic_dipoles.txt"    # Magnetic-dipole file inside input_data/system

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


def load_dipoles(system_input_folder, filename, num_sites, label):
    """
    Load site dipoles from input_data/system.
    """
    if filename is None:
        raise ValueError(f"{label}_dipoles_file must be specified explicitly.")

    file_path = system_input_folder / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find {file_path}")

    dipoles = np.loadtxt(file_path, dtype=float)

    if dipoles.ndim == 1:
        dipoles = dipoles.reshape(1, -1)

    if dipoles.shape != (num_sites, 3):
        raise ValueError(f"{label} dipoles must have shape ({num_sites}, 3), got {dipoles.shape}.")

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


def compute_circular_dichroism_correlation(optical_coherence, electric_dipoles, magnetic_dipoles):
    """
    Compute the electric-magnetic correlation entering the CD signal.

    This follows the same convention as the absorption correlation,

        C_abs(t) = sum_{i,j} dot(conj(mu_i), mu_j) optical_coherence[t,i,j],

    but with the first electric dipole replaced by the magnetic dipole,

        C_CD(t) = sum_{i,j} dot(conj(m_i), mu_j) optical_coherence[t,i,j].
    """
    num_times = optical_coherence.shape[0]
    num_sites = optical_coherence.shape[1]
    circular_dichroism_correlation = np.zeros(num_times, dtype=complex)

    for time_index in range(num_times):
        value = 0.0 + 0.0j
        for initial_index in range(num_sites):
            for final_index in range(num_sites):
                factor = 0.0 + 0.0j
                for axis in range(3):
                    factor += np.conjugate(magnetic_dipoles[initial_index, axis]) * electric_dipoles[final_index, axis]
                value += factor * optical_coherence[time_index, initial_index, final_index]
        circular_dichroism_correlation[time_index] = value

    return circular_dichroism_correlation


def interpolate_circular_dichroism_correlation(times, circular_dichroism_correlation, factor):
    """
    Interpolate the circular-dichroism correlation on a finer time grid.
    """
    if factor <= 1:
        return times, circular_dichroism_correlation

    num_interpolated_times = (len(times) - 1) * int(factor) + 1
    interpolated_times = np.linspace(times[0], times[-1], num_interpolated_times)

    real_spline = interpolate.CubicSpline(times, np.real(circular_dichroism_correlation))
    imag_spline = interpolate.CubicSpline(times, np.imag(circular_dichroism_correlation))

    interpolated_circular_dichroism_correlation = real_spline(interpolated_times) + 1j * imag_spline(interpolated_times)
    return interpolated_times, interpolated_circular_dichroism_correlation


def apply_gaussian_time_filter(times, circular_dichroism_correlation, sigma):
    """
    Apply a Gaussian time filter to the circular-dichroism correlation.
    """
    filtered_circular_dichroism_correlation = np.zeros_like(circular_dichroism_correlation, dtype=complex)

    for time_index in range(len(times)):
        time = times[time_index] - times[0]
        filter_value = np.exp(-time**2 / (2.0 * sigma**2))
        filtered_circular_dichroism_correlation[time_index] = circular_dichroism_correlation[time_index] * filter_value

    return filtered_circular_dichroism_correlation


def compute_fourier_spectrum(times, circular_dichroism_correlation, frequency_min, frequency_max, num_frequencies):
    """
    Compute the real part of the one-sided Fourier transform.
    """
    frequencies = np.linspace(frequency_min, frequency_max, int(num_frequencies))
    spectrum = np.zeros(len(frequencies), dtype=float)
    for frequency_index in range(len(frequencies)):
        frequency = frequencies[frequency_index]
        integrand = np.exp(1j * frequency * times) * circular_dichroism_correlation
        spectrum[frequency_index] = np.real(np.trapezoid(integrand, x=times))

    return frequencies, spectrum


def normalize_circular_dichroism_spectrum(spectrum):
    """
    Normalize the CD spectrum by the maximum absolute value, preserving the sign.
    """
    maximum = np.max(np.abs(spectrum))

    if maximum == 0.0:
        return spectrum

    return spectrum / maximum


def process_optical_coherence(times, optical_coherence, electric_dipoles, magnetic_dipoles):
    """
    Compute all post-processed circular-dichroism data for one optical-coherence dataset.
    """
    circular_dichroism_correlation = compute_circular_dichroism_correlation(optical_coherence, electric_dipoles, magnetic_dipoles)
    interpolated_times, interpolated_circular_dichroism_correlation = interpolate_circular_dichroism_correlation(times, circular_dichroism_correlation, interpolation_factor)

    if apply_gaussian_filter:
        filtered_circular_dichroism_correlation = apply_gaussian_time_filter(interpolated_times, interpolated_circular_dichroism_correlation, gaussian_sigma)
        spectrum_input = filtered_circular_dichroism_correlation
    else:
        filtered_circular_dichroism_correlation = None
        spectrum_input = interpolated_circular_dichroism_correlation

    frequencies, spectrum = compute_fourier_spectrum(interpolated_times, spectrum_input, frequency_min, frequency_max, frequency_points)
    frequencies = frequencies + frequency_shift

    if normalize_spectrum:
        spectrum = normalize_circular_dichroism_spectrum(spectrum)

    return interpolated_times, interpolated_circular_dichroism_correlation, filtered_circular_dichroism_correlation, frequencies, spectrum


def plot_circular_dichroism_correlation(times, circular_dichroism_correlation, filtered_circular_dichroism_correlation, times_2, circular_dichroism_correlation_2, filtered_circular_dichroism_correlation_2, label_1, label_2):
    """
    Plot the real and imaginary parts of the circular-dichroism correlation.
    """
    fig, ax = plt.subplots(figsize=one_column_figure_size)

    ax.plot(times, np.real(circular_dichroism_correlation), label=fr"{label_1}: Re $C_{{\rm CD}}(t)$")
    ax.plot(times, np.imag(circular_dichroism_correlation), "--", label=fr"{label_1}: Im $C_{{\rm CD}}(t)$")

    if filtered_circular_dichroism_correlation is not None:
        ax.plot(times, np.real(filtered_circular_dichroism_correlation), label=fr"{label_1}: Re $C_{{\rm CD,filt}}(t)$")

    if circular_dichroism_correlation_2 is not None:
        ax.plot(times_2, np.real(circular_dichroism_correlation_2), ":", label=fr"{label_2}: Re $C_{{\rm CD}}(t)$")
        ax.plot(times_2, np.imag(circular_dichroism_correlation_2), "-.", label=fr"{label_2}: Im $C_{{\rm CD}}(t)$")

        if filtered_circular_dichroism_correlation_2 is not None:
            ax.plot(times_2, np.real(filtered_circular_dichroism_correlation_2), "--", label=fr"{label_2}: Re $C_{{\rm CD,filt}}(t)$")

    ax.set_xlabel(r"$t$")
    ax.set_title(r"Circular dichroism")
    ax.set_xlim(frequency_min, frequency_max)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("circular_dichroism_correlation.pdf", bbox_inches="tight")
    return


def plot_circular_dichroism_spectrum(frequencies, spectrum, frequencies_2, spectrum_2, label_1, label_2):
    """
    Plot the circular dichroism spectrum.
    """
    fig, ax = plt.subplots(figsize=one_column_figure_size)

    ax.plot(frequencies, spectrum, label=label_1)

    if spectrum_2 is not None:
        ax.plot(frequencies_2, spectrum_2, "--", label=label_2)

    ax.set_xlabel(r"$\omega$")
    ax.set_title(r"Circular dichroism")
    ax.set_xlim(frequency_min, frequency_max)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig("circular_dichroism_spectrum.pdf", bbox_inches="tight")
    return


def save_processed_data(
    output_folder,
    file_suffix,
    times,
    circular_dichroism_correlation,
    filtered_circular_dichroism_correlation,
    frequencies,
    spectrum,
):
    """
    Save the circular-dichroism spectrum.
    """
    spectrum_data = np.zeros((len(frequencies), 2), dtype=float)

    for frequency_index in range(len(frequencies)):
        spectrum_data[frequency_index, 0] = frequencies[frequency_index]
        spectrum_data[frequency_index, 1] = spectrum[frequency_index]

    np.save(
        output_folder / f"circular_dichroism_spectrum{file_suffix}.npy",
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
electric_dipoles = load_dipoles(system_input_folder, electric_dipoles_file, num_sites, "electric")
magnetic_dipoles = load_dipoles(system_input_folder, magnetic_dipoles_file, num_sites, "magnetic")

interpolated_times, interpolated_circular_dichroism_correlation, filtered_circular_dichroism_correlation, frequencies, spectrum = process_optical_coherence(times, optical_coherence, electric_dipoles, magnetic_dipoles)

interpolated_times_2 = None
interpolated_circular_dichroism_correlation_2 = None
filtered_circular_dichroism_correlation_2 = None
frequencies_2 = None
spectrum_2 = None
if optical_coherence_2 is not None:
    interpolated_times_2, interpolated_circular_dichroism_correlation_2, filtered_circular_dichroism_correlation_2, frequencies_2, spectrum_2 = process_optical_coherence(times_2, optical_coherence_2, electric_dipoles, magnetic_dipoles)

save_processed_data(output_folder, "", interpolated_times, interpolated_circular_dichroism_correlation, filtered_circular_dichroism_correlation, frequencies, spectrum)
if optical_coherence_2 is not None:
    save_processed_data(output_folder, "_2", interpolated_times_2, interpolated_circular_dichroism_correlation_2, filtered_circular_dichroism_correlation_2, frequencies_2, spectrum_2)

plot_circular_dichroism_correlation(interpolated_times, interpolated_circular_dichroism_correlation, filtered_circular_dichroism_correlation, interpolated_times_2, interpolated_circular_dichroism_correlation_2, filtered_circular_dichroism_correlation_2, label_1, label_2)
plot_circular_dichroism_spectrum(frequencies, spectrum, frequencies_2, spectrum_2, label_1, label_2)

plt.show()
