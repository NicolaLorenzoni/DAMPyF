"""
Estimate the required local pseudomode Fock-space dimensions based on the
pseudomode parameter file in input_data/pseudomodes.

Only modify the section marked "User input".
"""

from pathlib import Path
import os
import sys

script_folder = Path(__file__).resolve().parent
project_folder = script_folder.parent
sys.path.insert(0, str(project_folder))

from dampf_modules import unit_conventions as units

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


# =============================================================================
# User input
# =============================================================================

# Existing local pseudomode-parameter file in the input_data/pseudomodes folder.
pseudomode_input_filename = "Simple_environment.txt"

# Output file containing the same pseudomode parameters with the estimated Fock dimensions.
updated_pseudomode_output_filename = ("Simple_environment_estimated_fock_dims.txt")

# Test propagation time and time step.
time = 100.0 * units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME
dt = 1.0 * units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME

# Population thresholds used to determine the cutoff Fock dimension.
negligible_population_threshold = 1.0e-6
negligible_average_population_threshold = 1.0e-6

# Search range for the tested Fock dimension.
minimum_test_fock_dimension = 2
maximum_test_fock_dimension = 30

# Extra dimension added on top of the estimated one
safety_padding = 0







# =============================================================================
# End of user input
# Do not modify below this line
# =============================================================================

os.chdir(project_folder)

input_folder = project_folder / "input_data" / "pseudomodes"
output_folder = input_folder


# =============================================================================
# Helper functions
# =============================================================================

def check_inputs(input_file, parameters=None):
    """
    Check file names, user-input parameters, and pseudomode parameters.
    """
    if Path(pseudomode_input_filename).name != pseudomode_input_filename:
        raise ValueError("pseudomode_input_filename should be a file name, not a path.")

    if (Path(updated_pseudomode_output_filename).name != updated_pseudomode_output_filename):
        raise ValueError("updated_pseudomode_output_filename should be a file name, not a path.")

    if not input_file.is_file():
        raise FileNotFoundError(f"Pseudomode file not found: {input_file}")

    if time < 0.0:
        raise ValueError("time must be non-negative.")

    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    if time > 0.0:
        ratio = time / dt
        if abs(ratio - round(ratio)) > 1.0e-12:
            raise ValueError("time must be an integer multiple of dt.")

    if not 0.0 < negligible_population_threshold < 1.0:
        raise ValueError(
            "negligible_population_threshold must be between 0 and 1."
        )

    if not 0.0 < negligible_average_population_threshold < 1.0:
        raise ValueError(
            "negligible_average_population_threshold must be between 0 and 1."
        )

    if minimum_test_fock_dimension < 2:
        raise ValueError("minimum_test_fock_dimension must be at least 2.")

    if maximum_test_fock_dimension < minimum_test_fock_dimension:
        raise ValueError(
            "maximum_test_fock_dimension must be >= "
            "minimum_test_fock_dimension."
        )

    if int(minimum_test_fock_dimension) != minimum_test_fock_dimension:
        raise ValueError("minimum_test_fock_dimension must be an integer.")

    if int(maximum_test_fock_dimension) != maximum_test_fock_dimension:
        raise ValueError("maximum_test_fock_dimension must be an integer.")

    if int(safety_padding) != safety_padding or safety_padding < 0:
        raise ValueError(
            "safety_padding must be a non-negative integer."
        )

    if parameters is None:
        return

    if parameters.ndim != 2:
        raise ValueError(
            "The pseudomode file could not be interpreted as a table."
        )

    if parameters.shape[1] not in (4, 5):
        raise ValueError(
            "The pseudomode file must have four or five columns: "
            "frequency, damping rate, coupling, thermal energy, and "
            "optional Fock dimension."
        )

    if not np.all(np.isfinite(parameters[:, :4])):
        raise ValueError(
            "Pseudomode frequencies, damping rates, couplings, and "
            "thermal energies must be finite."
        )

    if np.any(np.imag(parameters[:, [0, 1, 3]]) != 0.0):
        raise ValueError(
            "Pseudomode frequencies, damping rates, and thermal energies "
            "must be real."
        )

    if np.any(np.real(parameters[:, 0]) == 0.0):
        raise ValueError("All pseudomode frequencies must be nonzero.")

    if np.any(np.real(parameters[:, 1]) <= 0.0):
        raise ValueError(
            "All pseudomode damping rates must be strictly positive."
        )

    if np.any(np.real(parameters[:, 3]) < 0.0):
        raise ValueError(
            "All pseudomode thermal energies must be non-negative."
        )

    if np.any(
        (np.real(parameters[:, 0]) < 0.0)
        & (np.real(parameters[:, 3]) > 0.0)
    ):
        raise ValueError(
            "Negative-frequency pseudomodes must have zero thermal energy."
        )


def read_pseudomode_file(filename):
    """
    Read a local pseudomode-parameter file.
    """
    input_file = input_folder / filename
    check_inputs(input_file)

    parameters = np.loadtxt(input_file, dtype=complex)
    if parameters.ndim == 1:
        parameters = parameters.reshape(1, -1)

    check_inputs(input_file, parameters)

    frequencies = np.real(parameters[:, 0]).copy()
    damping_rates = np.real(parameters[:, 1]).copy()
    couplings = parameters[:, 2].copy()
    thermal_energies = np.real(parameters[:, 3]).copy()

    return frequencies, damping_rates, couplings, thermal_energies


def estimate_thermal_fock_dimension(frequency, thermal_energy):
    """
    Estimate the Fock dimension required by the initial thermal state.
    """
    if thermal_energy <= 0.0:
        return 1

    q = float(np.exp(-frequency / thermal_energy))

    thermal_dimension = int(
        np.ceil(
            np.log(negligible_population_threshold)
            / np.log(q)
        )
    )

    return max(1, thermal_dimension)


def lindblad_superoperator(jump_operator, identity):
    """
    Return the vectorized Lindblad dissipator D[L].

    Column-major vectorization is used:
    vec(rho) = rho.reshape(-1, order="F").
    """
    jump_dagger = jump_operator.conjugate().transpose()
    jump_dagger_jump = jump_dagger @ jump_operator

    dissipator = sparse.kron(
        jump_operator.conjugate(),
        jump_operator,
        format="csr",
    )
    dissipator -= 0.5 * sparse.kron(
        identity,
        jump_dagger_jump,
        format="csr",
    )
    dissipator -= 0.5 * sparse.kron(
        jump_dagger_jump.transpose(),
        identity,
        format="csr",
    )

    return dissipator


def build_liouvillian(
    frequency,
    damping_rate,
    coupling,
    thermal_energy,
    dim,
):
    """
    Build the sparse Liouvillian for one displaced damped oscillator.
    """
    annihilation = sparse.diags(
        np.sqrt(np.arange(1, dim, dtype=float)),
        offsets=1,
        shape=(dim, dim),
        dtype=complex,
        format="csr",
    )
    creation = annihilation.conjugate().transpose()
    identity = sparse.identity(dim, dtype=complex, format="csr")

    number_operator = creation @ annihilation
    hamiltonian = frequency * number_operator
    hamiltonian += (
        coupling * creation
        + np.conjugate(coupling) * annihilation
    )

    liouvillian = -1.0j * (
        sparse.kron(identity, hamiltonian, format="csr")
        - sparse.kron(
            hamiltonian.transpose(),
            identity,
            format="csr",
        )
    )

    if thermal_energy <= 0.0:
        thermal_occupation = 0.0
    else:
        exponent = frequency / thermal_energy
        thermal_occupation = (
            0.0
            if exponent > 700.0
            else 1.0 / np.expm1(exponent)
        )

    liouvillian += (
        damping_rate
        * (thermal_occupation + 1.0)
        * lindblad_superoperator(annihilation, identity)
    )
    liouvillian += (
        damping_rate
        * thermal_occupation
        * lindblad_superoperator(creation, identity)
    )

    return liouvillian.tocsr()


def initial_thermal_state_vector(
    frequency,
    thermal_energy,
    dim,
):
    """
    Return the vectorized truncated thermal state.
    """
    if thermal_energy <= 0.0:
        q = 0.0
    else:
        q = float(np.exp(-frequency / thermal_energy))

    probabilities = q ** np.arange(dim, dtype=float)
    if q < 1.0:
        probabilities *= 1.0 - q

    probabilities /= np.sum(probabilities)

    return np.diag(
        probabilities.astype(complex)
    ).reshape(-1, order="F")


def highest_fock_state_population_over_time(
    liouvillian,
    rho_initial_vector,
    dim,
):
    """
    Propagate the oscillator and return the highest-Fock-state population.
    """
    number_of_steps = int(round(time / dt))

    if number_of_steps == 0:
        rho_vectors = rho_initial_vector.reshape(1, -1)
    else:
        rho_vectors = expm_multiply(
            liouvillian,
            rho_initial_vector,
            start=0.0,
            stop=time,
            num=number_of_steps + 1,
            endpoint=True,
        )

    highest_fock_state_index = (
        (dim - 1) + dim * (dim - 1)
    )
    highest_fock_state_population = np.abs(
        rho_vectors[:, highest_fock_state_index]
    )

    return highest_fock_state_population


def test_fock_dimension(
    frequency,
    damping_rate,
    coupling,
    thermal_energy,
    dim,
):
    """
    Test one Fock dimension and return the cutoff-population data.
    """
    liouvillian = build_liouvillian(
        frequency,
        damping_rate,
        coupling,
        thermal_energy,
        dim,
    )
    rho_initial_vector = initial_thermal_state_vector(
        frequency,
        thermal_energy,
        dim,
    )
    highest_fock_state_population = (
        highest_fock_state_population_over_time(
            liouvillian,
            rho_initial_vector,
            dim,
        )
    )

    maximum_population = float(
        np.max(highest_fock_state_population)
    )
    average_population = float(
        np.mean(highest_fock_state_population)
    )

    return maximum_population, average_population


def estimate_dimension(
    mode_index,
    frequency,
    damping_rate,
    coupling,
    thermal_energy,
):
    """
    Estimate the recommended production Fock dimension for one pseudomode.
    """
    thermal_dimension = estimate_thermal_fock_dimension(
        frequency,
        thermal_energy,
    )
    first_test_dimension = max(
        int(minimum_test_fock_dimension),
        thermal_dimension + 1,
    )

    for test_dimension in range(
        first_test_dimension,
        int(maximum_test_fock_dimension) + 1,
    ):
        maximum_population, average_population = (
            test_fock_dimension(
                frequency,
                damping_rate,
                coupling,
                thermal_energy,
                test_dimension,
            )
        )

        if (
            maximum_population < negligible_population_threshold
            and average_population
            < negligible_average_population_threshold
        ):
            recommended_dimension = (
                test_dimension - 1 + int(safety_padding)
            )
            recommended_dimension = max(
                recommended_dimension,
                thermal_dimension,
                1,
            )

            return {
                "thermal_dimension": int(thermal_dimension),
                "tested_dimension": int(test_dimension),
                "recommended_dimension": int(
                    recommended_dimension
                ),
                "maximum_highest_fock_state_population": float(
                    maximum_population
                ),
                "average_highest_fock_state_population": float(
                    average_population
                ),
            }

    raise RuntimeError(
        f"Pseudomode {mode_index} did not converge up to "
        "maximum_test_fock_dimension = "
        f"{maximum_test_fock_dimension}. The script stops here "
        "and the output file is not written. Increase "
        "maximum_test_fock_dimension or relax "
        "negligible_population_threshold / "
        "negligible_average_population_threshold."
    )


def print_header(Q):
    """
    Print a summary of the estimator settings.
    """
    print("Pseudomode Fock-dimension estimator")
    print("------------------------------------")
    print(
        f"Input file                             : "
        f"{pseudomode_input_filename}"
    )
    print(f"Number of local pseudomodes            : {Q}")
    print(f"time                                   : {time}")
    print(f"dt                                     : {dt}")
    print(
        f"negligible_population_threshold         : "
        f"{negligible_population_threshold}"
    )
    print(
        f"negligible_average_population_threshold : "
        f"{negligible_average_population_threshold}"
    )
    print(
        f"minimum_test_fock_dimension             : "
        f"{minimum_test_fock_dimension}"
    )
    print(
        f"maximum_test_fock_dimension             : "
        f"{maximum_test_fock_dimension}"
    )
    print(f"safety_padding                          : {safety_padding}")
    print("")

    print(
        f"{'mode':>4}  "
        f"{'freq':>11}  "
        f"{'damping':>11}  "
        f"{'coup':>16}  "
        f"{'thermal_en':>10}  "
        f"{'max_highest':>12}  "
        f"{'avg_highest':>12}  "
        f"{'Estimated Fock dim':>18}"
    )
    print(
        f"{'-' * 4}  "
        f"{'-' * 11}  "
        f"{'-' * 11}  "
        f"{'-' * 16}  "
        f"{'-' * 10}  "
        f"{'-' * 12}  "
        f"{'-' * 12}  "
        f"{'-' * 18}"
    )


def print_result_line(
    mode_index,
    frequency,
    damping_rate,
    coupling,
    thermal_energy,
    result,
):
    """
    Print one formatted result line.
    """
    coupling = complex(coupling)
    coupling_string = f"{coupling.real:.4g}{coupling.imag:+.4g}j"

    print(
        f"{mode_index:4d}  "
        f"{frequency:11.4f}  "
        f"{damping_rate:11.4f}  "
        f"{coupling_string:>16}  "
        f"{thermal_energy:10.3f}  "
        f"{result['maximum_highest_fock_state_population']:12.4e}  "
        f"{result['average_highest_fock_state_population']:12.4e}  "
        f"{result['recommended_dimension']:18d}"
    )


def write_updated_file(
    frequencies,
    damping_rates,
    couplings,
    thermal_energies,
    recommended_dimensions,
):
    """
    Write a new pseudomode-parameter file with estimated Fock dimensions.
    """
    output_file = (
        output_folder / updated_pseudomode_output_filename
    )

    header = (
        "frequency damping_rate coupling thermal_energy fock_dim"
    )
    with open(output_file, "w") as file_handle:
        file_handle.write(f"# {header}\n")

        for mode_index in range(len(frequencies)):
            coupling = complex(couplings[mode_index])
            coupling_string = f"{coupling.real:.10g}{coupling.imag:+.10g}j"
            file_handle.write(
                f"{frequencies[mode_index]:10.4f} "
                f"{damping_rates[mode_index]:10.4f} "
                f"{coupling_string:>24} "
                f"{thermal_energies[mode_index]:10.4f} "
                f"{int(recommended_dimensions[mode_index]):5d}\n"
            )

    print("")
    print(f"Wrote {output_file.name} in {output_folder}")


# =============================================================================
# Main script
# =============================================================================

frequencies, damping_rates, couplings, thermal_energies = (
    read_pseudomode_file(pseudomode_input_filename)
)
Q = len(frequencies)

print_header(Q)

recommended_dimensions = np.zeros(Q, dtype=int)

for mode_index in range(Q):
    result = estimate_dimension(
        mode_index,
        frequencies[mode_index],
        damping_rates[mode_index],
        couplings[mode_index],
        thermal_energies[mode_index],
    )
    recommended_dimensions[mode_index] = (
        result["recommended_dimension"]
    )
    print_result_line(
        mode_index,
        frequencies[mode_index],
        damping_rates[mode_index],
        couplings[mode_index],
        thermal_energies[mode_index],
        result,
    )

write_updated_file(
    frequencies,
    damping_rates,
    couplings,
    thermal_energies,
    recommended_dimensions,
)
