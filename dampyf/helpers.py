"""
    Importable array-construction and estimation helpers for DAMPyF.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from .parameters import create_pseudomode_parameters, create_system_parameters
from .structures import LocalPseudomodes
from .utils import estimate_memory_requirements

from .configuration import DampfConfig


def build_system_hamiltonian(site_energies, system_couplings, energy_shift=0.0):
    """
        Build a Hermitian site-basis Hamiltonian from energies and couplings.
    """
    site_energies = np.asarray(site_energies, dtype=float)
    system_couplings = np.asarray(system_couplings, dtype=complex)
    if site_energies.ndim != 1 or len(site_energies) == 0:
        raise ValueError("site_energies must be a non-empty one-dimensional array.")
    number_of_sites = len(site_energies)
    if system_couplings.shape != (number_of_sites, number_of_sites):
        raise ValueError(f"system_couplings must have shape ({number_of_sites}, {number_of_sites}).")
    if not np.all(np.isfinite(site_energies)) or not np.all(np.isfinite(system_couplings)):
        raise ValueError("Site energies and system couplings must be finite.")
    if not np.allclose(np.diag(system_couplings), 0.0, atol=1.0e-14):
        raise ValueError("The diagonal of system_couplings must be zero.")

    if np.allclose(np.tril(system_couplings, k=-1), 0.0, atol=1.0e-14):
        system_couplings = system_couplings + system_couplings.T.conj()
    elif not np.allclose(system_couplings, system_couplings.T.conj(), atol=1.0e-10):
        raise ValueError("system_couplings must be Hermitian or contain only its upper triangle.")

    return np.diag(site_energies + float(energy_shift)).astype(complex) + system_couplings


def build_magnetic_dipoles(electric_dipoles, site_positions):
    """
        Construct magnetic transition dipoles from positions and electric dipoles.
    """
    electric_dipoles = np.asarray(electric_dipoles, dtype=complex)
    site_positions = np.asarray(site_positions, dtype=float)
    if electric_dipoles.ndim != 2 or electric_dipoles.shape[1] != 3:
        raise ValueError("electric_dipoles must have shape (number_of_sites, 3).")
    if site_positions.shape != electric_dipoles.shape:
        raise ValueError(f"site_positions must have shape {electric_dipoles.shape}.")
    if not np.all(np.isfinite(electric_dipoles)) or not np.all(np.isfinite(site_positions)):
        raise ValueError("Dipoles and site positions must be finite.")
    return np.cross(site_positions, electric_dipoles)


def site_density_matrix(number_of_sites, site):
    """
        Return a local site excitation initial density matrix.
    """
    if isinstance(number_of_sites, (bool, np.bool_)) or not isinstance(number_of_sites, (int, np.integer)):
        raise ValueError("number_of_sites must be a positive integer.")
    if isinstance(site, (bool, np.bool_)) or not isinstance(site, (int, np.integer)):
        raise ValueError("site must be an integer.")
    number_of_sites = int(number_of_sites)
    site = int(site)
    if number_of_sites <= 0 or not 0 <= site < number_of_sites:
        raise ValueError("site must identify one of the system sites.")
    density_matrix = np.zeros((number_of_sites, number_of_sites), dtype=complex)
    density_matrix[site, site] = 1.0
    return density_matrix


def eigenstate_density_matrix(hamiltonian, eigenstate):
    """
        Return one Hamiltonian eigenstate projector, ordered by increasing energy.
    """
    system = create_system_parameters(hamiltonian)
    if isinstance(eigenstate, (bool, np.bool_)) or not isinstance(eigenstate, (int, np.integer)):
        raise ValueError("eigenstate must be an integer.")
    eigenstate = int(eigenstate)
    if not 0 <= eigenstate < system.N:
        raise ValueError(f"eigenstate must be between 0 and {system.N - 1}.")
    amplitudes = system.eig_transf_matrix[:, eigenstate]
    return np.outer(amplitudes, amplitudes.conj())


def polarized_density_matrix(electric_dipoles, polarization):
    """
        Return the normalized state prepared by a linearly polarized pulse.
    """
    electric_dipoles = np.asarray(electric_dipoles, dtype=complex)
    polarization = np.asarray(polarization, dtype=float)
    if electric_dipoles.ndim != 2 or electric_dipoles.shape[1] != 3:
        raise ValueError("electric_dipoles must have shape (number_of_sites, 3).")
    if not np.all(np.isfinite(electric_dipoles)) or not np.all(np.isfinite(polarization)):
        raise ValueError("electric_dipoles and polarization must be finite.")
    if polarization.shape != (3,) or np.linalg.norm(polarization) == 0.0:
        raise ValueError("polarization must be a nonzero vector with shape (3,).")
    amplitudes = electric_dipoles @ (polarization / np.linalg.norm(polarization))
    weight = np.sum(np.abs(amplitudes) ** 2)
    if weight == 0.0:
        raise ValueError("The pulse polarization is orthogonal to every transition dipole.")
    amplitudes /= np.sqrt(weight)
    return np.outer(amplitudes, amplitudes.conj())


def estimate_required_memory(config, hamiltonian, pseudomodes):
    """
        Return the same order-of-magnitude memory components used by DAMPyF, in GiB.
    """
    if not isinstance(config, DampfConfig):
        raise TypeError("config must be a DampfConfig instance.")
    maximum_bond_dimension = config.maximum_bond_dimension
    if (
        isinstance(maximum_bond_dimension, (bool, np.bool_))
        or not isinstance(maximum_bond_dimension, (int, np.integer))
        or maximum_bond_dimension <= 0
    ):
        raise ValueError("maximum_bond_dimension must be a positive integer.")
    if config.simulation_mode not in ("energy_transfer", "linear_spectra"):
        raise ValueError("simulation_mode must be either 'energy_transfer' or 'linear_spectra'.")
    if config.trotter_order not in ("first", "second"):
        raise ValueError("trotter_order must be either 'first' or 'second'.")
    number_of_sites = create_system_parameters(hamiltonian).N
    parameters = create_pseudomode_parameters(pseudomodes, number_of_sites)
    return estimate_memory_requirements(config, parameters)


def _lindblad_superoperator(jump_operator, identity):
    jump_dagger = jump_operator.conj().T
    jump_dagger_jump = jump_dagger @ jump_operator
    dissipator = sparse.kron(jump_operator.conj(), jump_operator, format="csr")
    dissipator -= 0.5 * sparse.kron(identity, jump_dagger_jump, format="csr")
    dissipator -= 0.5 * sparse.kron(jump_dagger_jump.T, identity, format="csr")
    return dissipator


def _oscillator_liouvillian(frequency, damping_rate, coupling, thermal_energy, dimension):
    annihilation = sparse.diags(
        np.sqrt(np.arange(1, dimension, dtype=float)),
        offsets=1,
        shape=(dimension, dimension),
        dtype=complex,
        format="csr",
    )
    creation = annihilation.conj().T
    identity = sparse.identity(dimension, dtype=complex, format="csr")
    hamiltonian = frequency * (creation @ annihilation) + coupling * creation + np.conjugate(coupling) * annihilation
    liouvillian = -1.0j * (
        sparse.kron(identity, hamiltonian, format="csr") - sparse.kron(hamiltonian.T, identity, format="csr")
    )
    exponent = np.inf if thermal_energy <= 0.0 else frequency / thermal_energy
    thermal_occupation = 0.0 if exponent > 700.0 else 1.0 / np.expm1(exponent)
    liouvillian += damping_rate * (thermal_occupation + 1.0) * _lindblad_superoperator(annihilation, identity)
    liouvillian += damping_rate * thermal_occupation * _lindblad_superoperator(creation, identity)
    return liouvillian.tocsr()


def _thermal_state_vector(frequency, thermal_energy, dimension):
    q = 0.0 if thermal_energy <= 0.0 else float(np.exp(-frequency / thermal_energy))
    probabilities = q ** np.arange(dimension, dtype=float)
    probabilities /= np.sum(probabilities)
    return np.diag(probabilities.astype(complex)).reshape(-1, order="F")


def estimate_fock_dimensions(
    environment,
    time,
    time_step,
    *,
    population_threshold=1.0e-6,
    average_population_threshold=1.0e-6,
    minimum_dimension=2,
    maximum_dimension=30,
    safety_padding=0,
):
    """Estimate local Fock dimensions and return dimensions plus per-mode diagnostics."""
    if not isinstance(environment, LocalPseudomodes):
        raise TypeError("environment must be a LocalPseudomodes object.")
    if time < 0.0 or time_step <= 0.0:
        raise ValueError("time must be non-negative and time_step must be positive.")
    if time > 0.0 and abs(time / time_step - round(time / time_step)) > 1.0e-12:
        raise ValueError("time must be an integer multiple of time_step.")
    if not 0.0 < population_threshold < 1.0 or not 0.0 < average_population_threshold < 1.0:
        raise ValueError("Population thresholds must lie strictly between zero and one.")
    if int(minimum_dimension) != minimum_dimension or minimum_dimension < 2:
        raise ValueError("minimum_dimension must be an integer of at least two.")
    if int(maximum_dimension) != maximum_dimension or maximum_dimension < minimum_dimension:
        raise ValueError("maximum_dimension must be an integer not smaller than minimum_dimension.")
    if int(safety_padding) != safety_padding or safety_padding < 0:
        raise ValueError("safety_padding must be a non-negative integer.")

    number_of_steps = int(round(time / time_step))
    dimensions = np.zeros(environment.number_of_modes, dtype=int)
    diagnostics = []

    for mode in range(environment.number_of_modes):
        frequency = environment.frequencies[mode]
        thermal_energy = environment.thermal_energies[mode]
        thermal_dimension = 1
        if thermal_energy > 0.0:
            q = float(np.exp(-frequency / thermal_energy))
            thermal_dimension = max(1, int(np.ceil(np.log(population_threshold) / np.log(q))))

        first_dimension = max(int(minimum_dimension), thermal_dimension + 1)
        for tested_dimension in range(first_dimension, int(maximum_dimension) + 1):
            liouvillian = _oscillator_liouvillian(
                frequency,
                environment.damping_rates[mode],
                environment.couplings[mode],
                thermal_energy,
                tested_dimension,
            )
            initial_state = _thermal_state_vector(frequency, thermal_energy, tested_dimension)
            if number_of_steps == 0:
                states = initial_state.reshape(1, -1)
            else:
                states = expm_multiply(
                    liouvillian,
                    initial_state,
                    start=0.0,
                    stop=time,
                    num=number_of_steps + 1,
                    endpoint=True,
                )
            highest_index = tested_dimension**2 - 1
            populations = np.abs(states[:, highest_index])
            maximum_population = float(np.max(populations))
            average_population = float(np.mean(populations))
            if maximum_population < population_threshold and average_population < average_population_threshold:
                recommended = max(tested_dimension - 1 + int(safety_padding), thermal_dimension, 1)
                dimensions[mode] = recommended
                diagnostics.append(
                    {
                        "mode": mode,
                        "thermal_dimension": thermal_dimension,
                        "tested_dimension": tested_dimension,
                        "recommended_dimension": recommended,
                        "maximum_highest_fock_state_population": maximum_population,
                        "average_highest_fock_state_population": average_population,
                    }
                )
                break
        else:
            raise RuntimeError(f"Pseudomode {mode} did not converge up to maximum_dimension={maximum_dimension}.")

    return {"fock_dimensions": dimensions, "mode_results": diagnostics}
