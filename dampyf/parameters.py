"""
    Validation and preparation of model parameters for DAMPyF.
"""

import numpy as np

from .structures import LocalPseudomodes, PseudomodeParameters, SystemParameters


def _validated_hamiltonian(hamiltonian):
    """
        Return a Hermitian Hamiltonian array.
    """
    hamiltonian = np.asarray(hamiltonian, dtype=complex)
    if hamiltonian.ndim != 2:
        raise ValueError("System Hamiltonian must be a two-dimensional array.")
    if hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("System Hamiltonian must be square.")
    if hamiltonian.shape[0] == 0:
        raise ValueError("System Hamiltonian must contain at least one site.")
    if not np.all(np.isfinite(hamiltonian)):
        raise ValueError("System Hamiltonian contains non-finite values.")
    if not np.allclose(hamiltonian, hamiltonian.T.conj(), atol=1.0e-10):
        raise ValueError("System Hamiltonian is not Hermitian.")
    return 0.5 * (hamiltonian + hamiltonian.T.conj())


def _validated_dipoles(dipoles, number_of_sites, label):
    """
        Return a validated optional dipole array.
    """
    if dipoles is None:
        return None
    dipoles = np.asarray(dipoles, dtype=complex)
    if dipoles.ndim == 1:
        dipoles = dipoles.reshape(1, -1)
    expected_shape = (int(number_of_sites), 3)
    if dipoles.shape != expected_shape:
        raise ValueError(f"{label} dipoles must have shape {expected_shape}, got {dipoles.shape}.")
    if not np.all(np.isfinite(dipoles)):
        raise ValueError(f"{label} dipoles contain non-finite values.")
    return np.array(dipoles, copy=True)


def create_system_parameters(hamiltonian, electric_dipoles=None, magnetic_dipoles=None):
    """
        Create internal system parameters from in-memory arrays.
    """
    hamiltonian = _validated_hamiltonian(hamiltonian)
    number_of_sites = int(hamiltonian.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return SystemParameters(
        N=number_of_sites,
        Hs=hamiltonian,
        eig_sys=eigenvalues,
        eig_transf_matrix=eigenvectors,
        electric_dipoles=_validated_dipoles(electric_dipoles, number_of_sites, "Electric"),
        magnetic_dipoles=_validated_dipoles(magnetic_dipoles, number_of_sites, "Magnetic"),
    )


def _environments_equal(first, second):
    if first is None or second is None:
        return first is second
    return all(
        np.array_equal(getattr(first, name), getattr(second, name))
        for name in (
            "frequencies",
            "damping_rates",
            "couplings",
            "thermal_energies",
            "fock_dimensions",
        )
    )


def create_pseudomode_parameters(local_environments, number_of_sites):
    """
        Create internal pseudomode parameters from local environment objects.
    """
    number_of_sites = int(number_of_sites)
    if isinstance(local_environments, LocalPseudomodes):
        environments = [local_environments for _ in range(number_of_sites)]
        same_local_environment = True
    else:
        try:
            environments = list(local_environments)
        except TypeError as error:
            raise TypeError("pseudomodes must be a LocalPseudomodes object or a sequence containing one entry per site.") from error
        if len(environments) != number_of_sites:
            raise ValueError(f"Expected {number_of_sites} local pseudomode entries, found {len(environments)}.")
        if any(environment is not None and not isinstance(environment, LocalPseudomodes) for environment in environments):
            raise TypeError("Every local pseudomode entry must be a LocalPseudomodes object or None.")
        same_local_environment = all(_environments_equal(environments[0], environment) for environment in environments[1:])

    if not any(environment is not None for environment in environments):
        raise ValueError("At least one pseudomode must be specified.")

    frequencies = []
    damping_rates = []
    couplings = []
    thermal_energies = []
    fock_dimensions = []
    numbers_of_modes = []

    for environment in environments:
        if environment is None:
            frequencies.append([])
            damping_rates.append([])
            couplings.append([])
            thermal_energies.append([])
            fock_dimensions.append([])
            numbers_of_modes.append(0)
            continue
        frequencies.append(environment.frequencies.tolist())
        damping_rates.append(environment.damping_rates.tolist())
        couplings.append(environment.couplings.tolist())
        thermal_energies.append(environment.thermal_energies.tolist())
        fock_dimensions.append(environment.fock_dimensions.tolist())
        numbers_of_modes.append(environment.number_of_modes)

    return PseudomodeParameters(
        N=number_of_sites,
        Q=numbers_of_modes,
        fock_dim=fock_dimensions,
        Omega_osc=frequencies,
        coupling=couplings,
        drate=damping_rates,
        thermal_energy=thermal_energies,
        same_local_environment=same_local_environment,
    )


def print_pseudomode_summary(pseudomodes):
    """
        Print the pseudomode-parameter summary.
    """
    print("Pseudomode-parameter summary")
    print("----------------------------")
    print(f"N                          : {pseudomodes.N}")
    print(f"same_local_environment     : {pseudomodes.same_local_environment}")
    print(f"Q per site                 : {pseudomodes.Q}")
    print(f"total number of oscillators: {pseudomodes.num_osc}")
    print("fock_dim per site          :")
    for site, fock_dimensions in enumerate(pseudomodes.fock_dim):
        print(f"    site {site}: {fock_dimensions}")
