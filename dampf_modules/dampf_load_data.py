"""
    Load input system and pseudomode data for DAMPF simulations.
"""

import os
import numpy as np

from .dampf_structures import SystemParameters, PseudomodeParameters

# Input folders for system and pseudomode parameters
SYSTEM_INPUT_FOLDER = "./input_data/system"
PSEUDOMODE_INPUT_FOLDER = "./input_data/pseudomodes"


def read_system_hamiltonian_file(filename):
    """
        Load the site-basis system Hamiltonian from file.
    """
    path = SYSTEM_INPUT_FOLDER + "/" + str(filename)

    # Consistency check.
    if not os.path.isfile(path):
        raise FileNotFoundError(f"System Hamiltonian file not found: {path}")

    Hs = np.loadtxt(path, dtype=float)

    # Consistency check.
    if Hs.ndim != 2:
        raise ValueError("System Hamiltonian must be a two-dimensional array.")
    if Hs.shape[0] != Hs.shape[1]:
        raise ValueError("System Hamiltonian must be square.")
    if not np.allclose(Hs, Hs.T.conj(), atol=1.0e-10):
        raise ValueError("System Hamiltonian is not Hermitian/symmetric.")

    return 0.5 * (Hs + Hs.T.conj())


def read_dipoles_file(filename, N, label):
    """
        Load optional site dipoles from file.
    """
    if filename is None:
        return None

    path = SYSTEM_INPUT_FOLDER + "/" + str(filename)

    # Consistency check.
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} dipoles file not found: {path}")

    dipoles = np.loadtxt(path, dtype=float)

    # Consistency check.
    if dipoles.shape != (N, 3):
        raise ValueError(f"{label} dipoles must have shape ({N}, 3), got {dipoles.shape}.")

    return dipoles


def diagonalize_system_hamiltonian(Hs):
    """
        Diagonalize Hs, sort eigenstates by increasing energy and identify eigenbasis transformation.
    """
    eig_sys, eig_transf_matrix = np.linalg.eigh(Hs)
    order = np.argsort(eig_sys)
    
    return eig_sys[order], eig_transf_matrix[:, order]


def load_system_parameters(params):
    """
        Collect the system parameters.
    """
    Hs = read_system_hamiltonian_file(params.system_hamiltonian_file)
    
    N = int(Hs.shape[0])
    eig_sys, eig_transf_matrix = diagonalize_system_hamiltonian(Hs)
    
    electric_dipoles_file = getattr(params, "electric_dipoles_file", getattr(params, "dipoles_file", None))
    magnetic_dipoles_file = getattr(params, "magnetic_dipoles_file", None)

    electric_dipoles = read_dipoles_file(electric_dipoles_file, N, "Electric")
    magnetic_dipoles = read_dipoles_file(magnetic_dipoles_file, N, "Magnetic")

    return SystemParameters(
        N=N,
        Hs=Hs,
        eig_sys=eig_sys,
        eig_transf_matrix=eig_transf_matrix,
        electric_dipoles=electric_dipoles,
        magnetic_dipoles=magnetic_dipoles,
    )


def read_pseudomode_file(filename):
    """
        Load pseudomode parameters from file.
    """
    path = PSEUDOMODE_INPUT_FOLDER + "/" + str(filename)

    # Consistency check.
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pseudomode file not found: {path}")

    omega = []
    drate = []
    coupling = []
    thermal_energy = []
    fock_dim = []

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        parts = stripped.split()

        # Consistency check.
        if len(parts) < 5:
            raise ValueError(f"Pseudomode files must have five columns: frequency, damping rate, coupling, thermal energy, and Fock dimension. Could not read line in {path}: {line!r}")

        omega_value = float(parts[0])
        drate_value = float(parts[1])
        coupling_value = complex(parts[2])
        thermal_energy_value = float(parts[3])
        fock_value_float = float(parts[4])

        # Consistency check.
        if not np.isfinite(omega_value) or omega_value == 0.0:
            raise ValueError(f"Invalid frequency={omega_value} in {path}. Frequencies must be real, finite, and nonzero.")
        if not np.isfinite(drate_value) or drate_value <= 0.0:
            raise ValueError(f"Invalid damping_rate={drate_value} in {path}. Damping rates must be real, finite, and strictly positive.")
        if not np.isfinite(coupling_value):
            raise ValueError(f"Invalid coupling={coupling_value} in {path}. Couplings must be finite.")
        if not np.isfinite(thermal_energy_value) or thermal_energy_value < 0.0:
            raise ValueError(f"Invalid thermal_energy={thermal_energy_value} in {path}. Thermal energies must be non-negative.")
        if omega_value < 0.0 and thermal_energy_value > 0.0:
            raise ValueError(f"Invalid thermal_energy={thermal_energy_value} for negative frequency={omega_value} in {path}. Negative-frequency pseudomodes must have zero thermal energy.")
        if not np.isfinite(fock_value_float) or int(fock_value_float) != fock_value_float or int(fock_value_float) <= 0:
            raise ValueError(f"Invalid fock_dim={fock_value_float} in {path}. fock_dim must be a positive integer.")

        omega.append(omega_value)
        drate.append(drate_value)
        coupling.append(coupling_value)
        thermal_energy.append(thermal_energy_value)
        fock_dim.append(int(fock_value_float))

    # Consistency check.
    if len(omega) == 0:
        raise ValueError(f"No pseudomodes were read from {path}")

    return {
        "Omega_osc": omega,
        "drate": drate,
        "coupling": coupling,
        "thermal_energy": thermal_energy,
        "fock_dim": fock_dim,
    }


def copy_pseudomode_parameters(source_parameters):
    """
        Return an independent copy of a same pseudomode parameter set.
    """
    return {
        "Omega_osc": list(source_parameters["Omega_osc"]),
        "drate": list(source_parameters["drate"]),
        "coupling": list(source_parameters["coupling"]),
        "thermal_energy": list(source_parameters["thermal_energy"]),
        "fock_dim": list(source_parameters["fock_dim"]),
    }


def load_pseudomode_parameters(params, N):
    """
        Collect the pseudomode parameters.
    """
    N = int(N)
    same = bool(params.same_local_environment)

    if same:
        # Consistency check.
        if params.pseudomode_parameter_file is None:
            raise ValueError(
                "pseudomode_parameter_file must be specified when same_local_environment=True."
            )

        shared_parameters = read_pseudomode_file(params.pseudomode_parameter_file)
        local_parameters = [copy_pseudomode_parameters(shared_parameters) for _ in range(N)]
        source_files = [PSEUDOMODE_INPUT_FOLDER + "/" + str(params.pseudomode_parameter_file) for _ in range(N)]
    else:
        # Consistency check.
        if params.pseudomode_parameter_files is None:
            raise ValueError(
                "pseudomode_parameter_files must be specified when same_local_environment=False."
            )

        files = list(params.pseudomode_parameter_files)

        # Consistency check.
        if len(files) != N:
            raise ValueError(f"same_local_environment=False requires {N} entries, found {len(files)}.")

        local_parameters = []
        source_files = []

        for filename in files:
            if filename is None:
                local_parameters.append(
                    {
                        "Omega_osc": [],
                        "drate": [],
                        "coupling": [],
                        "thermal_energy": [],
                        "fock_dim": [],
                    }
                )
                source_files.append(None)
            else:
                local_parameters.append(read_pseudomode_file(filename))
                source_files.append(PSEUDOMODE_INPUT_FOLDER + "/" + str(filename))

    Q = []
    fock_dim = []
    Omega_osc = []
    coupling = []
    drate = []
    thermal_energy = []

    for n in range(N):
        qn = len(local_parameters[n]["Omega_osc"])
        Q.append(qn)
        fock_dim.append(list(local_parameters[n]["fock_dim"]))
        Omega_osc.append(list(local_parameters[n]["Omega_osc"]))
        coupling.append(list(local_parameters[n]["coupling"]))
        drate.append(list(local_parameters[n]["drate"]))
        thermal_energy.append(list(local_parameters[n]["thermal_energy"]))

        # Consistency check.
        if len(fock_dim[n]) != Q[n]:
            raise ValueError(f"Site {n}: fock_dim has length {len(fock_dim[n])}, but Q={Q[n]}.")

    # Consistency check.
    if sum(Q) == 0:
        raise ValueError(
            "At least one pseudomode must be specified. "
            "Completely environment-free simulations are not currently supported."
        )

    return PseudomodeParameters(N=N, Q=Q, fock_dim=fock_dim, Omega_osc=Omega_osc, coupling=coupling, drate=drate, thermal_energy=thermal_energy, source_files=source_files, same_local_environment=same)


def print_pseudomode_summary(pseudomodes):
    """
        Print the pseudomode parameters summary.
    """
    print("Pseudomode-parameter summary")
    print("----------------------------")
    print(f"N                          : {pseudomodes.N}")
    print(f"same_local_environment     : {pseudomodes.same_local_environment}")
    print(f"Q per site                 : {pseudomodes.Q}")
    print(f"total number of oscillators: {pseudomodes.num_osc}")
    print("fock_dim per site          :")
    for n, fock_dim_site in enumerate(pseudomodes.fock_dim):
        print(f"    site {n}: {fock_dim_site}")

    return
