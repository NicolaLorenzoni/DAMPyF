DAMPF_BANNER = r"""
#-----------------------------------------------------------------------------#
#\         _____        _       __    __   _____             ______          /#
#\        |   _ '.     /.\     |  \  /  | |  __ '.          |  ____]         /#
#\         | | '. \   / _ \    |   \/   | | |__| |  _   __  | |_             /#
#\         | |  | |  / ___ \   | |\  /| | |  ___.' [ '.[  ] |  _]            /#
#\         | |_.' / / /   \ \  | | \/ | | | |       \ \/ /  | |              /#
#\        |_____.' /_/     \_\ |_|    |_| |_|     ___\  /   |_|              /#
#\                                               [_____/                     /#
#\                                                                           /#
#\                                                                           /#
#\                 Python implementation of the DAMPF method                 /#
#\                        Developed by Nicola Lorenzoni                      /#
#\                             ITP Ulm University                            /#
#\                               Version: 1.0                                /#
#\                      Contributors: Nicola Lorenzoni                       /#
#\                                                                           /#
#-----------------------------------------------------------------------------#
"""

import time                 as timelib
import numpy                as np
import configure_simulation as p

from dampf_modules.dampf_basis           import prepare_basis
from dampf_modules.dampf_load_data       import load_system_parameters, load_pseudomode_parameters
from dampf_modules.dampf_evolution       import evolve
from dampf_modules.dampf_initial_state   import initialize_state_refs
from dampf_modules.dampf_model           import prepare_ray_operator_refs
from dampf_modules.dampf_observables     import reduced_system_density_matrix, optical_coherence_matrix
from dampf_modules.dampf_parallelization import initialize_ray, shutdown_ray
from dampf_modules.dampf_storage         import save_system_data, save_simulation_data, save_checkpoint, maximum_state_bond_dimension
from dampf_modules.dampf_structures      import generate_system_index_maps
from dampf_modules.dampf_utils           import check_user_parameters, print_summaries


def run_propagation(system, pseudomodes, system_index_maps, operator_refs, state_refs, rho_sys_initial):
    """
        Run the DAMPF propagation and save output data.
    """
    # Initialize time and system density matrix data collection
    times = []
    system_density_matrix_list = []
    # Initialize backup time clock
    backup_clock = timelib.perf_counter()

    num_data_steps = int(round(p.time / p.dtdata))
    maximum_bond_dimension_reached = 1

    for td in range(num_data_steps + 1):
        progress_message = f"Timestep {td} of {num_data_steps}"
        print(f"\r{progress_message:<100}", end="", flush=True)
    
        normalization = 1.0
        
        # Energy-transfer data collection
        if p.simulation_mode == "energy_transfer":
            system_density_matrix = reduced_system_density_matrix(state_refs, pseudomodes, system_index_maps, p)
            normalization = np.trace(system_density_matrix)
            if abs(normalization) == 0.0:
                raise ValueError("Cannot normalize with zero system trace.")
            system_density_matrix = system_density_matrix / normalization
        # Absorption data collection
        else:
            system_density_matrix = optical_coherence_matrix(state_refs, pseudomodes)
    
        times.append(td * p.dtdata)
        system_density_matrix_list.append(system_density_matrix)
    
        if td == num_data_steps:
            break
    
        step_clock = timelib.perf_counter()
        # Evolution between data collection steps
        state_refs, interval_maximum_bond_dimension = evolve(state_refs, p, operator_refs, system_index_maps, normalization)
        maximum_bond_dimension_reached = max(maximum_bond_dimension_reached, interval_maximum_bond_dimension)
        # Printing computational time required for the evolution step
        elapsed_time = timelib.perf_counter() - step_clock
        progress_message = f"Timestep {td + 1} of {num_data_steps} | time per output step: {elapsed_time:.6f} s"
        print(f"\r{progress_message:<100}", end="", flush=True)

        # Backup
        if timelib.perf_counter() - backup_clock >= p.backuptime_seconds:
            save_system_data(times, system_density_matrix_list, p)
            save_checkpoint(state_refs, td + 1, p)
            backup_clock = timelib.perf_counter()

    print()
    # Data storage
    system_output_file = save_system_data(times, system_density_matrix_list, p)
    final_maximum_bond_dimension = maximum_state_bond_dimension(state_refs, p)
    maximum_bond_dimension_reached = max(maximum_bond_dimension_reached, final_maximum_bond_dimension)

    return system_output_file, final_maximum_bond_dimension, maximum_bond_dimension_reached

def dampf():
    """
        Initialize and run the DAMPF algorithm.
    """
    print(DAMPF_BANNER)

    # Load system and pseudomode parameters.
    system  = load_system_parameters(p)
    pseudomodes = load_pseudomode_parameters(p, system.N)

    # Identify relevant system indices and check user parameters.
    system_index_maps = generate_system_index_maps(system.N)
    check_user_parameters(p, system, system_index_maps)

    # Prepare the local basis.
    basis_data = prepare_basis(pseudomodes)

    # Initialize Ray.
    parallelization_summary = initialize_ray(p, pseudomodes, system_index_maps)

    operator_refs = prepare_ray_operator_refs(p, system, pseudomodes, basis_data, system_index_maps)

    state_refs, rho_sys_initial = initialize_state_refs(p, system, pseudomodes, system_index_maps, basis_data)

    # Print initial summary.
    print_summaries(p, system, pseudomodes, system_index_maps, operator_refs, rho_sys_initial, parallelization_summary)

    print()
    print("Starting DAMPF propagation")
    print("--------------------------")
    start = timelib.perf_counter()
    try:
        system_output_file, final_maximum_bond_dimension, maximum_bond_dimension_reached = run_propagation(
            system, pseudomodes, system_index_maps, operator_refs, state_refs, rho_sys_initial
        )
    finally:
        shutdown_ray()
    elapsed_time_seconds = timelib.perf_counter() - start
    save_simulation_data(
        p,
        system,
        pseudomodes,
        rho_sys_initial,
        system_output_file,
        elapsed_time_seconds,
        final_maximum_bond_dimension,
        maximum_bond_dimension_reached,
    )
    print(f"Total time: {elapsed_time_seconds:.6f} s")

    return


if __name__ == "__main__":
    dampf()
