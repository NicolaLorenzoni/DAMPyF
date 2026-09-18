"""
    Importable top-level execution routines for DAMPyF.
"""

import time as timelib

import numpy as np
import ray

from .basis import prepare_basis
from .evolution import evolve
from .initial_state import initialize_state_refs
from .model import prepare_ray_operator_refs
from .observables import optical_coherence_matrix, reduced_system_density_matrix
from .parallelization import initialize_ray
from .storage import (
    maximum_state_bond_dimension,
    save_checkpoint,
    save_simulation_data,
    save_system_data,
)
from .structures import DampfResult
from .utils import check_user_parameters, generate_system_index_maps, print_summaries

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
#\                      Contributors: Nicola Lorenzoni                       /#
#\                                                                           /#
#-----------------------------------------------------------------------------#
"""


def run_propagation(config, pseudomodes, system_index_maps, operator_refs, state_refs, output_directory, output_identifier, verbose):
    """
        Propagate one prepared state and optionally save data and checkpoints.
    """
    times = []
    system_data_list = []
    checkpoint_clock = timelib.perf_counter()
    number_of_data_steps = int(round(config.time / config.data_time_step))
    maximum_bond_dimension_reached = 1

    for data_step in range(number_of_data_steps + 1):
        if verbose and data_step == 0:
            print(f"\rTimestep {data_step} of {number_of_data_steps}", end="", flush=True)

        normalization = 1.0
        if config.simulation_mode == "energy_transfer":
            system_data = reduced_system_density_matrix(state_refs, pseudomodes, system_index_maps)
            normalization = np.trace(system_data)
            if abs(normalization) == 0.0:
                raise ValueError("Cannot normalize with zero system trace.")
            system_data = system_data / normalization
        else:
            system_data = optical_coherence_matrix(state_refs, pseudomodes)

        times.append(data_step * config.data_time_step)
        system_data_list.append(system_data)
        if data_step == number_of_data_steps:
            break

        step_clock = timelib.perf_counter()
        state_refs, interval_maximum = evolve(state_refs, config, operator_refs, system_index_maps, normalization)
        maximum_bond_dimension_reached = max(maximum_bond_dimension_reached, interval_maximum)
        elapsed_time = timelib.perf_counter() - step_clock
        if verbose:
            print(f"\rTimestep {data_step + 1}/{number_of_data_steps} | {elapsed_time:.6f} s", end="", flush=True)

        checkpoint_interval = config.checkpoint_interval_seconds
        if checkpoint_interval is not None and timelib.perf_counter() - checkpoint_clock >= checkpoint_interval:
            save_system_data(times, system_data_list, config, output_directory, output_identifier)
            save_checkpoint(state_refs, data_step + 1, config, output_directory, output_identifier)
            checkpoint_clock = timelib.perf_counter()

    if verbose:
        print()

    system_output_file = None
    if output_directory is not None:
        system_output_file = save_system_data(times, system_data_list, config, output_directory, output_identifier)

    final_maximum_bond_dimension = maximum_state_bond_dimension(state_refs, config)
    maximum_bond_dimension_reached = max(maximum_bond_dimension_reached, final_maximum_bond_dimension)
    return (
        np.asarray(times, dtype=float),
        np.asarray(system_data_list, dtype=complex),
        system_output_file,
        final_maximum_bond_dimension,
        maximum_bond_dimension_reached,
    )


def run_dampf(config, system, pseudomodes, *, initial_density_matrix, restart_from, output_directory, output_identifier, verbose):
    """
        Initialize and run DAMPyF using prepared in-memory model data.
    """
    if verbose:
        print(DAMPF_BANNER)

    system_index_maps = generate_system_index_maps(system.N)
    check_user_parameters(
        config,
        system,
        initial_density_matrix,
        restart_from,
        output_directory,
        output_identifier,
    )
    basis_data = prepare_basis(pseudomodes)
    parallelization_summary = initialize_ray(config, pseudomodes, system_index_maps)
    start = timelib.perf_counter()

    try:
        operator_refs = prepare_ray_operator_refs(config, system, pseudomodes, basis_data, system_index_maps)
        state_refs, prepared_initial_density_matrix = initialize_state_refs(
            config,
            system,
            pseudomodes,
            system_index_maps,
            basis_data,
            initial_density_matrix,
            restart_from,
        )
        if verbose:
            print_summaries(
                config,
                system,
                pseudomodes,
                system_index_maps,
                prepared_initial_density_matrix,
                restart_from,
                parallelization_summary,
            )
            print("Starting DAMPF propagation")
            print("--------------------------")

        times, data, system_output_file, final_bond_dimension, maximum_bond_dimension = run_propagation(
            config,
            pseudomodes,
            system_index_maps,
            operator_refs,
            state_refs,
            output_directory,
            output_identifier,
            verbose,
        )
    finally:
        if ray.is_initialized():
            ray.shutdown()

    elapsed_time_seconds = timelib.perf_counter() - start
    simulation_data_file = None
    if output_directory is not None:
        simulation_data_file = save_simulation_data(
            config,
            system,
            pseudomodes,
            prepared_initial_density_matrix,
            restart_from,
            system_output_file,
            elapsed_time_seconds,
            final_bond_dimension,
            maximum_bond_dimension,
            output_directory,
            output_identifier,
        )
    if verbose:
        print(f"Total time: {elapsed_time_seconds:.6f} s")

    return DampfResult(
        times=times,
        rho_system=data if config.simulation_mode == "energy_transfer" else None,
        optical_coherence=data if config.simulation_mode == "linear_spectra" else None,
        system_output_file=system_output_file,
        simulation_data_file=simulation_data_file,
        elapsed_time_seconds=elapsed_time_seconds,
        final_maximum_bond_dimension=final_bond_dimension,
        maximum_bond_dimension_reached=maximum_bond_dimension,
    )
