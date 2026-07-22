"""
    Estimate the memory required by the current DAMPyF configuration.
"""

from pathlib import Path
import os
import sys


# =============================================================================
# Locate project and import DAMPyF modules
# =============================================================================
script_folder = Path(__file__).resolve().parent
project_folder = script_folder.parent

sys.path.insert(0, str(project_folder))
os.chdir(project_folder)

import configure_simulation as p

from dampf_modules.dampf_load_data import load_system_parameters
from dampf_modules.dampf_load_data import load_pseudomode_parameters
from dampf_modules.dampf_structures import generate_system_index_maps
from dampf_modules.dampf_utils import check_user_parameters


# =============================================================================
# Helper functions
# =============================================================================

def print_memory_line(label, value):
    """
        Print one formatted memory line.
    """
    print(f"{label:<55}: {value:10.4f} GB")
    return


def estimate_one_mps_memory_gb(params, pseudomodes):
    """
        Estimate the memory required by one MPS.

        The estimate assumes that each local tensor has bond dimension BD
        and local physical dimension fock_dim^2.
    """
    bytes_for_complex = 16
    pseudomode_dimension_factor = 0

    for n in range(pseudomodes.N):
        for q in range(pseudomodes.Q[n]):
            pseudomode_dimension_factor += int(pseudomodes.fock_dim[n][q]) ** 2

    memory_required_one_mps = bytes_for_complex * int(params.BD) ** 2 * pseudomode_dimension_factor / 1024**3

    return memory_required_one_mps


def estimate_one_rank_one_mpo_memory_gb(pseudomodes):
    """
        Estimate the memory required by one rank-1 MPO.

        A local MPO tensor maps vectorized local density matrices to
        vectorized local density matrices. Therefore, for one pseudomode
        with Fock dimension d, the physical contribution scales as d^4.
    """
    bytes_for_complex = 16
    pseudomode_mpo_dimension_factor = 0

    for n in range(pseudomodes.N):
        for q in range(pseudomodes.Q[n]):
            pseudomode_mpo_dimension_factor += int(pseudomodes.fock_dim[n][q]) ** 4

    memory_required_one_rank_one_mpo = bytes_for_complex * pseudomode_mpo_dimension_factor / 1024**3

    return memory_required_one_rank_one_mpo


def estimate_num_stored_components(params, system, system_index_maps):
    """
        Estimate the number of system MPS components stored by DAMPyF.
    """
    if params.simulation_mode == "energy_transfer":
        stored_components = int(system_index_maps.num_stored_indices)
    else:
        stored_components = int(system.N) ** 2

    return stored_components


def estimate_num_local_update_mpos(params, system, system_index_maps):
    """
        Estimate the number of local-update MPOs stored in Ray.
    """
    if params.simulation_mode == "energy_transfer":
        num_local_mpos = int(system_index_maps.num_stored_indices)
    else:
        num_local_mpos = int(system.N)

    if params.trotter_order == "first":
        return num_local_mpos

    return 2 * num_local_mpos


def estimate_memory_requirements(params, system, pseudomodes, system_index_maps):
    """
        Estimate memory requirements.
    """
    N = int(system.N)

    memory_required_one_mps = estimate_one_mps_memory_gb(params, pseudomodes)
    memory_required_one_rank_one_mpo = estimate_one_rank_one_mpo_memory_gb(pseudomodes)

    stored_components = estimate_num_stored_components(params, system, system_index_maps)
    num_local_update_mpos = estimate_num_local_update_mpos(params, system, system_index_maps)

    memory_required_stored_state = stored_components * memory_required_one_mps

    memory_required_local_update_mpos = num_local_update_mpos * memory_required_one_rank_one_mpo
    memory_required_adjoint_mpo = memory_required_one_rank_one_mpo
    memory_required_mpos = memory_required_local_update_mpos + memory_required_adjoint_mpo

    # One local-update task applies one MPO to one MPS.
    # The factor 2 accounts, approximately, for input/output/intermediate MPS-like data.
    memory_required_one_local_update_task = 2.0 * memory_required_one_mps + memory_required_one_rank_one_mpo

    # All stored system components may be updated locally at the same time.
    memory_required_all_local_update_tasks = stored_components * memory_required_one_local_update_task

    # System-update task estimate.
    # One system-update task builds intermediate combinations involving all N sites.
    memory_required_one_system_update_task = 3.0 * N * memory_required_one_mps

    # All N system-update tasks may be updated at once, unless DAMPyF reduces
    # their parallel execution because of memory limitations.
    memory_required_all_system_update_tasks = N * memory_required_one_system_update_task

    memory_required_peak_parallel_tasks = max(
        memory_required_all_local_update_tasks,
        memory_required_all_system_update_tasks,
    )

    memory_required_total = memory_required_stored_state + memory_required_mpos + memory_required_peak_parallel_tasks

    return {
        "memory_required_total": memory_required_total,
        "memory_required_one_mps": memory_required_one_mps,
        "memory_required_stored_state": memory_required_stored_state,
        "memory_required_one_rank_one_mpo": memory_required_one_rank_one_mpo,
        "memory_required_local_update_mpos": memory_required_local_update_mpos,
        "memory_required_adjoint_mpo": memory_required_adjoint_mpo,
        "memory_required_mpos": memory_required_mpos,
        "memory_required_one_local_update_task": memory_required_one_local_update_task,
        "memory_required_all_local_update_tasks": memory_required_all_local_update_tasks,
        "memory_required_one_system_update_task": memory_required_one_system_update_task,
        "memory_required_all_system_update_tasks": memory_required_all_system_update_tasks,
        "memory_required_peak_parallel_tasks": memory_required_peak_parallel_tasks,
        "stored_components": stored_components,
        "num_local_update_mpos": num_local_update_mpos,
    }


# =============================================================================
# Main script
# =============================================================================

system = load_system_parameters(p)
pseudomodes = load_pseudomode_parameters(p, system.N)
system_index_maps = generate_system_index_maps(system.N)
check_user_parameters(p, system, system_index_maps)

memory_requirements = estimate_memory_requirements(
    p,
    system,
    pseudomodes,
    system_index_maps,
)

print_memory_line(
    "Total estimated memory requirement",
    memory_requirements["memory_required_total"],
)
print()

print("DAMPyF memory estimate")
print("======================")
print(f"simulation_mode                                      : {p.simulation_mode}")
print(f"N                                                    : {system.N}")
print(f"Q per site                                           : {pseudomodes.Q}")
print(f"total number of pseudomodes                          : {pseudomodes.num_osc}")
print(f"BD                                                   : {p.BD}")
print(f"trotter_order                                        : {p.trotter_order}")
print(f"compression_svd_method                               : {p.compression_svd_method}")
print()

print("MPS/state estimates")
print("-------------------")
print(f"stored system components                         : {memory_requirements['stored_components']}")
print_memory_line("memory required for one MPS", memory_requirements["memory_required_one_mps"])
print_memory_line("memory required for stored state", memory_requirements["memory_required_stored_state"])
print()

print("MPO estimates")
print("-------------")
print(f"number of local-update MPOs                          : {memory_requirements['num_local_update_mpos']}")
print_memory_line("memory required for local-update MPOs", memory_requirements["memory_required_local_update_mpos"])
print_memory_line("memory required for adjoint MPO", memory_requirements["memory_required_adjoint_mpo"])
print_memory_line("memory required for MPOs", memory_requirements["memory_required_mpos"])
print()

print("Parallel-task estimates")
print("-----------------------")
print_memory_line("memory required for one local-update task", memory_requirements["memory_required_one_local_update_task"])
print_memory_line("memory required for all local-update tasks", memory_requirements["memory_required_all_local_update_tasks"])
print_memory_line("memory required for one system-update task", memory_requirements["memory_required_one_system_update_task"])
print_memory_line("memory required for all system-update tasks", memory_requirements["memory_required_all_system_update_tasks"])
print_memory_line("peak parallel-task memory", memory_requirements["memory_required_peak_parallel_tasks"])
print()

print("Notes")
print("-----")
print("These are order-of-magnitude estimates, not exact memory-profiler measurements.")
print("DAMPyF automatically reduces the number of system-update tasks that can")
print("run in parallel if the memory required for all system-update tasks exceeds")
print("the available memory.")
print("Actual memory can be larger because of temporary arrays, SVD workspaces, Ray")
print("serialization, NumPy/SciPy overhead, and filesystem/checkpoint buffers.")