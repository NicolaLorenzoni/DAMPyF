"""
Ray parallelization helpers for DAMPF.
"""

import os
import psutil
import ray

from .dampf_evolution import SYSTEM_UPDATE_RESOURCE


def measure_computational_resources(params):
    """
        Measure CPU and memory resources available.
    """
    if params.execution_mode == "local":
        num_cpus = os.cpu_count()
        if num_cpus is None:
            raise RuntimeError("Could not identify the local number of CPUs.")

        try:
            available_memory = psutil.virtual_memory().available / 1024**3
        except Exception as error:
            raise RuntimeError("Could not identify the local available memory.") from error
    # If the code is launched via Slurm, the resources are taken from those specified in the batch script
    elif params.execution_mode == "slurm":
        try:
            num_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
        except KeyError as error:
            raise ValueError("execution_mode = 'slurm' requires SLURM_CPUS_PER_TASK.") from error
        except ValueError as error:
            raise ValueError("SLURM_CPUS_PER_TASK must be an integer.") from error

        value = os.environ.get("SLURM_MEM_PER_NODE")
        if value is not None and value.strip() != "":
            try:
                available_memory = int(value) / 1024.0
            except ValueError as error:
                raise ValueError("SLURM_MEM_PER_NODE must be an integer.") from error
        else:
            value = os.environ.get("SLURM_MEM_PER_CPU")
            if value is None or value.strip() == "":
                raise ValueError("execution_mode = 'slurm' requires SLURM_MEM_PER_NODE or SLURM_MEM_PER_CPU.")

            try:
                available_memory = int(value) * num_cpus / 1024.0
            except ValueError as error:
                raise ValueError("SLURM_MEM_PER_CPU must be an integer.") from error

    else:
        raise ValueError("execution_mode must be either 'local' or 'slurm'.")

    return {
        "num_cpus": int(num_cpus),
        "available_memory": float(available_memory),
    }


def estimate_memory_requirements(params, pseudomodes, system_index_maps, resources):
    """
        Estimate DAMPF memory requirements in GB.
    """
    bytes_for_complex = 16
    pseudomode_dimension_factor = 0

    for n in range(pseudomodes.N):
        for q in range(pseudomodes.Q[n]):
            pseudomode_dimension_factor += int(pseudomodes.fock_dim[n][q]) ** 2

    memory_required_one_mps = bytes_for_complex * (int(params.BD)**2) * pseudomode_dimension_factor / 1024**3

    if params.simulation_mode == "energy_transfer":
        stored_components = int(system_index_maps.num_stored_indices)
    else:
        stored_components = int(system_index_maps.N) ** 2

    memory_required_stored_state = stored_components * memory_required_one_mps
    memory_required_system_update_task = 3 * int(pseudomodes.N) * memory_required_one_mps
    object_store_memory = min(0.30 * resources["available_memory"], 200.0)

    return {
        "object_store_memory": object_store_memory,
        "memory_required_one_mps": memory_required_one_mps,
        "memory_required_stored_state": memory_required_stored_state,
        "memory_required_system_update_task": memory_required_system_update_task,
        "stored_components": stored_components,
    }


def initialize_ray(params, pseudomodes, system_index_maps):
    """
        Check memory requirements and start Ray.
    """
    if ray.is_initialized():
        ray.shutdown()

    resources = measure_computational_resources(params)
    memory_requirements = estimate_memory_requirements(params, pseudomodes, system_index_maps, resources)
    available_worker_memory = resources["available_memory"] - memory_requirements["object_store_memory"]

    # Compute eventual limitation on the number of parallel tasks with respect the system update (heaviest operation in the code)
    if params.ray_max_parallel_system_updates is None:
        ray_num_parallel_system_updates = int(system_index_maps.N)
    else:
        ray_num_parallel_system_updates = min(int(system_index_maps.N), int(params.ray_max_parallel_system_updates))

    memory_required_system_updates = int(ray_num_parallel_system_updates) * memory_requirements["memory_required_system_update_task"]

    if memory_required_system_updates > available_worker_memory:
        if memory_requirements["object_store_memory"] >= resources["available_memory"]:
            raise MemoryError(
                "The estimated Ray object-store memory is larger than or equal to the available memory: "
                f"object_store_memory = {memory_requirements['object_store_memory']:.4f} GB, "
                f"available_memory = {resources['available_memory']:.4f} GB."
            )

        max_parallel_from_memory = int(available_worker_memory // memory_requirements["memory_required_system_update_task"])
        if max_parallel_from_memory < 1:
            raise MemoryError(
                "A single system-update task is estimated to require "
                f"{memory_requirements['memory_required_system_update_task']:.4f} GB, but only "
                f"{available_worker_memory:.4f} GB are estimated to be available after Ray object-store memory."
            )

        ray_num_parallel_system_updates = min(ray_num_parallel_system_updates, max_parallel_from_memory)
        memory_required_system_updates = int(ray_num_parallel_system_updates) * memory_requirements["memory_required_system_update_task"]

    parallelization_summary = {
        "resources": resources,
        "memory_requirements": memory_requirements,
        "ray_num_parallel_system_updates": ray_num_parallel_system_updates,
        "available_worker_memory": available_worker_memory,
        "memory_required_system_updates": memory_required_system_updates,
    }

    ray.init(
        num_cpus=resources["num_cpus"],
        object_store_memory=int(memory_requirements["object_store_memory"] * 1024**3),
        resources={SYSTEM_UPDATE_RESOURCE: float(ray_num_parallel_system_updates)},
    )

    return parallelization_summary


def print_parallelization_summary(params, parallelization_summary):
    """
        Print the Ray parallelization summary.
    """
    resources = parallelization_summary["resources"]
    memory_requirements = parallelization_summary["memory_requirements"]

    print("Ray parallelization summary")
    print("---------------------------")
    print(f"simulation_mode                       : {params.simulation_mode}")
    print(f"compression_svd_method                : {params.compression_svd_method}")
    print(f"execution_mode                        : {params.execution_mode}")
    print(f"Ray CPUs                              : {resources['num_cpus']}")
    print(f"available memory                      : {resources['available_memory']:10.4f} GB")
    print(f"estimated Ray object-store memory     : {memory_requirements['object_store_memory']:10.4f} GB")
    print(f"estimated Ray worker memory           : {parallelization_summary['available_worker_memory']:10.4f} GB")
    print(f"memory_required_one_mps               : {memory_requirements['memory_required_one_mps']:10.4f} GB")
    print(f"memory_required_stored_state          : {memory_requirements['memory_required_stored_state']:10.4f} GB")
    print(f"memory_required_system_update_task       : {memory_requirements['memory_required_system_update_task']:10.4f} GB")
    print(f"ray_max_parallel_system_updates   : {params.ray_max_parallel_system_updates}")
    print(f"simultaneous system-update tasks         : {int(parallelization_summary['ray_num_parallel_system_updates'])}")
    print(f"memory_required_system_updates    : {parallelization_summary['memory_required_system_updates']:10.4f} GB")

    return


def shutdown_ray():
    """
    Shut down Ray.
    """
    if ray.is_initialized():
        ray.shutdown()

    return
