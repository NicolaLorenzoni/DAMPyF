"""
    Ray initialization and runtime resource estimates for DAMPyF.
"""

import os

import psutil
import ray

from .evolution import SYSTEM_UPDATE_RESOURCE
from .utils import BYTES_PER_GIB, estimate_memory_requirements


def measure_computational_resources(config):
    """
        Return the CPUs and memory available to this process.
    """
    if config.execution_mode == "local":
        number_of_cpus = os.cpu_count()
        if number_of_cpus is None:
            raise RuntimeError("Could not identify the local number of CPUs.")
        try:
            available_memory_gib = psutil.virtual_memory().available / BYTES_PER_GIB
        except Exception as error:
            raise RuntimeError("Could not identify the local available memory.") from error
    elif config.execution_mode == "slurm":
        try:
            number_of_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
        except KeyError as error:
            raise ValueError("execution_mode='slurm' requires SLURM_CPUS_PER_TASK.") from error
        except ValueError as error:
            raise ValueError("SLURM_CPUS_PER_TASK must be an integer.") from error

        memory_per_node_mib = os.environ.get("SLURM_MEM_PER_NODE")
        memory_per_cpu_mib = os.environ.get("SLURM_MEM_PER_CPU")
        if memory_per_node_mib:
            try:
                available_memory_gib = int(memory_per_node_mib) / 1024.0
            except ValueError as error:
                raise ValueError("SLURM_MEM_PER_NODE must contain an integer value in MiB.") from error
        elif memory_per_cpu_mib:
            try:
                available_memory_gib = int(memory_per_cpu_mib) * number_of_cpus / 1024.0
            except ValueError as error:
                raise ValueError("SLURM_MEM_PER_CPU must contain an integer value in MiB.") from error
        else:
            raise ValueError("execution_mode='slurm' requires SLURM_MEM_PER_NODE or SLURM_MEM_PER_CPU.")
    else:
        raise ValueError("execution_mode must be either 'local' or 'slurm'.")

    return {"num_cpus": int(number_of_cpus), "available_memory_gib": float(available_memory_gib)}


def initialize_ray(config, pseudomodes, system_index_maps):
    """Check the coarse memory limit, initialize Ray, and return a summary."""
    if ray.is_initialized():
        raise RuntimeError(
            "run_dampf manages its own Ray runtime and requires Ray to be uninitialized. "
            "Finish the existing Ray workload and shut it down explicitly, or run DAMPyF in a separate process."
        )

    resources = measure_computational_resources(config)
    requirements = estimate_memory_requirements(config, pseudomodes)
    requirements["object_store_gib"] = 0.30 * resources["available_memory_gib"]
    available_worker_memory_gib = resources["available_memory_gib"] - requirements["object_store_gib"]

    requested_tasks = system_index_maps.N
    if config.ray_max_parallel_system_updates is not None:
        requested_tasks = min(requested_tasks, int(config.ray_max_parallel_system_updates))

    system_update_tasks_gib = requested_tasks * requirements["one_system_update_task_gib"]
    if system_update_tasks_gib > available_worker_memory_gib:
        maximum_tasks = int(available_worker_memory_gib // requirements["one_system_update_task_gib"])
        if maximum_tasks < 1:
            raise MemoryError(
                f"One system-update task requires an estimated {requirements['one_system_update_task_gib']:.4f} GiB, "
                f"but only {available_worker_memory_gib:.4f} GiB remains after the Ray object store."
            )
        requested_tasks = min(requested_tasks, maximum_tasks)
        system_update_tasks_gib = requested_tasks * requirements["one_system_update_task_gib"]

    summary = {
        "resources": resources,
        "memory_requirements": requirements,
        "ray_num_parallel_system_updates": requested_tasks,
        "available_worker_memory_gib": available_worker_memory_gib,
        "system_update_tasks_gib": system_update_tasks_gib,
    }
    ray.init(
        address="local",
        num_cpus=resources["num_cpus"],
        object_store_memory=int(requirements["object_store_gib"] * BYTES_PER_GIB),
        resources={SYSTEM_UPDATE_RESOURCE: float(requested_tasks)},
        include_dashboard=False,
    )
    return summary


def print_parallelization_summary(config, summary):
    """Print the Ray resource summary with explicit binary-memory units."""
    resources = summary["resources"]
    requirements = summary["memory_requirements"]
    print("Ray parallelization summary")
    print("---------------------------")
    print(f"simulation_mode                    : {config.simulation_mode}")
    print(f"compression_svd_method             : {config.compression_svd_method}")
    print(f"execution_mode                     : {config.execution_mode}")
    print(f"Ray CPUs                           : {resources['num_cpus']}")
    print(f"available memory                   : {resources['available_memory_gib']:10.4f} GiB")
    print(f"Ray object-store reservation       : {requirements['object_store_gib']:10.4f} GiB")
    print(f"estimated Ray worker memory        : {summary['available_worker_memory_gib']:10.4f} GiB")
    print(f"memory required by one MPS         : {requirements['one_mps_gib']:10.4f} GiB")
    print(f"memory required by stored state    : {requirements['stored_state_gib']:10.4f} GiB")
    print(f"one system-update task             : {requirements['one_system_update_task_gib']:10.4f} GiB")
    print(f"ray_max_parallel_system_updates    : {config.ray_max_parallel_system_updates}")
    print(f"simultaneous system-update tasks   : {summary['ray_num_parallel_system_updates']}")
    print(f"all concurrent system-update tasks : {summary['system_update_tasks_gib']:10.4f} GiB")
