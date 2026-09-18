#!/bin/bash

##### RESOURCES ALLOCATION (SLURM) #############
### THESE ARE THE EDITABLE PARAMETERS ###########
# In here we give the job a name
#SBATCH --job-name=DAMPyF
#
# In here we specify the required nodes
#SBATCH --nodes=1
#
# In here we specify the tasks per node. Keep this equal to 1.
#SBATCH --ntasks-per-node=1
#
# In here we specify the number of cores for the task.
# These cores are used by Ray workers.
#SBATCH --cpus-per-task=48
#
# In here we specify the memory per node.
#SBATCH --mem=250G
#
# Send mail when job begins, aborts, and ends.
#SBATCH --mail-type=ALL
#
# In here we specify the required time.
#SBATCH --time=100:00:00
###############################################


#### LOADING ENVIRONMENT CONTAINING RAY #################
# Load modules or your own conda environment here.
source ~/.bashrc
conda activate DAMPyF_environment
#########################################################


#### NUMERICAL THREADING SETTINGS ########################
# Each Ray worker should use one BLAS/OpenMP thread.
# Otherwise 48 Ray workers could each spawn 48 threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
#########################################################


#### DISPLAYING INFO (SAVED IN SLURM-JOBID.OUT) ##########
echo "Submit Directory:                     ${SLURM_SUBMIT_DIR}"
echo "Working Directory:                    ${PWD}"
echo "Running on host:                      ${HOSTNAME}"
echo "Job id:                               ${SLURM_JOB_ID}"
echo "Job name:                             ${SLURM_JOB_NAME}"
echo "Number of nodes allocated to job:     ${SLURM_JOB_NUM_NODES}"
echo "Number of tasks:                      ${SLURM_NTASKS}"
echo "CPUs per task:                        ${SLURM_CPUS_PER_TASK}"
if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then
    echo "Memory per node:                      $(awk "BEGIN {printf \"%.4f\", ${SLURM_MEM_PER_NODE}/1024}") GiB"
else
    echo "Memory per node:                      not set"
fi
#########################################################


#### LAUNCHING CODE ######################################
# Install DAMPyF in this environment before submitting: python -m pip install -e "/path/to/DAMPyF"
# In run_simulation.py, set execution_mode="slurm" in DampfConfig.
# Submit from this example folder: sbatch submit_slurm.sh
python "${SLURM_SUBMIT_DIR}/run_simulation.py" > "${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
#########################################################
