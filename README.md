# DAMPyF

DAMPyF implements dissipation-assisted matrix product factorisation (DAMPF) for open quantum systems
coupled to bosonic pseudomodes. It uses matrix product states and Ray parallelization to simulate
energy-transfer dynamics and linear optical spectra through an importable Python API.

## Installation

Use Python 3.10 or newer. Activate your Python or Conda environment, then run the following command
from the project root, which contains `pyproject.toml`:

```bash
python -m pip install -e .
```

This installs DAMPyF and its dependencies: NumPy, SciPy, Ray, h5py, psutil, and Matplotlib.
The `-e` option links the installation to this folder, so edits to the Python source take effect
without reinstalling. Omit `-e` for a regular installation.

## Run the test examples

From the project root, run either example:

```bash
python test_examples/Dimer_energy_transfer/run_simulation.py
python test_examples/Dimer_linear_spectra/run_simulation.py
```

Each script runs a simulation and plots a comparison with the data in its `reference_data` folder.
Results and comparison figures are saved in the example's `output_data` folder.
Model arrays and numerical settings are defined at the beginning of each `run_simulation.py`.

## Use DAMPyF in your own scripts

```python
from dampyf import DampfConfig, LocalPseudomodes, run_dampf, units
```

Define the Hamiltonian and pseudomode parameters as NumPy arrays, construct `LocalPseudomodes` and
`DampfConfig` objects, and call `run_dampf` inside an `if __name__ == "__main__":` block.
Use the test examples as complete usage references.

The returned object provides `result.times` and either `result.rho_system` for energy transfer or
`result.optical_coherence` for linear spectra. Supply `output_directory` to also save results to disk.

For energies in cm⁻¹, convert times from femtoseconds by multiplying by
`units.FEMTOSECOND_TO_SPECTROSCOPIC_TIME`. Supply thermal energies as `k_B T` in the same energy units.
All arguments and defaults are documented in [run_dampf](dampyf/api.py).

## Estimate resources

Edit the arrays and settings in the standalone helper scripts, then run:

```bash
python test_examples/helpers/estimate_memory.py
python test_examples/helpers/estimate_fock_dimensions.py
```

These estimate memory requirements and suitable local Fock dimensions, respectively.

## Run with Slurm

Set `execution_mode="slurm"` in the example's `DampfConfig` and edit `submit_slurm.sh` to select your
environment and requested resources. Install DAMPyF in that environment, then submit from the
chosen example folder:

```bash
sbatch submit_slurm.sh
```
