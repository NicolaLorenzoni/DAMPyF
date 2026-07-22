# DAMPyF

DAMPyF is a Python implementation of the dissipation-assisted matrix product
factorisation (DAMPF) method. It uses pseudomodes and matrix-product states to
simulate the non-perturbative dynamics of multisite open quantum systems coupled
to structured bosonic environments.

DAMPyF provides two main workflows:

- `energy_transfer` propagates an initial system density matrix and returns the
  reduced system dynamics.
- `linear_spectra` propagates site-resolved optical coherences for the subsequent
  calculation of absorption and circular-dichroism spectra.

## Documentation and citation

The complete description of the method, installation instructions, input and
output formats, convergence parameters, and examples is provided in the
[DAMPyF documentation publication](PUBLICATION_LINK_TO_BE_ADDED).

Once available, this publication should be cited in work that uses DAMPyF.

## Requirements

DAMPyF requires Python 3, NumPy, SciPy, h5py, psutil, and Ray. The plotting tools
additionally require Matplotlib and a LaTeX installation.

## Getting started

Two ready-to-use dimer examples are included:

- `Examples/Dimer_energy_transfer/`
- `Examples/Dimer_linear_spectra/`

To run an example locally:

1. Open the corresponding example directory.
2. Edit `configure_simulation.py` and the files in `input_data/` as needed.
3. Set `execution_mode = "local"` and run:

   ```bash
   python run_dampf.py
   ```

For execution on a Slurm-managed system, configure the requested resources in
`slurm_code_launcher.sh`, set `execution_mode = "slurm"`, and run:

```bash
sbatch slurm_code_launcher.sh
```

Simulation results are written to `output_data/`. Post-processing scripts are
provided in `analysis_tools/`, while utilities for generating input files and
estimating numerical requirements are collected in `helper_tools/`.

All Hamiltonian parameters, pseudomode parameters, thermal energies, and times
must be supplied in a mutually consistent unit convention in which energy times
time is dimensionless (`hbar = 1`).
