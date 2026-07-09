# ADAPT-VQE Project Guide

This repository contains a small Variational Quantum Eigensolver (VQE)
workflow for molecular ground-state experiments. It supports fixed ansatz
baselines and an adaptive UCC-style ansatz that grows only when extra
excitations improve the energy enough to be worth keeping.

The main entry point is `vqe.py`.

## What This Project Does

The workflow is:

1. Choose a molecule: `H2`, `LiH`, or `NH3`.
2. Build the electronic-structure problem with PySCF through Qiskit Nature.
3. Optionally reduce the molecule to an active-space problem.
4. Map the fermionic Hamiltonian to qubits with Jordan-Wigner mapping.
5. Build either a fixed ansatz or an adaptive learned ansatz.
6. Optimize the circuit parameters with classical optimizers.
7. Report the final molecular energy, exact reference energy, error, circuit
   size, depth, and optimizer history.
8. Optionally run a noise study with ideal, noisy, and zero-noise extrapolated
   energies.

LiH and NH3 use active-space reduction by default because their full-space
problems are much slower. H2 runs without active-space reduction by default.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

If your system has both `python` and `python3`, prefer the virtual-environment
interpreter explicitly:

```bash
/Users/hebron/Documents/Adapt-VQE/.venv/bin/python vqe.py
```

## How To Run

Interactive run:

```bash
python3 vqe.py
```

The program first asks for the molecule, then asks whether to run basic or
advanced mode.

Run LiH with the adaptive workflow:

```bash
python3 vqe.py --molecule lih --mode advanced
```

Run LiH quickly with active-space reduction, tiny optimizer budget, and no
noise study:

```bash
python3 vqe.py --molecule lih --mode advanced --budget 10 --skip-noise
```

Run the fixed UCCSD baseline:

```bash
python3 vqe.py --molecule lih --mode basic --basic-ansatz uccsd
```

Run H2:

```bash
python3 vqe.py --molecule h2 --mode advanced --budget 50 --skip-noise
```

Force active-space reduction:

```bash
python3 vqe.py --molecule lih --mode advanced --active-space
```

## CLI Options

- `--molecule h2|lih|nh3`: molecule preset to run.
- `--mode basic|advanced`: choose fixed ansatz or adaptive ansatz workflow.
- `--basic-ansatz hea|uccsd|twolocal`: ansatz for basic mode.
- `--budget N`: optimizer budget per adaptive stage.
- `--seed N`: random seed for new adaptive ansatz parameters.
- `--plateau-threshold X`: energy-change threshold used by ansatz plateau detection.
- `--growth-threshold X`: minimum energy improvement required to keep a grown ansatz.
- `--active-space`: force active-space reduction.
- `--skip-noise`: skip ideal/noisy/ZNE noise comparison after optimization.
- `--verbose`: show optimizer logging.

## Files

### `vqe.py`

The orchestration layer and command-line entry point.

Important pieces:

- `prepare_problem(...)`: builds the molecular problem, applies active-space
  reduction when requested, maps the Hamiltonian to qubits, and computes the
  exact reference energy.
- `run_basic_vqe(...)`: runs a fixed ansatz from `ansatz.py` with COBYLA.
- `run_vqe(...)`: runs the adaptive learned-ansatz workflow.
- `AdaptiveVQECostFunction`: evaluates energy and tells the adaptive ansatz
  manager about every optimizer evaluation.
- `_format_report(...)`: prints the final report.

`vqe.py` also adds the Hamiltonian constant offsets back into reported
energies. This matters for active-space calculations: the quantum circuit
optimizes the active-space qubit Hamiltonian, but Qiskit Nature stores nuclear
repulsion and active-space/frozen-core offsets separately. The printed
`Final energy` and `Reference total energy` are molecular-scale energies with
those constants included.

### `hamiltonian.py`

Defines molecule presets and Hamiltonian helpers.

It contains embedded geometries for:

- `h2`
- `lih`
- `nh3`

It can build Qiskit Nature `PySCFDriver` objects for the main VQE workflow, and
also contains OpenFermion helper functions for building OpenFermion fermionic
and qubit Hamiltonians.

The main workflow in `vqe.py` uses the Qiskit Nature path:

```text
build_driver_for_molecule -> get_electronic_structure_problem -> mapper.map(...)
```

### `ansatz.py`

Builds fixed ansatz circuits for baseline runs.

Supported ansatz choices:

- `hea`: hardware-efficient ansatz using `EfficientSU2`.
- `uccsd`: chemistry-inspired UCCSD ansatz with Hartree-Fock initial state.
- `twolocal`: configurable `TwoLocal` ansatz.

The dispatcher is:

```python
get_ansatz("uccsd", problem=problem, mapper=mapper)
```

Basic mode in `vqe.py` uses this file.

### `learned_ansatz.py`

Owns the adaptive UCC ansatz logic.

It builds a singles-then-doubles excitation pool and starts with a smaller
circuit. When the current circuit plateaus, it tries adding more excitations.
If the new circuit improves the settled energy enough, the growth is kept. If
not, the manager rolls back to the smaller circuit and stops.

This is why output can show lines like:

```text
4 -> 5 params: ... [KEPT]
5 -> 6 params: ... [ROLLED BACK (stopped here)]
```

The rollback behavior is intentional: the goal is to find a good energy with a
smaller circuit than full UCCSD.

### `optimizer.py`

Contains optimizer wrappers and optimization reports.

Important classes:

- `OptimizationHistory`: stores all evaluated energies and parameters.
- `ConvergenceCriterion`: detects when best-so-far energy has stopped improving.
- `VQECostFunction`: estimator-backed energy function for fixed ansatz runs.
- `AdaptiveVQEOptimizer`: runs SPSA first, then switches to COBYLA when the
  convergence criterion triggers. It can fall back to L-BFGS-B if COBYLA fails.
- `SingleOptimizerRunner`: runs one optimizer from start to finish for baselines.

In reports, `iters=1` for SPSA means one reported SPSA optimizer iteration.
It may still involve many energy evaluations because SPSA performs calibration
and perturbation evaluations.

### `denoiser.py`

Runs the optional noise study.

It provides:

- ideal statevector evaluation,
- noisy Aer estimator evaluation,
- a depolarizing/readout noise model,
- global circuit folding,
- zero-noise extrapolation using linear, quadratic, or Richardson extrapolation.

Use `--skip-noise` when testing larger molecules or low-budget runs. The noise
study can be much slower than the optimizer, especially for larger circuits.

### `requirements.txt`

Lists Python dependencies:

- Qiskit
- Qiskit Nature
- Qiskit Algorithms
- Qiskit Aer
- PySCF
- OpenFermion
- OpenFermion-PySCF
- NumPy

## Basic vs Advanced Mode

Basic mode:

```text
fixed ansatz -> noisy estimator -> COBYLA -> report
```

Use this when comparing ansatz families such as HEA, UCCSD, and TwoLocal.

Advanced mode:

```text
adaptive ansatz -> SPSA -> COBYLA -> possible ansatz growth/rollback -> report
```

Use this when testing the learned/adaptive ansatz behavior.

## Active-Space Reduction And Energy Reporting

Active-space reduction shrinks the simulated qubit problem. For LiH, the active
space is much faster:

```text
active-space LiH: 3 spatial orbitals, 6 qubits
full-space LiH:   6 spatial orbitals, 12 qubits
```

The raw active-space qubit energy may look like `-1.x`, but that is not the
full molecular energy scale. `vqe.py` adds Qiskit Nature's stored Hamiltonian
constants back into the displayed final and reference energies, so active-space
LiH reports values near `-8`.

The growth-trial lines from `learned_ansatz.py` still show raw internal
active-space energies. Use the main `Final energy`, `Reference total energy`,
and `Absolute error` lines when judging the final result.

## Reading The Output

Example fields:

- `Molecule`: selected molecule.
- `Ansatz`: fixed ansatz or adaptive learned ansatz.
- `Optimizer`: optimizer workflow used.
- `Active space`: whether active-space reduction was used.
- `Qubits`: qubit count after mapping.
- `Final optimizer energy`: energy from the last optimizer stage.
- `Final energy`: energy of the final accepted circuit, with constants added.
- `Reference total energy`: exact diagonalization reference, with constants added.
- `Absolute error`: absolute difference between final and reference energy.
- `Final excitations`: number of active adaptive excitations kept.
- `Parameters`: number of circuit parameters.
- `Transpiled depth`: approximate circuit depth after transpilation.
- `Runtime`: wall-clock runtime.

If `Final optimizer energy` differs from `Final energy`, the last optimizer run
may have belonged to a trial ansatz that was rolled back. `Final energy` is the
accepted final answer.

## Practical Tips

Start with small runs:

```bash
python3 vqe.py --molecule lih --mode advanced --budget 10 --skip-noise
```

Increase budget once the setup works:

```bash
python3 vqe.py --molecule lih --mode advanced --budget 120 --skip-noise
```

Only enable the noise study after the optimizer behavior looks reasonable:

```bash
python3 vqe.py --molecule lih --mode advanced --budget 120
```

Full-space LiH and NH3 can be very slow. Active-space runs are the practical
default for quick experiments.

## Quick Checks

Check syntax:

```bash
python3 -m py_compile hamiltonian.py ansatz.py optimizer.py learned_ansatz.py denoiser.py vqe.py
```

Run a fast VQE test:

```bash
python3 vqe.py --molecule h2 --mode advanced --budget 10 --skip-noise
```

Run a fast LiH active-space test:

```bash
python3 vqe.py --molecule lih --mode advanced --budget 10 --skip-noise
```
