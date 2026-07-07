# ADAPT-VQE — Baseline VQE Modules

Built with `qiskit==2.5.0`, `qiskit-nature==0.8.0`, `qiskit-algorithms==0.4.0`,
`qiskit-aer==0.17.2`, `pyscf`.

```
pip install qiskit qiskit-nature qiskit-algorithms qiskit-aer pyscf
```

Every file below exposes small, independent functions — nothing is fused
together. `vqe.py` (not included here) is the only place that should import
across all of these and wire a full run together.

## geometry.txt
Plain-text benchmark geometries for **H2, LiH, H2O, NH3** (sto-3g, equilibrium
bond lengths). Parsed by `hamiltonian.parse_geometry_file()`.

## hamiltonian.py — molecule name → qubit Hamiltonian
- `parse_geometry_file(name)` — read one molecule block from `geometry.txt`
- `build_driver(name)` — build a `PySCFDriver`
- `get_electronic_structure_problem(driver)` — run PySCF, get the problem
- `get_second_q_hamiltonian(problem)` — fermionic operator
- `get_mapper(mapping)` / `map_to_qubit_hamiltonian(op, mapping)` — Jordan-Wigner or Bravyi-Kitaev
- `get_qubit_hamiltonian(name, mapping)` — the whole pipeline in one call, still just plumbing
- `get_reference_energy(problem)` — exact diagonalization (FCI) reference for validation
- `get_nuclear_repulsion_energy(problem)`

## active_space_reduction.py — shrink the problem
- `freeze_core(problem)` — `FreezeCoreTransformer`
- `reduce_active_space(problem, num_electrons, num_spatial_orbitals)` — explicit active space
- `auto_select_active_space(problem, n_occupied, n_virtual)` — HOMO/LUMO-window heuristic (useful for H2O/NH3)
- `report_problem_size(problem)` — orbitals/particles printout

## ansatz.py — Phase 2A (HEA / UCCSD / TwoLocal)
- `build_hartree_fock_state(problem, mapper)`
- `build_hea_ansatz(num_qubits, reps, entanglement)` — `EfficientSU2`-based hardware-efficient ansatz
- `build_uccsd_ansatz(problem, mapper, reps)` — chemistry-inspired UCCSD (+ HF reference state)
- `build_twolocal_ansatz(num_qubits, rotation_blocks, entanglement_blocks, reps)` — generic configurable `TwoLocal`
- `get_ansatz(name, ...)` — single dispatcher (`"hea" | "uccsd" | "twolocal"`)
- `ansatz_stats(ansatz)` — depth / parameter count / gate breakdown after transpiling to a fixed basis

## optimizer.py — Phase 2B (SPSA / COBYLA / SLSQP / Gradient Descent)
- `get_spsa`, `get_cobyla`, `get_slsqp`, `get_lbfgsb`, `get_gradient_descent`
- `get_optimizer(name, maxiter)` — dispatcher
- `make_cost_function(estimator, ansatz, hamiltonian, history)` — cost(params) → energy, logs every eval
- `run_optimization(optimizer, cost_fn, x0, history)` — runs `.minimize()`, returns final energy,
  iteration count, function-eval count, runtime, stability (std-dev of tail), full energy history

## denoiser.py — Phase 2D (Ideal / Noisy / Noisy+ZNE)
- `get_ideal_estimator()` / `run_ideal(...)` — noiseless `StatevectorEstimator`
- `build_noise_model(single_qubit_error, two_qubit_error, readout_error)` — depolarizing + readout noise
- `get_noisy_estimator(noise_model, shots)` / `run_noisy(...)` — Aer noisy estimator
- `fold_circuit(circuit, scale_factor)` — global unitary folding (U → U(U†U)ⁿ) to scale noise
- `zne_extrapolate(ansatz, hamiltonian, params, noise_model, scale_factors, extrapolator)` — linear /
  quadratic / Richardson extrapolation to the zero-noise limit
- `run_noise_comparison(...)` — one-call ideal vs. noisy vs. noisy+ZNE report with errors

## Not included here (by design)
- `learned_ansatz.py`, `vqe.py`, `denoiser.py`'s downstream usage in a full loop are left for you to
  assemble Phase 1/2 experiments on top of these building blocks.

## Quick sanity check
Each module has a `if __name__ == "__main__":` smoke test at the bottom, e.g.:
```
python hamiltonian.py     # prints reference energies for H2, LiH
python ansatz.py          # prints depth/param stats for hea/twolocal/uccsd on H2
python optimizer.py       # runs cobyla/slsqp/spsa on H2 UCCSD
python denoiser.py        # ideal vs noisy vs ZNE energy for H2
python active_space_reduction.py  # H2O full vs reduced active space
```