"""
vqe.py
------
Orchestration layer. This is the ONLY file that combines
hamiltonian.py + active_space_reduction.py + ansatz.py + optimizer.py +
denoiser.py into an actual VQE run. Every other module stays standalone.

Provides:
    Phase 1 (baseline):
        run_baseline_vqe(...)          -> single VQE run + validation vs. reference

    Phase 2A (ansatz comparison):
        compare_ansatze(...)

    Phase 2B (optimizer comparison):
        compare_optimizers(...)

    Phase 2C (initialization comparison):
        get_initial_point(...)
        compare_initializations(...)

    Phase 2D (noise study):
        run_noise_study(...)
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np
from qiskit.primitives import StatevectorEstimator
from qiskit_nature.second_q.algorithms.initial_points import HFInitialPoint, MP2InitialPoint

from hamiltonian import get_qubit_hamiltonian, get_reference_energy
from active_space_reduction import auto_select_active_space, report_problem_size
from ansatz import get_ansatz, ansatz_stats
from optimizer import get_optimizer, make_cost_function, run_optimization
from denoiser import build_noise_model, run_noise_comparison


# --------------------------------------------------------------------------- #
# Shared setup helper: molecule name -> (qubit_op, problem, mapper, nuc_rep)
# --------------------------------------------------------------------------- #
def prepare_problem(
    molecule_name: str,
    mapping: str = "jordan_wigner",
    use_active_space: bool = False,
    n_occupied: int = 2,
    n_virtual: int = 2,
):
    """Build the qubit Hamiltonian for a molecule, optionally reducing the
    active space first (recommended for H2O / NH3)."""
    from hamiltonian import build_driver, get_electronic_structure_problem, get_mapper

    driver = build_driver(molecule_name)
    problem = get_electronic_structure_problem(driver)

    if use_active_space:
        report_problem_size(problem, f"{molecule_name} full space")
        problem = auto_select_active_space(problem, n_occupied=n_occupied, n_virtual=n_virtual)
        report_problem_size(problem, f"{molecule_name} active space")

    mapper = get_mapper(mapping)
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())
    return qubit_op, problem, mapper


# --------------------------------------------------------------------------- #
# Phase 2C: initial point strategies
# --------------------------------------------------------------------------- #
def get_initial_point(
    strategy: str,
    ansatz,
    problem=None,
    seed: int = 42,
    scale: float = 0.1,
) -> np.ndarray:
    """
    strategy : "random" | "hf" | "mp2"

    "hf"   -> all-zero amplitudes (Hartree-Fock reference, only valid/meaningful
              for UCC-type ansatze where params=0 means "just HF").
    "mp2"  -> MP2-informed amplitudes seeded into the UCC ansatz
              (falls back to zeros if the ansatz/problem doesn't support it,
              e.g. HEA / TwoLocal have no chemical meaning for MP2 amplitudes).
    "random" -> small random perturbation around zero.
    """
    strategy = strategy.strip().lower()
    n = ansatz.num_parameters

    if strategy == "random":
        rng = np.random.default_rng(seed)
        return rng.uniform(-scale, scale, n)

    if strategy == "hf":
        try:
            hf_ip = HFInitialPoint()
            hf_ip.ansatz = ansatz
            hf_ip.problem = problem
            return hf_ip.to_numpy_array()
        except Exception:
            return np.zeros(n)

    if strategy == "mp2":
        try:
            mp2_ip = MP2InitialPoint()
            mp2_ip.ansatz = ansatz
            mp2_ip.problem = problem
            return mp2_ip.to_numpy_array()
        except Exception:
            # Ansatz has no UCC excitation structure (e.g. HEA/TwoLocal) ->
            # MP2 amplitudes don't map onto it; fall back to zeros.
            return np.zeros(n)

    raise ValueError(f"Unknown initialization strategy '{strategy}'. Use random/hf/mp2.")


# --------------------------------------------------------------------------- #
# Phase 1: baseline VQE
# --------------------------------------------------------------------------- #
def run_baseline_vqe(
    molecule_name: str,
    ansatz_name: str = "uccsd",
    optimizer_name: str = "cobyla",
    mapping: str = "jordan_wigner",
    init_strategy: str = "hf",
    maxiter: int = 200,
    reps: int = 1,
    use_active_space: bool = False,
    n_occupied: int = 2,
    n_virtual: int = 2,
) -> dict:
    """
    Full Phase 1 baseline run:
        build Hamiltonian -> build ansatz -> pick initial point ->
        optimize -> compare to classical reference energy.
    """
    qubit_op, problem, mapper = prepare_problem(
        molecule_name, mapping, use_active_space, n_occupied, n_virtual
    )
    nuclear_repulsion = problem.nuclear_repulsion_energy

    ansatz, _hf_state = get_ansatz(ansatz_name, problem=problem, mapper=mapper, reps=reps)
    x0 = get_initial_point(init_strategy, ansatz, problem=problem)

    estimator = StatevectorEstimator()
    history = {"energies": [], "params": [], "n_evals": 0}
    cost_fn = make_cost_function(estimator, ansatz, qubit_op, history)

    optimizer = get_optimizer(optimizer_name, maxiter=maxiter)
    report = run_optimization(optimizer, cost_fn, x0, history)

    computed_total_energy = report["final_energy"] + nuclear_repulsion
    reference_total_energy = get_reference_energy(problem)

    return {
        "molecule": molecule_name,
        "ansatz": ansatz_name,
        "optimizer": optimizer_name,
        "mapping": mapping,
        "init_strategy": init_strategy,
        "computed_energy": computed_total_energy,
        "reference_energy": reference_total_energy,
        "absolute_error": abs(computed_total_energy - reference_total_energy),
        "convergence_curve": [e + nuclear_repulsion for e in report["energy_history"]],
        "num_iterations": report["num_iterations"],
        "num_function_evals": report["num_function_evals"],
        "runtime_sec": report["runtime_sec"],
        "optimal_params": report["optimal_params"],
        "ansatz_stats": ansatz_stats(ansatz),
    }


# --------------------------------------------------------------------------- #
# Phase 2A: ansatz comparison
# --------------------------------------------------------------------------- #
def compare_ansatze(
    molecule_name: str,
    ansatz_names: Sequence[str] = ("hea", "twolocal", "uccsd"),
    optimizer_name: str = "cobyla",
    mapping: str = "jordan_wigner",
    maxiter: int = 200,
    reps: int = 2,
) -> dict:
    """Run the same molecule/optimizer with each ansatz type and collect
    accuracy / circuit depth / parameter count / runtime / convergence speed."""
    results = {}
    for name in ansatz_names:
        results[name] = run_baseline_vqe(
            molecule_name,
            ansatz_name=name,
            optimizer_name=optimizer_name,
            mapping=mapping,
            init_strategy="hf" if name == "uccsd" else "random",
            maxiter=maxiter,
            reps=reps,
        )
    return results


# --------------------------------------------------------------------------- #
# Phase 2B: optimizer comparison
# --------------------------------------------------------------------------- #
def compare_optimizers(
    molecule_name: str,
    optimizer_names: Sequence[str] = ("spsa", "cobyla", "slsqp", "gradient_descent"),
    ansatz_name: str = "uccsd",
    mapping: str = "jordan_wigner",
    maxiter: int = 200,
    reps: int = 1,
) -> dict:
    """Run the same molecule/ansatz with each optimizer and collect
    iterations / function evals / runtime / stability / final energy."""
    results = {}
    for name in optimizer_names:
        results[name] = run_baseline_vqe(
            molecule_name,
            ansatz_name=ansatz_name,
            optimizer_name=name,
            mapping=mapping,
            init_strategy="hf",
            maxiter=maxiter,
            reps=reps,
        )
    return results


# --------------------------------------------------------------------------- #
# Phase 2C: initialization comparison
# --------------------------------------------------------------------------- #
def compare_initializations(
    molecule_name: str,
    init_strategies: Sequence[str] = ("random", "hf", "mp2"),
    ansatz_name: str = "uccsd",
    optimizer_name: str = "cobyla",
    mapping: str = "jordan_wigner",
    maxiter: int = 200,
    reps: int = 1,
) -> dict:
    """
    Run the same molecule/ansatz/optimizer with each initialization strategy
    and collect initial energy / convergence speed / final accuracy.
    """
    qubit_op, problem, mapper = prepare_problem(molecule_name, mapping)
    nuclear_repulsion = problem.nuclear_repulsion_energy
    results = {}

    for strategy in init_strategies:
        ansatz, _ = get_ansatz(ansatz_name, problem=problem, mapper=mapper, reps=reps)
        x0 = get_initial_point(strategy, ansatz, problem=problem)

        estimator = StatevectorEstimator()
        initial_energy = float(
            estimator.run([(ansatz, qubit_op, x0)]).result()[0].data.evs
        ) + nuclear_repulsion

        history = {"energies": [], "params": [], "n_evals": 0}
        cost_fn = make_cost_function(estimator, ansatz, qubit_op, history)
        optimizer = get_optimizer(optimizer_name, maxiter=maxiter)
        report = run_optimization(optimizer, cost_fn, x0, history)

        final_energy = report["final_energy"] + nuclear_repulsion
        reference_energy = get_reference_energy(problem)

        results[strategy] = {
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "reference_energy": reference_energy,
            "absolute_error": abs(final_energy - reference_energy),
            "num_iterations": report["num_iterations"],
            "num_function_evals": report["num_function_evals"],
            "convergence_curve": [e + nuclear_repulsion for e in report["energy_history"]],
        }

    return results


# --------------------------------------------------------------------------- #
# Phase 2D: noise study
# --------------------------------------------------------------------------- #
def run_noise_study(
    molecule_name: str,
    ansatz_name: str = "twolocal",
    optimizer_name: str = "cobyla",
    mapping: str = "jordan_wigner",
    maxiter: int = 100,
    reps: int = 1,
    single_qubit_error: float = 0.001,
    two_qubit_error: float = 0.01,
    readout_error: float = 0.02,
    zne_scale_factors: Sequence[int] = (1, 3, 5),
) -> dict:
    """
    Optimize on the ideal simulator first (cheap), then evaluate the
    converged parameters under: ideal / noisy / noisy+ZNE. This isolates
    the *evaluation*-time noise degradation from optimizer noise-robustness
    (which is instead covered in compare_optimizers with SPSA vs gradient-based).
    """
    baseline = run_baseline_vqe(
        molecule_name,
        ansatz_name=ansatz_name,
        optimizer_name=optimizer_name,
        mapping=mapping,
        init_strategy="hf" if ansatz_name == "uccsd" else "random",
        maxiter=maxiter,
        reps=reps,
    )

    qubit_op, problem, mapper = prepare_problem(molecule_name, mapping)
    ansatz, _ = get_ansatz(ansatz_name, problem=problem, mapper=mapper, reps=reps)

    noise_model = build_noise_model(single_qubit_error, two_qubit_error, readout_error)
    electronic_reference = baseline["reference_energy"] - problem.nuclear_repulsion_energy

    noise_report = run_noise_comparison(
        ansatz,
        qubit_op,
        baseline["optimal_params"],
        reference_energy=electronic_reference,
        noise_model=noise_model,
        scale_factors=zne_scale_factors,
    )

    nuc = problem.nuclear_repulsion_energy
    return {
        "molecule": molecule_name,
        "ansatz": ansatz_name,
        "ideal_energy": noise_report["ideal_energy"] + nuc,
        "noisy_energy": noise_report["noisy_energy"] + nuc,
        "zne_energy": noise_report["zne_energy"] + nuc,
        "ideal_error": noise_report["ideal_error"],
        "noisy_error": noise_report["noisy_error"],
        "zne_error": noise_report["zne_error"],
        "zne_details": noise_report["zne_details"],
        "reference_energy": baseline["reference_energy"],
    }


# --------------------------------------------------------------------------- #
# Demo / smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=== Phase 1: baseline VQE (H2, UCCSD, COBYLA, HF init) ===")
    result = run_baseline_vqe("H2", ansatz_name="uccsd", optimizer_name="cobyla")
    print(f"computed={result['computed_energy']:.6f}  reference={result['reference_energy']:.6f}  "
          f"error={result['absolute_error']:.2e}  nfev={result['num_function_evals']}")

    print("\n=== Phase 2A: ansatz comparison (H2) ===")
    ansatz_results = compare_ansatze("H2", maxiter=150)
    for name, r in ansatz_results.items():
        s = r["ansatz_stats"]
        print(f"{name:10s} energy={r['computed_energy']:.6f}  error={r['absolute_error']:.2e}  "
              f"depth={s['transpiled_depth']:4d}  params={s['num_parameters']:3d}  "
              f"runtime={r['runtime_sec']:.2f}s")

    print("\n=== Phase 2B: optimizer comparison (H2, UCCSD) ===")
    opt_results = compare_optimizers("H2", maxiter=150)
    for name, r in opt_results.items():
        print(f"{name:18s} energy={r['computed_energy']:.6f}  error={r['absolute_error']:.2e}  "
              f"nfev={r['num_function_evals']:4d}  runtime={r['runtime_sec']:.2f}s")

    print("\n=== Phase 2C: initialization comparison (H2, UCCSD) ===")
    init_results = compare_initializations("H2", maxiter=150)
    for name, r in init_results.items():
        print(f"{name:8s} initial={r['initial_energy']:.6f}  final={r['final_energy']:.6f}  "
              f"error={r['absolute_error']:.2e}  nfev={r['num_function_evals']}")

    print("\n=== Phase 2D: noise study (H2, TwoLocal) ===")
    noise_result = run_noise_study("H2", ansatz_name="twolocal", maxiter=100)
    print(f"ideal={noise_result['ideal_energy']:.6f}  noisy={noise_result['noisy_energy']:.6f}  "
          f"zne={noise_result['zne_energy']:.6f}")
    print(f"ideal_err={noise_result['ideal_error']:.4f}  noisy_err={noise_result['noisy_error']:.4f}  "
          f"zne_err={noise_result['zne_error']:.4f}")