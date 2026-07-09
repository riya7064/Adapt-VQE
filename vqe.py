"""
vqe.py
------
Orchestration layer that wires the full Adapt-VQE pipeline:

    hamiltonian.py     ->  molecule + qubit Hamiltonian
    learned_ansatz.py  ->  adaptive UCC ansatz (grow / rollback)
    optimizer.py       ->  adaptive SPSA -> COBYLA optimization
    denoiser.py        ->  ideal / noisy / ZNE evaluation

Run
---
    python vqe.py
    python vqe.py --molecule h2 --budget 100
    python vqe.py --molecule lih --skip-noise
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from qiskit import transpile
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_nature.second_q.mappers import JordanWignerMapper, QubitMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer

from denoiser import build_noise_model, run_noise_comparison
from hamiltonian import (
    build_driver_for_molecule,
    get_electronic_structure_problem,
    get_molecule_settings,
)
from learned_ansatz import AdaptiveAnsatzManager
from optimizer import (
    AdaptiveVQEOptimizer,
    ConvergenceCriterion,
    OptimizationHistory,
    OptimizationReport,
    OptimizerInterrupted,
    VQECostFunction,
)

logger = logging.getLogger(__name__)

ACTIVE_SPACE_BY_MOLECULE: Dict[str, Dict[str, Any]] = {
    "lih": {
        "num_electrons": 2,
        "num_spatial_orbitals": 3,
        "active_orbitals": [1, 2, 3],
    },
    "nh3": {
        "num_electrons": 4,
        "num_spatial_orbitals": 3,
        "active_orbitals": None,
    },
}


class AnsatzChangedError(OptimizerInterrupted):
    """Raised when the adaptive ansatz grows or rolls back mid-optimization."""


@dataclass
class VQEProblem:
    molecule: str
    problem: ElectronicStructureProblem
    qubit_hamiltonian: SparsePauliOp
    mapper: QubitMapper
    reference_total_energy: float
    used_active_space: bool


@dataclass
class VQEResult:
    molecule: str
    optimizer: str
    computed_total_energy: float
    reference_total_energy: float
    absolute_error: float
    optimal_params: np.ndarray
    final_num_excitations: int
    total_pool_excitations: int
    ansatz_summary: str
    optimization_report: Optional[OptimizationReport]
    ansatz_stats: Dict[str, Any]
    noise_report: Dict[str, Any]
    runtime_sec: float
    growth_log: Sequence[Any]


def _exact_ground_state_energy(qubit_hamiltonian: SparsePauliOp) -> float:
    return float(np.min(np.linalg.eigvalsh(qubit_hamiltonian.to_matrix(sparse=False))))


def prepare_problem(
    molecule_name: str = "h2",
    qubit_mapper: Optional[QubitMapper] = None,
    use_active_space: Optional[bool] = None,
) -> VQEProblem:
    """Build the electronic-structure problem and qubit Hamiltonian."""
    key = molecule_name.strip().lower()
    get_molecule_settings(key)

    driver = build_driver_for_molecule(key)
    problem = get_electronic_structure_problem(driver)

    if use_active_space is None:
        use_active_space = key in ACTIVE_SPACE_BY_MOLECULE

    if use_active_space:
        if key not in ACTIVE_SPACE_BY_MOLECULE:
            raise ValueError(
                f"No active-space preset for '{key}'. "
                f"Supported: {', '.join(sorted(ACTIVE_SPACE_BY_MOLECULE))}"
            )
        active = ACTIVE_SPACE_BY_MOLECULE[key]
        problem = ActiveSpaceTransformer(
            num_electrons=active["num_electrons"],
            num_spatial_orbitals=active["num_spatial_orbitals"],
            active_orbitals=active["active_orbitals"],
        ).transform(problem)

    mapper = qubit_mapper or JordanWignerMapper()
    qubit_hamiltonian = mapper.map(problem.hamiltonian.second_q_op())

    return VQEProblem(
        molecule=key,
        problem=problem,
        qubit_hamiltonian=qubit_hamiltonian,
        mapper=mapper,
        reference_total_energy=_exact_ground_state_energy(qubit_hamiltonian),
        used_active_space=use_active_space,
    )


class AdaptiveVQECostFunction(VQECostFunction):
    """Cost function that feeds every evaluation to AdaptiveAnsatzManager."""

    def __init__(
        self,
        estimator,
        ansatz_manager: AdaptiveAnsatzManager,
        hamiltonian: SparsePauliOp,
        history: Optional[OptimizationHistory] = None,
    ) -> None:
        self.estimator = estimator
        self.ansatz_manager = ansatz_manager
        self.hamiltonian = hamiltonian
        self.history = history if history is not None else OptimizationHistory()
        self._phase = "unspecified"

    @property
    def ansatz(self):
        return self.ansatz_manager.circuit

    def __call__(self, params: np.ndarray) -> float:
        job = self.estimator.run([(self.ansatz, self.hamiltonian, params)])
        energy = float(job.result()[0].data.evs)
        self.history.record(energy, params, phase=self._phase)
        allow_growth = self._phase != "cobyla"
        if self.ansatz_manager.observe(energy, params=params, allow_growth=allow_growth):
            raise AnsatzChangedError()
        return energy


def _evaluate_energy(
    estimator,
    ansatz_manager: AdaptiveAnsatzManager,
    hamiltonian: SparsePauliOp,
    params: np.ndarray,
) -> float:
    job = estimator.run([(ansatz_manager.circuit, hamiltonian, params)])
    return float(job.result()[0].data.evs)


def _circuit_stats(circuit) -> Dict[str, Any]:
    transpiled = transpile(circuit, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
    return {
        "num_parameters": circuit.num_parameters,
        "num_qubits": circuit.num_qubits,
        "raw_depth": circuit.depth(),
        "transpiled_depth": transpiled.depth(),
        "transpiled_gate_count": sum(transpiled.count_ops().values()),
    }


def run_vqe(
    molecule_name: str = "h2",
    *,
    optimizer_name: str = "adaptive",
    optimizer_budget: int = 120,
    use_active_space: Optional[bool] = None,
    plateau_threshold: float = 1e-4,
    plateau_patience: int = 5,
    growth_benefit_threshold: float = 1e-3,
    growth_batch_size: int = 1,
    seed: int = 42,
    run_noise_study: bool = True,
    single_qubit_error: float = 0.001,
    two_qubit_error: float = 0.01,
    readout_error: float = 0.02,
    zne_scale_factors: Sequence[int] = (1, 3, 5),
) -> VQEResult:
    """Run the full Adapt-VQE pipeline with a learned (adaptive) ansatz."""
    t0 = time.time()
    vqe_problem = prepare_problem(molecule_name, use_active_space=use_active_space)

    ansatz_manager = AdaptiveAnsatzManager(
        num_spatial_orbitals=vqe_problem.problem.num_spatial_orbitals,
        num_particles=tuple(vqe_problem.problem.num_particles),
        qubit_mapper=vqe_problem.mapper,
        plateau_threshold=plateau_threshold,
        plateau_patience=plateau_patience,
        growth_benefit_threshold=growth_benefit_threshold,
        growth_batch_size=growth_batch_size,
        new_param_init="small_random",
        seed=seed,
    )

    if optimizer_name.strip().lower() != "adaptive":
        raise ValueError(
            "The learned ansatz requires the adaptive optimizer loop. "
            "Use --optimizer adaptive (default)."
        )

    adaptive_optimizer = AdaptiveVQEOptimizer(
        criterion=ConvergenceCriterion(patience=5, min_delta=1e-4, min_evals=10),
        fallback_name="lbfgsb",
    )

    estimator = StatevectorEstimator()
    last_report: Optional[OptimizationReport] = None
    max_stages = len(ansatz_manager.pool) + 2

    for _stage in range(1, max_stages + 1):
        if ansatz_manager.is_done:
            break

        cost_fn = AdaptiveVQECostFunction(
            estimator,
            ansatz_manager,
            vqe_problem.qubit_hamiltonian,
        )

        try:
            last_report = adaptive_optimizer.optimize(
                cost_fn,
                ansatz_manager.initial_point,
                total_budget=optimizer_budget,
            )
        except AnsatzChangedError:
            if ansatz_manager.is_done:
                break
            continue

        changed = ansatz_manager.finalize_stage(
            last_report.final_energy,
            last_report.optimal_params,
        )
        if changed:
            if ansatz_manager.is_done:
                break
            continue
        break

    final_params = ansatz_manager.parameters
    computed_total_energy = _evaluate_energy(
        estimator,
        ansatz_manager,
        vqe_problem.qubit_hamiltonian,
        final_params,
    )

    if not run_noise_study:
        logger.warning("run_noise_study=False requested, but denoiser is required and will run.")

    noise_model = build_noise_model(
        single_qubit_error=single_qubit_error,
        two_qubit_error=two_qubit_error,
        readout_error=readout_error,
    )
    noise_report: Dict[str, Any] = run_noise_comparison(
        ansatz_manager.circuit,
        vqe_problem.qubit_hamiltonian,
        final_params,
        reference_energy=vqe_problem.reference_total_energy,
        noise_model=noise_model,
        scale_factors=zne_scale_factors,
    )

    return VQEResult(
        molecule=vqe_problem.molecule,
        optimizer="adaptive",
        computed_total_energy=computed_total_energy,
        reference_total_energy=vqe_problem.reference_total_energy,
        absolute_error=abs(computed_total_energy - vqe_problem.reference_total_energy),
        optimal_params=final_params,
        final_num_excitations=ansatz_manager.num_active,
        total_pool_excitations=len(ansatz_manager.pool),
        ansatz_summary=ansatz_manager.summary(),
        optimization_report=last_report,
        ansatz_stats=_circuit_stats(ansatz_manager.circuit),
        noise_report=noise_report,
        runtime_sec=time.time() - t0,
        growth_log=list(ansatz_manager.growth_log),
    )


def _format_report(result: VQEResult, problem: VQEProblem) -> str:
    stats = result.ansatz_stats
    lines = [
        f"Molecule              : {result.molecule.upper()}",
        f"Ansatz                : adaptive UCC (learned_ansatz.py)",
        f"Optimizer             : {result.optimizer}",
        f"Active space          : {'yes' if problem.used_active_space else 'no'}",
        f"Qubits                : {problem.qubit_hamiltonian.num_qubits}",
        f"Computed total energy : {result.computed_total_energy:.8f}",
        f"Reference total energy: {result.reference_total_energy:.8f}",
        f"Absolute error        : {result.absolute_error:.2e}",
        (
            f"Final excitations     : {result.final_num_excitations}/"
            f"{result.total_pool_excitations}"
        ),
        f"Parameters            : {stats['num_parameters']}",
        f"Transpiled depth      : {stats['transpiled_depth']}",
        f"Runtime               : {result.runtime_sec:.2f}s",
        "",
        result.ansatz_summary,
    ]

    if result.optimization_report is not None:
        lines.extend(["", "Last optimizer run:"])
        for phase in result.optimization_report.phases:
            lines.append(
                f"  {phase.optimizer:12s}  iters={phase.iterations:4d}  {phase.reason}"
            )
        lines.append(
            f"  function evals={result.optimization_report.num_function_evals}  "
            f"switched_at_eval={result.optimization_report.switched_at_eval}"
        )

    if result.noise_report:
        lines.extend(
            [
                "",
                "Noise study (converged parameters):",
                f"  ideal : {result.noise_report['ideal_energy']:.8f}  "
                f"(err {result.noise_report['ideal_error']:.2e})",
                f"  noisy : {result.noise_report['noisy_energy']:.8f}  "
                f"(err {result.noise_report['noisy_error']:.2e})",
                f"  zne   : {result.noise_report['zne_energy']:.8f}  "
                f"(err {result.noise_report['zne_error']:.2e})",
            ]
        )

    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the full Adapt-VQE pipeline.")
    parser.add_argument("--molecule", default="h2", help="Molecule preset: h2, lih, nh3.")
    parser.add_argument("--budget", type=int, default=120, help="Optimizer budget per ansatz stage.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for new ansatz parameters.")
    parser.add_argument(
        "--plateau-threshold",
        type=float,
        default=1e-4,
        help="Ansatz plateau sensitivity.",
    )
    parser.add_argument(
        "--growth-threshold",
        type=float,
        default=1e-3,
        help="Minimum energy drop to keep a growth trial.",
    )
    parser.add_argument(
        "--active-space",
        action="store_true",
        help="Force active-space reduction (auto-enabled for lih/nh3).",
    )
    parser.add_argument(
        "--skip-noise",
        action="store_true",
        help="Skip the denoiser ideal/noisy/ZNE comparison.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Show optimizer logs.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    if args.skip_noise:
        logger.warning("--skip-noise is ignored: denoiser execution is required for this workflow.")

    use_active_space = True if args.active_space else None

    print(f"=== Adapt-VQE: {args.molecule.upper()} ===")
    problem = prepare_problem(args.molecule, use_active_space=use_active_space)
    space_label = "active space" if problem.used_active_space else "full space"
    print(
        f"Problem: {problem.problem.num_spatial_orbitals} spatial orbitals, "
        f"particles={problem.problem.num_particles}, "
        f"qubits={problem.qubit_hamiltonian.num_qubits} ({space_label})"
    )
    print(f"Reference total energy: {problem.reference_total_energy:.8f}\n")

    result = run_vqe(
        args.molecule,
        optimizer_budget=args.budget,
        use_active_space=use_active_space,
        plateau_threshold=args.plateau_threshold,
        growth_benefit_threshold=args.growth_threshold,
        seed=args.seed,
        run_noise_study=True,
    )
    print(_format_report(result, problem))


if __name__ == "__main__":
    main()
