"""
optimizer.py
------------
Everything needed for the Phase 2B optimizer comparison:
    - SPSA
    - COBYLA
    - SLSQP (falls back to L-BFGS-B if requested)
    - Gradient Descent

Each optimizer has its own builder function. A single generic
`run_optimization()` function drives any of them against a cost function
and records the metrics asked for in the spec: iterations, function
evaluations, runtime, stability (energy std-dev near the end), final energy.
"""

from __future__ import annotations

import time
from typing import Callable, List, Optional

import numpy as np
from qiskit_algorithms.optimizers import SPSA, COBYLA, SLSQP, L_BFGS_B, GradientDescent


# --------------------------------------------------------------------------- #
# 1. Individual optimizer builders
# --------------------------------------------------------------------------- #
def get_spsa(maxiter: int = 100, **kwargs) -> SPSA:
    """SPSA: good for noisy/hardware cost functions, gradient-free, 2 evals/iter."""
    return SPSA(maxiter=maxiter, **kwargs)


def get_cobyla(maxiter: int = 100, **kwargs) -> COBYLA:
    """COBYLA: gradient-free, constrained/unconstrained local optimizer."""
    return COBYLA(maxiter=maxiter, **kwargs)


def get_slsqp(maxiter: int = 100, **kwargs) -> SLSQP:
    """SLSQP: gradient-based (uses finite-difference gradients if none given)."""
    return SLSQP(maxiter=maxiter, **kwargs)


def get_lbfgsb(maxiter: int = 100, **kwargs) -> L_BFGS_B:
    """L-BFGS-B: quasi-Newton gradient-based optimizer, used if SLSQP unavailable."""
    return L_BFGS_B(maxiter=maxiter, **kwargs)


def get_gradient_descent(maxiter: int = 100, learning_rate: float = 0.01, **kwargs) -> GradientDescent:
    """Plain gradient descent (finite-difference gradient by default)."""
    return GradientDescent(maxiter=maxiter, learning_rate=learning_rate, **kwargs)


# --------------------------------------------------------------------------- #
# 2. Dispatcher
# --------------------------------------------------------------------------- #
def get_optimizer(name: str, maxiter: int = 100, **kwargs):
    """
    name : "spsa" | "cobyla" | "slsqp" | "lbfgsb" | "gradient_descent"
    """
    name = name.strip().lower().replace("-", "_").replace(" ", "_")
    builders = {
        "spsa": get_spsa,
        "cobyla": get_cobyla,
        "slsqp": get_slsqp,
        "l_bfgs_b": get_lbfgsb,
        "lbfgsb": get_lbfgsb,
        "gradient_descent": get_gradient_descent,
        "gd": get_gradient_descent,
    }
    if name not in builders:
        raise ValueError(f"Unknown optimizer '{name}'. Options: {list(builders.keys())}")
    return builders[name](maxiter=maxiter, **kwargs)


# --------------------------------------------------------------------------- #
# 3. Cost function factory (wraps an Estimator + ansatz + Hamiltonian)
# --------------------------------------------------------------------------- #
def make_cost_function(estimator, ansatz, hamiltonian, history: Optional[dict] = None) -> Callable:
    """
    Build a scalar cost function cost(params) -> energy, using the given
    Estimator primitive. Every call is logged into `history` (if provided)
    so we can plot convergence curves and report function-eval counts.

    history keys populated: 'energies' (list), 'params' (list), 'n_evals' (int)
    """
    if history is None:
        history = {"energies": [], "params": [], "n_evals": 0}
    history.setdefault("energies", [])
    history.setdefault("params", [])
    history.setdefault("n_evals", 0)

    def cost(params: np.ndarray) -> float:
        job = estimator.run([(ansatz, hamiltonian, params)])
        energy = float(job.result()[0].data.evs)
        history["energies"].append(energy)
        history["params"].append(np.array(params, copy=True))
        history["n_evals"] += 1
        return energy

    return cost


# --------------------------------------------------------------------------- #
# 4. Generic optimization runner (records everything Phase 2B needs)
# --------------------------------------------------------------------------- #
def run_optimization(
    optimizer,
    cost_fn: Callable,
    initial_point: np.ndarray,
    history: Optional[dict] = None,
) -> dict:
    """
    Run `optimizer.minimize(cost_fn, initial_point)` and return a report dict:
        final_energy, num_iterations, num_function_evals, runtime_sec,
        stability (std-dev of energy over the last 10% of evaluations),
        energy_history (for the convergence curve), optimal_params
    """
    t0 = time.time()
    result = optimizer.minimize(fun=cost_fn, x0=initial_point)
    runtime = time.time() - t0

    energies = history["energies"] if history is not None else []
    tail = max(1, len(energies) // 10)
    stability = float(np.std(energies[-tail:])) if energies else float("nan")

    return {
        "final_energy": float(result.fun),
        "optimal_params": result.x,
        "num_iterations": int(result.nit) if result.nit is not None else None,
        "num_function_evals": int(result.nfev) if result.nfev is not None else len(energies),
        "runtime_sec": runtime,
        "stability_std": stability,
        "energy_history": list(energies),
    }


if __name__ == "__main__":
    from qiskit.primitives import StatevectorEstimator
    from hamiltonian import get_qubit_hamiltonian
    from ansatz import get_ansatz

    qubit_op, problem, mapper = get_qubit_hamiltonian("H2", mapping="jordan_wigner")
    ansatz, hf_state = get_ansatz("uccsd", problem=problem, mapper=mapper)

    estimator = StatevectorEstimator()
    x0 = np.zeros(ansatz.num_parameters)

    for opt_name in ["cobyla", "slsqp", "spsa"]:
        history = {"energies": [], "params": [], "n_evals": 0}
        cost_fn = make_cost_function(estimator, ansatz, qubit_op, history)
        optimizer = get_optimizer(opt_name, maxiter=50)
        report = run_optimization(optimizer, cost_fn, x0, history)
        total = report["final_energy"] + problem.nuclear_repulsion_energy
        print(f"{opt_name:10s} total_energy={total:.6f}  "
              f"nfev={report['num_function_evals']}  runtime={report['runtime_sec']:.3f}s")