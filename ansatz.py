"""
ansatz.py
---------
Builders for every ansatz used in the Phase 2A comparison study:
    - Hardware-Efficient Ansatz (HEA)      -> build_hea_ansatz()
    - UCCSD (chemistry-inspired)           -> build_uccsd_ansatz()
    - TwoLocal (generic configurable)      -> build_twolocal_ansatz()

Plus:
    - build_hartree_fock_state()  : HF reference state, used as the
                                     initial_state for UCCSD (and optionally
                                     prepended to HEA/TwoLocal).
    - get_ansatz()                : single dispatcher by name, so
                                     vqe.py can loop over ansatz names.
    - ansatz_stats()               : circuit-depth / parameter-count report,
                                     used for the "Accuracy / depth / params /
                                     runtime" comparison table.
"""

from __future__ import annotations

from typing import Optional, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal, EfficientSU2
from qiskit.circuit import Parameter
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.mappers import QubitMapper
from qiskit_nature.second_q.problems import ElectronicStructureProblem


# --------------------------------------------------------------------------- #
# Hartree-Fock reference state
# --------------------------------------------------------------------------- #
def build_hartree_fock_state(
    problem: ElectronicStructureProblem, mapper: QubitMapper
) -> HartreeFock:
    """Return the Hartree-Fock circuit for this problem/mapper (used as a
    starting reference state, e.g. for UCCSD or for HF-informed init)."""
    return HartreeFock(problem.num_spatial_orbitals, problem.num_particles, mapper)


# --------------------------------------------------------------------------- #
# 1. Hardware-Efficient Ansatz (HEA)
# --------------------------------------------------------------------------- #
def build_hea_ansatz(
    num_qubits: int,
    reps: int = 3,
    entanglement: str = "linear",
    initial_state: Optional[QuantumCircuit] = None,
) -> QuantumCircuit:
    """
    Hardware-efficient ansatz built from EfficientSU2 (alternating
    single-qubit rotations + entangling CNOT layers). Cheap in depth,
    no chemistry structure baked in.
    """
    hea = EfficientSU2(num_qubits, reps=reps, entanglement=entanglement, insert_barriers=False)
    if initial_state is not None:
        circuit = QuantumCircuit(num_qubits)
        circuit.compose(initial_state, inplace=True)
        circuit.compose(hea, inplace=True)
        return circuit
    return hea


# --------------------------------------------------------------------------- #
# 2. UCCSD (chemistry-inspired, unitary coupled cluster singles & doubles)
# --------------------------------------------------------------------------- #
def build_uccsd_ansatz(
    problem: ElectronicStructureProblem,
    mapper: QubitMapper,
    reps: int = 1,
    use_hf_initial_state: bool = True,
) -> Tuple[UCCSD, Optional[HartreeFock]]:
    """
    Build a UCCSD ansatz for the given problem/mapper.
    Returns (ansatz, hf_state_used_as_reference).
    """
    hf_state = build_hartree_fock_state(problem, mapper) if use_hf_initial_state else None
    ansatz = UCCSD(
        problem.num_spatial_orbitals,
        problem.num_particles,
        mapper,
        initial_state=hf_state,
        reps=reps,
    )
    return ansatz, hf_state


# --------------------------------------------------------------------------- #
# 3. TwoLocal (generic configurable hardware-efficient-style ansatz)
# --------------------------------------------------------------------------- #
def build_twolocal_ansatz(
    num_qubits: int,
    rotation_blocks=("ry", "rz"),
    entanglement_blocks: str = "cx",
    entanglement: str = "linear",
    reps: int = 3,
    initial_state: Optional[QuantumCircuit] = None,
) -> QuantumCircuit:
    """
    Generic TwoLocal ansatz. Distinct from build_hea_ansatz(): here the
    rotation/entangling gate set is fully configurable, giving a separate
    comparison point from the fixed EfficientSU2-based HEA.
    """
    twolocal = TwoLocal(
        num_qubits,
        rotation_blocks=list(rotation_blocks),
        entanglement_blocks=entanglement_blocks,
        entanglement=entanglement,
        reps=reps,
        insert_barriers=False,
    )
    if initial_state is not None:
        circuit = QuantumCircuit(num_qubits)
        circuit.compose(initial_state, inplace=True)
        circuit.compose(twolocal, inplace=True)
        return circuit
    return twolocal


# --------------------------------------------------------------------------- #
# 4. Dispatcher, so vqe.py can just do get_ansatz("uccsd", problem=..., mapper=...)
# --------------------------------------------------------------------------- #
def get_ansatz(
    name: str,
    num_qubits: Optional[int] = None,
    problem: Optional[ElectronicStructureProblem] = None,
    mapper: Optional[QubitMapper] = None,
    reps: int = 3,
    use_hf_initial_state: bool = True,
):
    """
    Single entry point for Phase 2A. Returns the ansatz circuit
    (for UCCSD also returns the HF reference state used).

    name : "hea" | "uccsd" | "twolocal"
    """
    name = name.strip().lower()

    if name == "uccsd":
        if problem is None or mapper is None:
            raise ValueError("UCCSD requires `problem` and `mapper`.")
        return build_uccsd_ansatz(problem, mapper, reps=reps, use_hf_initial_state=use_hf_initial_state)

    if num_qubits is None:
        if problem is None or mapper is None:
            raise ValueError("Provide `num_qubits`, or `problem`+`mapper` to infer it.")
        num_qubits = mapper.map(problem.hamiltonian.second_q_op()).num_qubits

    hf_state = None
    if use_hf_initial_state and problem is not None and mapper is not None:
        hf_state = build_hartree_fock_state(problem, mapper)

    if name == "hea":
        return build_hea_ansatz(num_qubits, reps=reps, initial_state=hf_state), hf_state
    if name == "twolocal":
        return build_twolocal_ansatz(num_qubits, reps=reps, initial_state=hf_state), hf_state

    raise ValueError(f"Unknown ansatz '{name}'. Use 'hea', 'uccsd', or 'twolocal'.")


# --------------------------------------------------------------------------- #
# 5. Stats helper (depth, param count) for the comparison table
# --------------------------------------------------------------------------- #
def ansatz_stats(ansatz: QuantumCircuit, basis_gates=("rz", "sx", "x", "cx")) -> dict:
    """Return depth / parameter-count / gate-count stats after transpiling
    to a fixed basis, so different ansatz types are compared fairly."""
    from qiskit import transpile

    transpiled = transpile(ansatz, basis_gates=list(basis_gates), optimization_level=1)
    return {
        "num_parameters": ansatz.num_parameters,
        "num_qubits": ansatz.num_qubits,
        "raw_depth": ansatz.depth(),
        "transpiled_depth": transpiled.depth(),
        "transpiled_gate_count": sum(transpiled.count_ops().values()),
        "gate_breakdown": dict(transpiled.count_ops()),
    }


if __name__ == "__main__":
    from hamiltonian import get_qubit_hamiltonian

    qubit_op, problem, mapper = get_qubit_hamiltonian("H2", mapping="jordan_wigner")

    for ansatz_name in ["hea", "twolocal", "uccsd"]:
        ansatz, _ = get_ansatz(ansatz_name, problem=problem, mapper=mapper, reps=2)
        stats = ansatz_stats(ansatz)
        print(ansatz_name, stats)