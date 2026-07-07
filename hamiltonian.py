"""
hamiltonian.py
--------------
Everything needed to go from a molecule name -> qubit Hamiltonian.

Pipeline:
    geometry.txt --> parse_geometry_file()
                  --> build_driver()
                  --> get_electronic_structure_problem()
                  --> get_second_q_hamiltonian()
                  --> map_to_qubit_hamiltonian()

Each stage is its own function so any other module (ansatz.py,
optimizer.py, active_space_reduction.py, vqe.py ...) can call exactly
the piece it needs instead of one giant "do everything" call.
"""

from __future__ import annotations

import os
from typing import Optional

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    BravyiKitaevMapper,
    QubitMapper,
)
from qiskit_nature.units import DistanceUnit
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit.quantum_info import SparsePauliOp

DEFAULT_GEOMETRY_FILE = os.path.join(os.path.dirname(__file__), "geometry.txt")


# --------------------------------------------------------------------------- #
# 1. Geometry parsing
# --------------------------------------------------------------------------- #
def parse_geometry_file(molecule_name: str, geometry_file: str = DEFAULT_GEOMETRY_FILE) -> dict:
    """
    Parse geometry.txt and return the block for `molecule_name`.

    Returns
    -------
    dict with keys: 'atom' (PySCF-style atom string), 'basis', 'charge', 'multiplicity'
    """
    if not os.path.exists(geometry_file):
        raise FileNotFoundError(f"Geometry file not found: {geometry_file}")

    blocks = {}
    current_name = None
    current_lines = []

    with open(geometry_file, "r") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()  # strip comments
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                if current_name is not None:
                    blocks[current_name] = current_lines
                current_name = line[1:-1].strip().upper()
                current_lines = []
            else:
                current_lines.append(line)
        if current_name is not None:
            blocks[current_name] = current_lines

    key = molecule_name.strip().upper()
    if key not in blocks:
        raise KeyError(
            f"Molecule '{molecule_name}' not found in {geometry_file}. "
            f"Available: {list(blocks.keys())}"
        )

    basis = "sto3g"
    charge = 0
    multiplicity = 1
    atom_lines = []

    for line in blocks[key]:
        if line.lower().startswith("basis:"):
            basis = line.split(":", 1)[1].strip()
        elif line.lower().startswith("charge:"):
            charge = int(line.split(":", 1)[1].strip())
        elif line.lower().startswith("multiplicity:"):
            multiplicity = int(line.split(":", 1)[1].strip())
        else:
            atom_lines.append(line)

    atom_str = "; ".join(atom_lines)

    return {
        "atom": atom_str,
        "basis": basis,
        "charge": charge,
        "multiplicity": multiplicity,
    }


# --------------------------------------------------------------------------- #
# 2. Driver construction
# --------------------------------------------------------------------------- #
def build_driver(
    molecule_name: str,
    geometry_file: str = DEFAULT_GEOMETRY_FILE,
    basis_override: Optional[str] = None,
) -> PySCFDriver:
    """Build a PySCFDriver for the requested molecule."""
    geom = parse_geometry_file(molecule_name, geometry_file)
    basis = basis_override or geom["basis"]

    driver = PySCFDriver(
        atom=geom["atom"],
        basis=basis,
        charge=geom["charge"],
        spin=geom["multiplicity"] - 1,  # PySCF wants 2S, multiplicity = 2S+1
        unit=DistanceUnit.ANGSTROM,
    )
    return driver


# --------------------------------------------------------------------------- #
# 3. Electronic structure problem
# --------------------------------------------------------------------------- #
def get_electronic_structure_problem(driver: PySCFDriver) -> ElectronicStructureProblem:
    """Run the classical driver (PySCF) and return the electronic structure problem."""
    return driver.run()


# --------------------------------------------------------------------------- #
# 4. Second-quantized Hamiltonian
# --------------------------------------------------------------------------- #
def get_second_q_hamiltonian(problem: ElectronicStructureProblem):
    """Return the fermionic (second-quantized) Hamiltonian operator."""
    return problem.hamiltonian.second_q_op()


# --------------------------------------------------------------------------- #
# 5. Qubit mapping
# --------------------------------------------------------------------------- #
def get_mapper(mapping: str = "jordan_wigner") -> QubitMapper:
    """Return a qubit mapper instance by name."""
    mapping = mapping.lower().replace("-", "_").replace(" ", "_")
    if mapping in ("jordan_wigner", "jw"):
        return JordanWignerMapper()
    if mapping in ("bravyi_kitaev", "bk"):
        return BravyiKitaevMapper()
    raise ValueError(f"Unknown mapping '{mapping}'. Use 'jordan_wigner' or 'bravyi_kitaev'.")


def map_to_qubit_hamiltonian(second_q_op, mapping: str = "jordan_wigner") -> SparsePauliOp:
    """Map a fermionic operator to a qubit operator (SparsePauliOp) using JW or BK."""
    mapper = get_mapper(mapping)
    return mapper.map(second_q_op)


# --------------------------------------------------------------------------- #
# 6. Convenience end-to-end pipeline (still just plumbing, no VQE logic here)
# --------------------------------------------------------------------------- #
def get_qubit_hamiltonian(
    molecule_name: str,
    mapping: str = "jordan_wigner",
    geometry_file: str = DEFAULT_GEOMETRY_FILE,
    basis_override: Optional[str] = None,
):
    """
    Convenience wrapper: molecule name -> (qubit_hamiltonian, problem, mapper).

    Kept separate from vqe.py -- this only prepares the Hamiltonian, it does
    not run any VQE.
    """
    driver = build_driver(molecule_name, geometry_file, basis_override)
    problem = get_electronic_structure_problem(driver)
    second_q_op = get_second_q_hamiltonian(problem)
    mapper = get_mapper(mapping)
    qubit_op = mapper.map(second_q_op)
    return qubit_op, problem, mapper


# --------------------------------------------------------------------------- #
# 7. Classical reference energy (for validation, Phase 1 deliverable)
# --------------------------------------------------------------------------- #
def get_reference_energy(problem: ElectronicStructureProblem) -> float:
    """
    Exact diagonalization (FCI within the active space) reference energy,
    including nuclear repulsion. Used to validate VQE results.
    """
    solver = NumPyMinimumEigensolver()
    from qiskit_nature.second_q.algorithms import GroundStateEigensolver

    mapper = JordanWignerMapper()
    gse = GroundStateEigensolver(mapper, solver)
    result = gse.solve(problem)
    return float(result.total_energies[0].real)


def get_nuclear_repulsion_energy(problem: ElectronicStructureProblem) -> float:
    """Nuclear repulsion energy (constant offset to add to electronic energy)."""
    return problem.nuclear_repulsion_energy


if __name__ == "__main__":
    # quick smoke test
    for mol in ["H2", "LiH"]:
        qubit_op, problem, mapper = get_qubit_hamiltonian(mol, mapping="jordan_wigner")
        ref = get_reference_energy(problem)
        print(f"{mol}: num_qubits={qubit_op.num_qubits}, reference_energy={ref:.6f} Ha")