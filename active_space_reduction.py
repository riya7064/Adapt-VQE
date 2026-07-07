"""
active_space_reduction.py
--------------------------
Reduce the size of the electronic structure problem by freezing core
orbitals / selecting an active space, so larger molecules (H2O, NH3)
stay tractable on simulators.

Each transformation is its own function so hamiltonian.py / vqe.py can
opt in only where needed (e.g. H2 and LiH usually don't need this,
H2O / NH3 typically do).
"""

from __future__ import annotations

from typing import Optional, List

from qiskit_nature.second_q.problems import ElectronicStructureProblem
from qiskit_nature.second_q.transformers import (
    ActiveSpaceTransformer,
    FreezeCoreTransformer,
)


# --------------------------------------------------------------------------- #
# 1. Freeze-core reduction (removes chemically inert core orbitals)
# --------------------------------------------------------------------------- #
def freeze_core(problem: ElectronicStructureProblem, remove_orbitals: Optional[List[int]] = None):
    """
    Apply a FreezeCoreTransformer: freezes core electrons/orbitals
    (e.g. 1s of O, N, Li) and optionally removes extra unoccupied orbitals.
    """
    transformer = FreezeCoreTransformer(remove_orbitals=remove_orbitals)
    return transformer.transform(problem)


# --------------------------------------------------------------------------- #
# 2. Explicit active space selection
# --------------------------------------------------------------------------- #
def reduce_active_space(
    problem: ElectronicStructureProblem,
    num_electrons: int,
    num_spatial_orbitals: int,
    active_orbitals: Optional[List[int]] = None,
) -> ElectronicStructureProblem:
    """
    Restrict the problem to `num_electrons` electrons in
    `num_spatial_orbitals` spatial orbitals (an (n_elec, n_orb) active space),
    optionally pinning which orbital indices form the active space.
    """
    transformer = ActiveSpaceTransformer(
        num_electrons=num_electrons,
        num_spatial_orbitals=num_spatial_orbitals,
        active_orbitals=active_orbitals,
    )
    return transformer.transform(problem)


# --------------------------------------------------------------------------- #
# 3. Heuristic auto-selection (HOMO/LUMO window)
# --------------------------------------------------------------------------- #
def auto_select_active_space(
    problem: ElectronicStructureProblem,
    n_occupied: int = 2,
    n_virtual: int = 2,
) -> ElectronicStructureProblem:
    """
    Simple heuristic: keep `n_occupied` occupied spatial orbitals closest
    to the HOMO and `n_virtual` virtual orbitals closest to the LUMO.
    Good default for quick H2O / NH3 experiments where a full active
    space would need too many qubits.
    """
    num_particles = problem.num_particles
    num_alpha = num_particles[0]

    n_occupied = min(n_occupied, num_alpha)
    n_virtual = min(n_virtual, problem.num_spatial_orbitals - num_alpha)

    num_active_electrons = 2 * n_occupied
    num_active_orbitals = n_occupied + n_virtual

    return reduce_active_space(problem, num_active_electrons, num_active_orbitals)


# --------------------------------------------------------------------------- #
# 4. Reporting helper
# --------------------------------------------------------------------------- #
def report_problem_size(problem: ElectronicStructureProblem, label: str = "") -> dict:
    """Return a small dict describing how big the problem currently is."""
    info = {
        "label": label,
        "num_spatial_orbitals": problem.num_spatial_orbitals,
        "num_particles": problem.num_particles,
    }
    print(f"[{label}] spatial_orbitals={info['num_spatial_orbitals']} "
          f"particles={info['num_particles']}")
    return info


if __name__ == "__main__":
    from hamiltonian import build_driver, get_electronic_structure_problem

    driver = build_driver("H2O")
    problem = get_electronic_structure_problem(driver)
    report_problem_size(problem, "H2O full space")

    reduced = auto_select_active_space(problem, n_occupied=2, n_virtual=2)
    report_problem_size(reduced, "H2O active space (2 occ, 2 virt)")