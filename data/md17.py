from ase import Atoms
import jax.numpy as jnp
from typing import Tuple, List


def get_molecules(raw_filepath, n=1000) -> Tuple[List[Atoms], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Given the file path to an MD17 trajectory, take the first n configurations
    Returns in the form (list of ASE atoms objects, energies (shape Mx1), forces (shape MxNx3), atomic num. (shape N))
    """
    data = jnp.load(raw_filepath)
    if n == -1:
        n = len(data['E'])
    atoms = []
    z = data['z']
    R = data['R'][:n]
    for i in range(n):
        atoms.append(
            Atoms(positions=R[i], numbers=z)
        )
    return atoms, data['E'][:n], data['F'][:n], z