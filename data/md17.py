from ase import Atoms
import jax.numpy as jnp
import jax
from typing import Tuple, List


def get_molecules(raw_filepath, n=1000, shuffle=False) -> Tuple[List[Atoms], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Given the file path to an MD17 trajectory, take the first n configurations
    Returns in the form (list of ASE atoms objects, energies (shape Mx1), forces (shape MxNx3), atomic num. (shape N))
    """
    data = jnp.load(raw_filepath)
    if n == -1:
        n = len(data['E'])
    atoms = []
    z = data['z']
    if shuffle:
        a = jnp.arange(len(data['R']))
        key = jax.random.PRNGKey(42)
        jax.random.shuffle(key, a)
        #idx = a[:n]
        idx = jax.lax.dynamic_slice_in_dim(a, 0, n)
    else:
        idx = jnp.arange(n)
    R = data['R'][idx]
    for i in range(n):
        atoms.append(
            Atoms(positions=R[i], numbers=z)
        )
    return atoms, data['E'][idx], data['F'][idx], z
