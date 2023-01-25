import jax.numpy as jnp
from jax import jacfwd
from typing import Tuple


def _inv_dist(pos: jnp.ndarray) -> jnp.ndarray:
    """Inverse square distance descriptor for a set of atoms given their cartesian coordinates

    Args:
        pos (jnp.ndarray): shape (N x 3) positions in a molecular configuration

    Returns:
        jnp.ndarray: vector of shape (N(N-1)/2, ) representing a geometric description of the configuration
    """
    ind1, ind2 = jnp.triu_indices(len(pos), k=1)
    pdiff = pos[ind1] - pos[ind2]  # pairwise difference
    sqdist = jnp.sum(pdiff * pdiff, axis=-1)
    return 1.0 / sqdist


def inv_dist(pos: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Inverse distance descriptor function that returns the descriptor as well as its Jacobian

    Args:
        pos (jnp.ndarray): shape (N x 3) positions in a molecular configuration

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: arrays of shape (N(N-1)/2, ) and (N(N-1)/2, N, 3), the descriptor and its Jacobian wrt positions, respectively
    """
    jac_inv_dist = jacfwd(_inv_dist)(pos)  # is jacfwd faster or jacrev?
    # flatten the last two dimensions of jac_inv_dist
    jac_inv_dist = jac_inv_dist.reshape(len(jac_inv_dist), -1)
    return _inv_dist(pos), jac_inv_dist
