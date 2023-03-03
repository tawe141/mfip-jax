import jax.numpy as jnp
from jax import jacfwd, value_and_grad, vmap, jit
from jax.lax import cond
from typing import Tuple
from .normalize import normalized_descriptors
from functools import partial
import pdb


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
    dist = jnp.sqrt(sqdist)
    return 1.0 / dist


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

"""
def get_descriptors(desc, pos, normalize: bool = False, mu=None, std=None):
    norm_fn = partial(_get_normalized_descriptors, desc)
    fn = partial(_get_descriptors, desc)
    compute_statistics = mu is None and std is None
    return cond(
            normalize, 
            cond(compute_statistics, ), 
            fn, 
            pos, mu, std)


@partial(jit, static_argnums=(0,))
def _get_descriptors(desc, pos, *args):
    descriptors = vmap(desc)(pos)
    jac_desc = vmap(jacfwd(desc))(pos).reshape(descriptors.shape[0], descriptors.shape[1], -1)
    return descriptors, jac_desc, mu, std


@partial(jit, static_argnums=(0,))
def _get_normalized_descriptors(desc, pos, mu, std):
    descriptors, (mu, std) = normalized_descriptors(vmap(desc), pos, mu, std)
    norm_desc_fn = lambda x: (desc(x) - mu) / std
    jac_desc = vmap(jacfwd(norm_desc_fn))(pos).reshape(descriptors.shape[0], descriptors.shape[1], -1)
    return descriptors, jac_desc, mu, std

def get_normalized_descriptors_known_mu_std(desc, pos, mu, std):
    descriptors = normalized_descriptors_known_mu_std(vmap(desc), pos)
    norm_desc_fn = lambda x: (desc(x) - mu) / std
    jac_desc = vmap(jacfwd(norm_desc_fn))(pos).reshape(descriptors.shape[0], descriptors.shape[1], -1)
    return descriptors, jac_desc, mu, std
"""
