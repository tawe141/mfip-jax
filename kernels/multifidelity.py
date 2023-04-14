import pdb
import jax
from jax import vmap, jit
import jax.numpy as jnp
from .hess import bilinear_hess, jac_K, flatten
from functools import partial
from typing import Callable


@partial(jit, static_argnums=0)
def perdikaris_kernel(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, f_x1, f_x2, lp, lf, ld):
    return kernel_fn(x1, x2, lp) * kernel_fn(f_x1, f_x2, lf) + kernel_fn(x1, x2, ld)


def get_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, f_x1, f_x2, **kernel_kwargs) -> jnp.ndarray:
    """
    Copied over from `kernels/hess.py`. 

    Finds the Hessian of the kernel matrix for a single sample `x1` and `x2`. Additional positional arguments 
    for Monte Carlo samples of the energy evaluation, as necessary for the multifidelity kernel.

    TODO: is this function really necessary? If it isn't, we could re-use a lot of the code developed for normal cov matrices
    """
    return bilinear_hess(kernel_fn, x1, x2, dx1, dx2, f_x1=f_x1, f_x2=f_x2, **kernel_kwargs)


def get_K_jac(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx2: jnp.array, f_x1, f_x2, **kernel_kwargs) -> jnp.ndarray:
    """
    Analogous to `get_K`; obtains the single Jacobian matrix of one row of x1 and x2
    TODO: terrible naming convention...
    """
    return jac_K(kernel_fn, x1, x2, dx2, f_x1=f_x2, f_x2=f_x2)


def _get_full_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, E_sample_x1, E_sample_x2, iterative=False, **kernel_kwargs) -> jnp.ndarray:
    #E_sample_x1, rng_key = _normal_sample(E_mu_x1, E_var_x1, rng_key)
    #E_sample_x2, rng_key = _normal_sample(E_mu_x2, E_var_x2, rng_key)
    k_fn = partial(perdikaris_kernel, kernel_fn, **kernel_kwargs)

    func = vmap(
        vmap(get_K, in_axes=(None, None, 0, None, 0, None, 0)),
        in_axes=(None, 0, None, 0, None, 0, None)
    )
    return func(k_fn, x1, x2, dx1, dx2, E_sample_x1, E_sample_x2)


def get_diag_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, E_sample_x1, E_sample_x2, **kernel_kwargs) -> jnp.ndarray:
    """
    Calculates the diagonal of the Hessian matrix for batched x1 and x2
    NOTE: this function works by getting the diagonal of `get_K`, but it's more efficient to have `get_K` get the diagonal directly
    so it computes a single vector rather than take a diagonal of the cov matrix.
    """
    k_fn = partial(perdikaris_kernel, kernel_fn, **kernel_kwargs)
    K_partial = partial(get_K, **kernel_kwargs)
    diag_fn = lambda k,a1,a2,da1,da2,e1,e2: jnp.diagonal(K_partial(k, a1, a2, da1, da2, e1, e2))
    func = vmap(diag_fn, in_axes=(None, 0, 0, 0, 0, 0, 0), out_axes=0)
    return func(k_fn, x1, x2, dx1, dx2, E_sample_x1, E_sample_x2).flatten()


def get_full_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, E_sample_x1, E_sample_x2, iterative=False, **kernel_kwargs) -> jnp.ndarray:
    K = jax.lax.cond(iterative, partial(_get_full_K_iterative, kernel_fn, **kernel_kwargs), partial(_get_full_K, kernel_fn, **kernel_kwargs), x1, x2, dx1, dx2, E_sample_x1, E_sample_x2)
    #K = jax.lax.cond(iterative, _get_full_K, _get_full_K_iterative, kernel_fn, x1, x2, dx1, dx2, E_sample_x1, E_sample_x2, **kernel_kwargs)
    #K = _get_full_K(kernel_fn, x1, x2, dx1, dx2, E_sample_x1, E_sample_x2, **kernel_kwargs)
    m1, m2, d1, d2 = K.shape
    return flatten(K, m1, d1, m2, d2)


def _get_full_K_iterative(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, E_sample_x1, E_sample_x2, **kernel_kwargs) -> jnp.ndarray:
    """
    Getting the full kernel matrix can be brutally memory intensive. Iterate over x1/dx1 and build the matrix row-by-row
    """
    k_fn = partial(perdikaris_kernel, kernel_fn, **kernel_kwargs)
    vec_over_x2 = vmap(get_K, in_axes=(None, None, 0, None, 0, None, 0), out_axes=0)
    #pdb.set_trace()
    def calc_row(i, val):
        val = val.at[i].set(vec_over_x2(k_fn, x1[i], x2, dx1[i], dx2, E_sample_x1[i], E_sample_x2))
        return val
    init_val = jnp.zeros((len(x1), len(x2), dx1.shape[-1], dx2.shape[-1]))
    return jax.lax.fori_loop(0, len(x1), calc_row, init_val)


def _get_jac_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx2: jnp.array, E_sample_x1, E_sample_x2, **kernel_kwargs) -> jnp.ndarray:
    """
    Returns the Jacobian of K wrt x2, ie d/dx2 K(x1, x2) in shape (N, N, D, E), where N = # in batch, D is dimensionality of the descriptor, E is the dimensionality of the inputs to the descriptor 
    """
    #K_partial = partial(jac_K, **kernel_kwargs)
    k_fn = partial(perdikaris_kernel, kernel_fn, **kernel_kwargs)
    func = vmap(
        vmap(get_K_jac, in_axes=(None, None, 0, 0, None, 0)),
        in_axes=(None, 0, None, None, 0, None)
    )
    return func(k_fn, x1, x2, dx2, E_sample_x1, E_sample_x2)


def get_jac_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx2: jnp.array, E_sample_x1, E_sample_x2, **kernel_kwargs) -> jnp.ndarray:
    K = _get_jac_K(kernel_fn, x1, x2, dx2, E_sample_x1, E_sample_x2, **kernel_kwargs)
    return K.reshape(len(x1), len(x2) * dx2.shape[-1])
