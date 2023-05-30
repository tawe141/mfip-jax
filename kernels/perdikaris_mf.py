import pdb
import jax
from jax import vmap, jit, grad, jacfwd, jvp
import jax.numpy as jnp
from .hess import bilinear_hess, jac_K, jac_K_dx1, flatten
from functools import partial
from typing import Callable


@partial(jit, static_argnums=0)
def perdikaris_kernel(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, f_x1, f_x2, lp, lf, ld, w):
    w_ = jax.nn.softplus(w)
    return w_ * kernel_fn(x1, x2, lp) * kernel_fn(f_x1, f_x2, lf) + (1 - w_) * kernel_fn(x1, x2, ld)


#def get_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, E_x1, E_x2, F_x1, F_x2, lp, lf, ld) -> jnp.ndarray:
#    """
#    Copied over from `kernels/hess.py`. 
#
#    Finds the Hessian of the kernel matrix for a single sample `x1` and `x2`. Additional positional arguments 
#    for Monte Carlo samples of the energy evaluation, as necessary for the multifidelity kernel.
#
#    TODO: is this function really necessary? If it isn't, we could re-use a lot of the code developed for normal cov matrices
#    """
#    grad_k_x1 = grad(kernel_fn, argnums=0)
#    grad_k_x2 = grad(kernel_fn, argnums=1)
#    hess_k = jacfwd(grad_k_x2, argnums=0)
#    #pdb.set_trace()
#    #res = jnp.outer(jac_K(kernel_fn, x1, x2, dx2, **lp), grad_k_x1(E_x1, E_x2, **lf) * -F_x1)
#    #res += jnp.outer(jac_K(kernel_fn, x2, x1, dx1, **lp), grad_k_x2(E_x1, E_x2, **lf) * -F_x2)
#    res = jnp.outer(grad_k_x1(E_x1, E_x2, **lf) * -F_x1, jac_K(kernel_fn, x1, x2, dx2, **lp))
#    res += jnp.outer(jac_K_dx1(kernel_fn, x1, dx1, x2, **lp), grad_k_x2(E_x1, E_x2, **lf) * -F_x2)
#    res += hess_k(E_x1, E_x2, **lf) * jnp.outer(-F_x1, -F_x2)
#    #res += jnp.outer(jnp.dot(-jnp.expand_dims(F_x1, 1), hess_k(E_x1, E_x2, **lf)), -F_x2)
#    res += kernel_fn(E_x1, E_x2, **lf) * bilinear_hess(kernel_fn, x1, x2, dx1, dx2, **lp)
#    res += bilinear_hess(kernel_fn, x1, x2, dx1, dx2, **ld)
#    return res
#


def get_K_jac_analytical(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx2: jnp.array, E_x1, E_x2, F_x2, lp, lf, ld, w) -> jnp.ndarray:
    """
    Analogous to `get_K`; obtains the single Jacobian matrix of one row of x1 and x2
    TODO: should this be wrt x2 or x1? here thinking it's dx2
    """
    grad_k = grad(kernel_fn, argnums=1)
    res = kernel_fn(x1, x2, **lp) * grad_k(E_x1, E_x2, **lf) * -F_x2
    res += kernel_fn(E_x1, E_x2, **lf) * jac_K(kernel_fn, x1, x2, dx2, **lp)
    res += jac_K(kernel_fn, x1, x2, dx2, **ld)
    return res


def get_K_jac(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx2: jnp.ndarray, E_x1, E_x2, F_x2, lp, lf, ld, w=0.0) -> jnp.ndarray:
    dx, dE = grad(perdikaris_kernel, argnums=(2, 4))(kernel_fn, x1, x2, E_x1, E_x2, lp, lf, ld, w)
    #pdb.set_trace()
    dx = dx @ dx2
    dE = dE * -F_x2
    #k_fn = partial(perdikaris_kernel, kernel_fn=kernel_fn, x1=x1, f_x1=E_x1, lp=lp, lf=lf, ld=ld)
    #k_fn_E_x2 = partial(k_fn, x2=x2)
    #dE = jvp(k_fn_E_x2, (E_x2, ), (-F_x2, ))
    #dE = grad(k_fn_E_x2)(E_x2) * -F_x2
    #k_fn_x2 = partial(k_fn, f_x2=E_x2)
    #dx = grad(k_fn_x2)(x2)
    return dx + dE


def get_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.ndarray, E_x1, E_x2, F_x1, F_x2, lp, lf, ld, w=0.0):
    # in principle, I should be able to re-use get_K_jac...
    dx, dE = jacfwd(get_K_jac, argnums=(1, 4))(kernel_fn, x1, x2, dx2, E_x1, E_x2, F_x2, lp, lf, ld, w)
    dx = (dx @ dx1).T
    dE = jnp.outer(-F_x1, dE)
    return dx + dE


def _get_full_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, E_sample_x1, E_sample_x2, F_sample_x1, F_sample_x2, **kernel_kwargs) -> jnp.ndarray:
    #E_sample_x1, rng_key = _normal_sample(E_mu_x1, E_var_x1, rng_key)
    #E_sample_x2, rng_key = _normal_sample(E_mu_x2, E_var_x2, rng_key)
    hess_kernel_fn = partial(get_K, kernel_fn, **kernel_kwargs)

    func = vmap(
        vmap(hess_kernel_fn, in_axes=(None, 0, None, 0, None, 0, None, 0)),
        in_axes=(0, None, 0, None, 0, None, 0, None)
    )
    return func(x1, x2, dx1, dx2, E_sample_x1, E_sample_x2, F_sample_x1, F_sample_x2)


def get_diag_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, E_sample_x1, E_sample_x2, F_sample_x1, F_sample_x2, **kernel_kwargs) -> jnp.ndarray:
    """
    Calculates the diagonal of the Hessian matrix for batched x1 and x2
    NOTE: this function works by getting the diagonal of `get_K`, but it's more efficient to have `get_K` get the diagonal directly
    so it computes a single vector rather than take a diagonal of the cov matrix.
    """
    K_partial = partial(get_K, **kernel_kwargs)
    diag_fn = lambda k,a1,a2,da1,da2,e1,e2, f1, f2: jnp.diagonal(K_partial(k, a1, a2, da1, da2, e1, e2, f1, f2))
    func = vmap(diag_fn, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0), out_axes=0)
    return func(kernel_fn, x1, x2, dx1, dx2, E_sample_x1, E_sample_x2, F_sample_x1, F_sample_x2).flatten()


def get_full_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, E_sample_x1, E_sample_x2, F_sample_x1, F_sample_x2, iterative=False, **kernel_kwargs) -> jnp.ndarray:
    # disabled iterative kernel matrix building for now

    #K = jax.lax.cond(iterative, partial(_get_full_K_iterative, kernel_fn, **kernel_kwargs), partial(_get_full_K, kernel_fn, **kernel_kwargs), x1, x2, dx1, dx2, E_sample_x1, E_sample_x2, F_sample_x1, F_sample_x2)
    #K = jax.lax.cond(iterative, _get_full_K, _get_full_K_iterative, kernel_fn, x1, x2, dx1, dx2, E_sample_x1, E_sample_x2, **kernel_kwargs)
    K = _get_full_K(kernel_fn, x1, x2, dx1, dx2, E_sample_x1, E_sample_x2, F_sample_x1, F_sample_x2, **kernel_kwargs)
    m1, m2, d1, d2 = K.shape
    return flatten(K, m1, d1, m2, d2)


def _get_full_K_iterative(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, E_sample_x1, E_sample_x2, F_sample_x1, F_sample_x2, **kernel_kwargs) -> jnp.ndarray:
    """
    Getting the full kernel matrix can be brutally memory intensive. Iterate over x1/dx1 and build the matrix row-by-row
    """
    get_K_fn = partial(get_K, kernel_fn, **kernel_kwargs)
    vec_over_x2 = vmap(get_K_fn, in_axes=(None, 0, None, 0, None, 0, None, 0), out_axes=0)
    #pdb.set_trace()
    def calc_row(i, val):
        val = val.at[i].set(vec_over_x2(x1[i], x2, dx1[i], dx2, E_sample_x1[i], E_sample_x2, F_sample_x1[i], F_sample_x2))
        return val
    init_val = jnp.zeros((len(x1), len(x2), dx1.shape[-1], dx2.shape[-1]))
    return jax.lax.fori_loop(0, len(x1), calc_row, init_val)


def _get_jac_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx2: jnp.array, E_sample_x1, E_sample_x2, F_sample_x2, **kernel_kwargs) -> jnp.ndarray:
    """
    Returns the Jacobian of K wrt x2, ie d/dx2 K(x1, x2) in shape (N, N, D, E), where N = # in batch, D is dimensionality of the descriptor, E is the dimensionality of the inputs to the descriptor 
    """
    #K_partial = partial(jac_K, **kernel_kwargs)
    get_K_jac_fn = partial(get_K_jac, kernel_fn, **kernel_kwargs)
    func = vmap(
        vmap(get_K_jac_fn, in_axes=(None, 0, 0, None, 0, 0)),
        in_axes=(0, None, None, 0, None, None)
    )
    return func(x1, x2, dx2, E_sample_x1, E_sample_x2, F_sample_x2)


def get_jac_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx2: jnp.array, E_sample_x1, E_sample_x2, F_sample_x2, **kernel_kwargs) -> jnp.ndarray:
    K = _get_jac_K(kernel_fn, x1, x2, dx2, E_sample_x1, E_sample_x2, F_sample_x2, **kernel_kwargs)
    return K.reshape(len(x1), len(x2) * dx2.shape[-1])
