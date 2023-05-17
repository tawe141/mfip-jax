import jax.numpy as jnp
from jax import disable_jit, jit, vmap
import jax
from typing import Callable
from functools import partial


@partial(jit, static_argnums=(0, 1))
def kernel_with_descriptor(k: Callable, descriptor_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, **kernel_kwargs):
    x1_ = descriptor_fn(x1)
    x2_ = descriptor_fn(x2)
    return k(x1_, x2_, **kernel_kwargs)


def _get_eps(x: jnp.ndarray):
    #if x.dtype == jnp.float32: 
    #    eps = 1e-4
    #elif x.dtype == jnp.float64: 
    #    eps = 1e-8
    #else:
    #    raise RuntimeError('unsupported type')
    #return eps
    return 1e-4

@partial(jit, static_argnums=0)
def finite_diff_dk_dx2(partial_f, x1, x2):
    """
    For two sample configurations (ie atom Cartesian coordinates) x1 and x2,
    find the Jacobian of the kernel as given by `partial_f` wrt x2 using 
    finite differences. Not intended for use in production, only for testing. 
    """
    eps = _get_eps(x1)

    init_val = jnp.zeros_like(x2)

    def inner_loop(i, val):
        def update(j, val):
            x2_ = x2.at[i, j].add(eps)
            fwd = partial_f(x1, x2_)
            x2_ = x2.at[i, j].add(-eps)
            rev = partial_f(x1, x2_)

            return val.at[i, j].set((fwd - rev) / 2 / eps)
        return jax.lax.fori_loop(0, x2.shape[1], update, val)

    return jax.lax.fori_loop(0, len(x2), inner_loop, init_val)


@partial(jit, static_argnums=0)
def finite_diff_hess_k(partial_f, x1, x2):
    """
    Finds the Hessian of K wrt x1 and x2 using finite differences. Not used for 
    production, only for testing.
    """
    
    eps = _get_eps(x1)

    init_val = jnp.zeros((*x1.shape, *x2.shape))

    def inner_loop(i, val):
        def update(j, val):
            x1_ = x1.at[i, j].add(eps)
            fwd = finite_diff_dk_dx2(partial_f, x1_, x2)
            x1_ = x1.at[i, j].add(-eps)
            rev = finite_diff_dk_dx2(partial_f, x1_, x2)

            return val.at[i, j].set((fwd - rev) / 2 / eps)
        return jax.lax.fori_loop(0, x1.shape[1], update, val)

    return jax.lax.fori_loop(0, len(x1), inner_loop, init_val)
   

@partial(jit, static_argnums=0)
def fd_hess_batch(partial_f, x1, x2):
    fn = vmap(
            vmap(lambda a,b: finite_diff_hess_k(partial_f, a, b), in_axes=(None, 0)),
            in_axes=(0, None)
    )
    return fn(x1, x2)
