from functools import partial
import jax
from jax import jacfwd, grad, jvp, vmap, jit
import jax.numpy as jnp
from typing import Callable


@jit
def rbf(x1: jnp.ndarray, x2: jnp.ndarray, l: float) -> float:
    diff = x1 / l - x2 / l
    sqdist = jnp.sum(diff * diff)
    return jnp.exp(-sqdist)


def explicit_hess_fn(kernel_fn: Callable) -> Callable:
    return jacfwd(grad(kernel_fn, argnums=0), argnums=1)


@partial(jit, static_argnames=['kernel_fn'])
def explicit_hess(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, l: float) -> jnp.ndarray:
    hess_fn = explicit_hess_fn(kernel_fn)
    return hess_fn(x1, x2, l)


@partial(jit, static_argnames=['kernel_fn'])
def hvp(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx2: jnp.array, **kernel_kwargs) -> jnp.ndarray:
    kernel_partial = partial(kernel_fn, **kernel_kwargs)
    def grad_kernel_vp(x1, x2, dx2):
        gradx1 = lambda x2: grad(kernel_partial)(x1, x2)  # should grad use argnums?
        return jvp(gradx1, (x2,), (dx2,))
    return vmap(grad_kernel_vp, in_axes=(None, None, 1), out_axes=1)(x1, x2, dx2)[1]


if __name__ == "__main__":
    key = jax.random.PRNGKey(42)
    a = jax.random.normal(key, shape=(16,))
    res = explicit_hess(rbf, a, a, 1.0)
    print(res.shape)
    print(jnp.allclose(res, 2.0 * jnp.eye(16)))

    # vectorize
    a = jax.random.normal(key, shape=(4, 16))
    v_explicit_hess = jax.vmap(
        jax.vmap(explicit_hess, in_axes=(None, 0, None, None), out_axes=0),
        in_axes=(None, None, 0, None),
        out_axes=1
    )
    vres = v_explicit_hess(rbf, a, a, 1.0)
    print(vres.shape)
    print(jnp.allclose(vres[0, 0], res))
    
    # how does the hvp work?
    new_key, subkey = jax.random.split(key)
    a = a[0]
    da = jax.random.normal(new_key, shape=(16, 8))  # pretend this is the jacobian of `a`
    hvp_res = hvp(rbf, a, a, da, l=1.0)[1]
    print(hvp_res.shape)
    print(jnp.allclose(res @ da, hvp_res))

    # performance comparison, done with the %timeit magic in ipython
    # %timeit (explicit_hess(rbf, a, a, 1.0) @ da).block_until_ready()
    # > 1.24 ms ± 50.7 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    # %timeit hvp(rbf, a, a, da, l=1.0)[1].block_until_ready()
    # > 1.27 ms ± 67.9 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    # no performance improvement... but hopefully there's some kind of memory usage decrease? 


