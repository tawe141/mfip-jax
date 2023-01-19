from functools import partial
import jax
from jax import jacfwd, grad, jvp, vmap, jit
import jax.numpy as jnp
from typing import Callable


@jit
def rbf(x1: jnp.ndarray, x2: jnp.ndarray, l: float) -> float:
    diff = x1 / l - x2 / l
    sqdist = jnp.sum(jnp.power(diff, 2))
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


def get_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, **kernel_kwargs) -> jnp.ndarray:
    return dx1.T @ hvp(kernel_fn, x1, x2, dx2, **kernel_kwargs)


def get_full_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, **kernel_kwargs) -> jnp.ndarray:
    # kernel_kwargs['kernel_fn'] = kernel_fn
    K_partial = partial(get_K, **kernel_kwargs)
    func = vmap(
        vmap(K_partial, in_axes=(None, None, 0, None, 0), out_axes=0),
        in_axes=(None, 0, None, 0, None),
        out_axes=0
    )
    # return func(x1=x1, x2=x2, dx1=dx1, dx2=dx2)
    return func(kernel_fn, x1, x2, dx1, dx2)


@partial(jit, static_argnames=['kernel_fn'])
def reordered_hvp(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx2: jnp.array, **kernel_kwargs):
    # grad, multiply, then grad again
    kernel_partial = partial(kernel_fn, **kernel_kwargs)
    def grad_kernel_vp(x1, x2, dx2):
        gradx1 = lambda x2: grad(kernel_partial)(x1, x2) @ dx2  # should grad use argnums?
        return grad(gradx1)(x2)
    return vmap(grad_kernel_vp, in_axes=(None, None, 1), out_axes=1)(x1, x2, dx2)


if __name__ == "__main__":
    n = 512
    key = jax.random.PRNGKey(42)
    a = jax.random.normal(key, shape=(n,))
    res = explicit_hess(rbf, a, a, 1.0)
    print(res.shape)
    print(jnp.allclose(res, 2.0 * jnp.eye(n)))

    # vectorize
    a = jax.random.normal(key, shape=(4, n))
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
    da = jax.random.normal(new_key, shape=(n, 8))  # pretend this is the jacobian of `a`
    hvp_res = hvp(rbf, a, a, da, l=1.0)
    print(hvp_res.shape)
    print(jnp.allclose(res @ da, hvp_res))

    reordered_res = reordered_hvp(rbf, a, a, da, l=1.0)
    print(reordered_res.shape)
    print(jnp.allclose(hvp_res, reordered_res))

    # performance comparison, done with the %timeit magic in ipython
    # %timeit (explicit_hess(rbf, a, a, 1.0) @ da).block_until_ready()
    # > 1.24 ms ± 50.7 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    # %timeit hvp(rbf, a, a, da, l=1.0)[1].block_until_ready()
    # > 1.27 ms ± 67.9 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    # no performance improvement... but hopefully there's some kind of memory usage decrease? 


