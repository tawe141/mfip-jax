from functools import partial
import jax
from jax import jacfwd, grad, jvp, vjp, vmap, jit, pmap
import jax.numpy as jnp
from typing import Callable


def flatten(x: jnp.ndarray, m1: int, d1: int, m2: int, d2: int):
    return x.transpose(0,2,1,3).reshape(m1*d1, m2*d2)


@jit
# @jax.checkpoint
def rbf(x1: jnp.ndarray, x2: jnp.ndarray, l: float) -> float:
    l_ = jax.nn.softplus(l)
    diff = x1 / l_ - x2 / l_
    sqdist = jnp.sum(jnp.square(diff))
    # sqdist = jnp.sum(jnp.power(diff, 2))
    # sqdist = jnp.sum(diff * diff)
    return jnp.exp(-sqdist)


@jit
def scaled_rbf(x1: jnp.ndarray, x2: jnp.ndarray, l: float, prefactor: float = 1.0) -> float:
    return prefactor * rbf(x1, x2, l)


# def kernel_with_descriptor(descriptor_fn, kernel_fn, x1, x2, **kernel_kwargs):
#     x1_ = descriptor_fn(x1)
#     x2_ = descriptor_fn(x2)
#     return kernel_fn(x1_, x2_, kernel_kwargs)


def bilinear_hess(kernel_fn, x1, x2, dx1, dx2, **kernel_kwargs):
    def jac_x2_vec_dot(kernel_fn, x1, x2, dx2, **kernel_kwargs):
        # finds the jacobian wrt x2 and applies linear map according to dx2
        partial_kernel = partial(kernel_fn, x1, **kernel_kwargs)
        jvp_col = lambda a: jvp(partial_kernel, (x2, ), (a, ))[1]
        jvp_columnwise = vmap(jvp_col, in_axes=1)
        return jvp_columnwise(dx2)

    partial_jac = partial(jac_x2_vec_dot, kernel_fn, x2=x2, dx2=dx2, **kernel_kwargs)
    jvp_col = lambda a: jvp(partial_jac, (x1, ), (a, ))[1]
    jvp_columnwise = vmap(jvp_col, in_axes=1)
    return jvp_columnwise(dx1)


def explicit_hess_fn(kernel_fn: Callable) -> Callable:
    return jacfwd(grad(kernel_fn, argnums=0), argnums=1)


@partial(jit, static_argnames=['kernel_fn'])
def explicit_hess(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, l: float) -> jnp.ndarray:
    hess_fn = explicit_hess_fn(kernel_fn)
    return hess_fn(x1, x2, l)


@partial(jit, static_argnames=['kernel_fn'])
def hvp(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx2: jnp.ndarray, **kernel_kwargs) -> jnp.ndarray:
    kernel_partial = partial(kernel_fn, **kernel_kwargs)
    def grad_kernel_vp(x1, x2, dx2):
        gradx1 = lambda x2: grad(kernel_partial)(x1, x2)  # should grad use argnums?
        return jvp(gradx1, (x2,), (dx2,))
    return vmap(grad_kernel_vp, in_axes=(None, None, 1), out_axes=1)(x1, x2, dx2)[1]


# def hvp_from_jvps(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.ndarray, **kernel_kwargs) -> jnp.ndarray:
#     kernel_partial = partial(kernel_fn, **kernel_kwargs)
#     def grad_kernel_vp(x1, x2, dx2):
#         kernel_x2 = partial(kernel_partial, x1=x1)
#         return jvp(kernel_x2, (x2, ), (dx2, ))[1]
#     partial_grad_kernel_vp = partial(grad_kernel_vp, x2=x2, dx2=dx2)
#     dx1_hess = vjp(partial_grad_kernel_vp, x1, x2, dx2)(dx1)
#     return dx1_hess


def get_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, **kernel_kwargs) -> jnp.ndarray:
    # return dx1.T @ hvp(kernel_fn, x1, x2, dx2, **kernel_kwargs)
    return bilinear_hess(kernel_fn, x1, x2, dx1, dx2, **kernel_kwargs)


def _get_full_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, **kernel_kwargs) -> jnp.ndarray:
    # kernel_kwargs['kernel_fn'] = kernel_fn
    K_partial = partial(get_K, **kernel_kwargs)
    func = vmap(
        vmap(K_partial, in_axes=(None, None, 0, None, 0), out_axes=0),
        in_axes=(None, 0, None, 0, None),
        out_axes=0
    )
    # return func(x1=x1, x2=x2, dx1=dx1, dx2=dx2)
    return func(kernel_fn, x1, x2, dx1, dx2)


def _get_full_K_pmap(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, **kernel_kwargs) -> jnp.ndarray:
    # kernel_kwargs['kernel_fn'] = kernel_fn
    K_partial = partial(get_K, **kernel_kwargs)
    func = pmap(
        vmap(K_partial, in_axes=(None, None, 0, None, 0), out_axes=0),
        in_axes=(None, 0, None, 0, None),
        out_axes=0
    )
    # return func(x1=x1, x2=x2, dx1=dx1, dx2=dx2)
    return func(kernel_fn, x1, x2, dx1, dx2)


def _get_full_K_iterative(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, **kernel_kwargs) -> jnp.ndarray:
    """
    Getting the full kernel matrix can be brutally memory intensive. Iterate over x1/dx1 and build the matrix row-by-row
    """
    K_partial = partial(get_K, **kernel_kwargs)
    vec_over_x2 = vmap(K_partial, in_axes=(None, None, 0, None, 0), out_axes=0)
    def calc_row(i, val):
        val = val.at[i].set(vec_over_x2(kernel_fn, x1[i], x2, dx1[i], dx2))
        return val
    init_val = jnp.zeros((len(x1), len(x2), dx1.shape[-1], dx2.shape[-1]))
    return jax.lax.fori_loop(0, len(x1), calc_row, init_val)


def get_diag_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, **kernel_kwargs) -> jnp.ndarray:
    """
    Calculates the diagonal of the Hessian matrix for batched x1 and x2
    NOTE: this function works by getting the diagonal of `get_K`, but it's more efficient to have `get_K` get the diagonal directly
    so it computes a single vector rather than take a diagonal of the cov matrix.
    """
    K_partial = partial(get_K, **kernel_kwargs)
    diag_fn = lambda k,a1,a2,da1,da2: jnp.diagonal(K_partial(k, a1, a2, da1, da2))
    func = vmap(diag_fn, in_axes=(None, 0, 0, 0, 0), out_axes=0)
    return func(kernel_fn, x1, x2, dx1, dx2).flatten()


def get_full_K(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, **kernel_kwargs) -> jnp.ndarray:
    K = _get_full_K(kernel_fn, x1, x2, dx1, dx2, **kernel_kwargs)
    m1, m2, d1, d2 = K.shape
    return flatten(K, m1, d1, m2, d2)


def get_full_K_iterative(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx1: jnp.ndarray, dx2: jnp.array, **kernel_kwargs) -> jnp.ndarray:
    K = _get_full_K_iterative(kernel_fn, x1, x2, dx1, dx2, **kernel_kwargs)
    m1, m2, d1, d2 = K.shape
    return flatten(K, m1, d1, m2, d2)


@partial(jit, static_argnames=['kernel_fn'])
def reordered_hvp(kernel_fn: Callable, x1: jnp.ndarray, x2: jnp.ndarray, dx2: jnp.array, **kernel_kwargs):
    # grad, multiply, then grad again
    kernel_partial = partial(kernel_fn, **kernel_kwargs)
    def grad_kernel_vp(x1, x2, dx2):
        gradx1 = lambda x2: grad(kernel_partial)(x1, x2) @ dx2  # should grad use argnums?
        return grad(gradx1)(x2)
    return vmap(grad_kernel_vp, in_axes=(None, None, 1), out_axes=1)(x1, x2, dx2)


def get_hess_K_from_coords(descriptor_fn, kernel_fn, coords1, coords2, **kernel_kwargs):
    x1, dx1 = descriptor_fn(coords1)
    x2, dx2 = descriptor_fn(coords2)
    return get_full_K(kernel_fn, x1, x2, dx1, dx2, **kernel_kwargs)


if __name__ == "__main__":
    n = 512
    key = jax.random.PRNGKey(42)
    a = jax.random.normal(key, shape=(n,))
    res = explicit_hess(rbf, a, a, 1.0)
    res.block_until_ready()
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
    vres.block_until_ready()
    print(vres.shape)
    print(jnp.allclose(vres[0, 0], res))
    
    # how does the hvp work?
    new_key, subkey = jax.random.split(key)
    a = a[0]
    da = jax.random.normal(new_key, shape=(n, 8))  # pretend this is the jacobian of `a`
    hvp_res = hvp(rbf, a, a, da, l=1.0)
    hvp_res.block_until_ready()
    print(hvp_res.shape)
    print(jnp.allclose(res @ da, hvp_res))

    reordered_res = reordered_hvp(rbf, a, a, da, l=1.0)
    reordered_res.block_until_ready()
    print(reordered_res.shape)
    print(jnp.allclose(hvp_res, reordered_res))

    # hessian matrix from jvp and vjp
    hess_from_jvps = hvp_from_jvps(rbf, a, a, da, da, l=1.0)
    print(jnp.allclose(res, hess_from_jvps))


    # jax.profiler.save_device_memory_profile('memory.prof')

    # performance comparison, done with the %timeit magic in ipython
    # %timeit (explicit_hess(rbf, a, a, 1.0) @ da).block_until_ready()
    # > 1.24 ms ± 50.7 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    # %timeit hvp(rbf, a, a, da, l=1.0)[1].block_until_ready()
    # > 1.27 ms ± 67.9 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    # no performance improvement... but hopefully there's some kind of memory usage decrease? 


