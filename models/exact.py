import jax.numpy as jnp
from jax.scipy.linalg import solve
from jax import vmap, jit
from kernels.hess import get_K
from typing import Callable
from functools import partial


# def get_alpha(train_x: jnp.ndarray, train_dx: jnp.ndarray, train_y: jnp.ndarray, kernel_fn: Callable) -> jnp.array:
#     K = vmap(
#         vmap(
#             get_K, 
#             in_axes=(None, None, 0, None, None, None),
#             out_axes=1
#         ),
#         in_axes=(None, 0, None, None, None, None), 
#         out_axes=0
#     )(kernel_fn, x1, x2, dx1, dx2)


def flatten(x: jnp.ndarray, m1: int, d1: int, m2: int, d2: int):
    return x.reshape(m1*d1, m2*d2)


def _gp_predict(test_x: jnp.ndarray, test_dx: jnp.ndarray, train_x: jnp.ndarray, train_dx: jnp.ndarray, train_y: jnp.ndarray, kernel_fn: Callable, **kernel_kwargs):
    m1, d1 = train_dx.shape
    m2, d2 = test_dx.shape
    get_K_partial = partial(get_K, **kernel_kwargs)
    K_fn = vmap(
        vmap(
            get_K_partial, 
            in_axes=(None, None, 0, None, None),
            out_axes=1
        ),
        in_axes=(None, 0, None, None, None), 
        out_axes=0
    )
    K_train = flatten(
        K_fn(kernel_fn, train_x, train_x, train_dx, train_dx, **kernel_kwargs),
        m1, d1, m1, d1
    )
    alpha = solve(K_train, train_y, assume_a='pos')
    K_test = flatten(
        K_fn(kernel_fn, test_x, train_x, test_dx, train_dx, **kernel_kwargs),
        m1, d1, m2, d2
    )

    mu = K_test @ alpha

    # uncertainties

    K_test_test = flatten(
        K_fn(kernel_fn, test_x, test_x, test_dx, test_dx, **kernel_kwargs),
        m2, d2, m2, d2
    )
    var = jnp.diag(K_test_test - K_test.T @ solve(K_train, K_test, assume_a='pos'))

    return mu, var


# @partial(jit, static_argnames=['kernel_fn'])
def gp_predict(
    test_x: jnp.ndarray, 
    test_dx: jnp.ndarray, 
    train_x: jnp.ndarray, 
    train_dx: jnp.ndarray, 
    train_y: jnp.ndarray, 
    kernel_fn: Callable, 
    **kernel_kwargs
    ):
    predict_partial = partial(_gp_predict, **kernel_kwargs)
    return vmap(predict_partial, in_axes=(0, None, None, None, None, None), out_axes=0)(test_x, test_dx, train_x, train_dx, train_y, kernel_fn)

