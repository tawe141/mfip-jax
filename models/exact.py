import jax.numpy as jnp
from jax.scipy.linalg import solve, cho_solve
from jax import vmap, jit
from kernels.hess import get_full_K, get_jac_K
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
    return x.transpose(0,2,1,3).reshape(m1*d1, m2*d2)


def gp_predict(test_x: jnp.ndarray, test_dx: jnp.ndarray, train_x: jnp.ndarray, train_dx: jnp.ndarray, train_y: jnp.ndarray, kernel_fn: Callable, **kernel_kwargs):
    # m1, d1, f1 = train_dx.shape
    # m2, d2, f2 = test_dx.shape
    K_train = get_full_K(kernel_fn, train_x, train_x, train_dx, train_dx, **kernel_kwargs)
    jitter = 1e-8 * jnp.eye(len(K_train))
    alpha = solve(K_train + jitter, train_y, assume_a='pos')
    K_test = get_full_K(kernel_fn, test_x, train_x, test_dx, train_dx, **kernel_kwargs)

    mu = K_test @ alpha

    # uncertainties

    K_test_test = get_full_K(kernel_fn, test_x, test_x, test_dx, test_dx, **kernel_kwargs)
    var = jnp.diag(K_test_test - K_test.T @ solve(K_train + jitter, K_test, assume_a='pos'))

    return mu, var


def gp_predict_energy(test_x: jnp.ndarray, test_dx: jnp.ndarray, train_x: jnp.ndarray, train_dx: jnp.ndarray, train_y: jnp.ndarray, kernel_fn: Callable, **kernel_kwargs):	
    K_train = get_full_K(kernel_fn, train_x, train_x, train_dx, train_dx, **kernel_kwargs)
    jitter = 1e-8 * jnp.eye(len(K_train))
    alpha = solve(K_train + jitter, train_y, assume_a='pos')
    K_test = get_jac_K(kernel_fn, test_x, train_x, train_dx, **kernel_kwargs)
	
    L = jnp.linalg.cholesky(K_train + jitter)
    mu = K_test @ cho_solve((L, True), train_y)

    batch_K_fn = vmap(
        vmap(partial(kernel_fn, **kernel_kwargs), in_axes=(None, 0)),
        in_axes=(0, None)
    )

    K_test_test = batch_K_fn(test_x, test_x) 
    var = jnp.diag(K_test_test - K_test @ cho_solve((L, True), K_test.T))

    return mu, var 


def gp_correct_energy(E_predict, E_ref):
    # finds integration constant and returns energy
    c = jnp.mean(E_predict + E_ref)
    return c - E_predict
