import tqdm
import pdb
import jax.numpy as jnp
from jax.scipy.linalg import solve, cho_solve, solve_triangular
from jax import vmap, jit, value_and_grad
from kernels.hess import get_full_K, get_jac_K, get_diag_K
from typing import Callable
from functools import partial
import optax


def flatten(x: jnp.ndarray, m1: int, d1: int, m2: int, d2: int):
    return x.transpose(0,2,1,3).reshape(m1*d1, m2*d2)


def gp_predict_from_matrices(K_train, K_test, K_test_test, train_y):
    """
    General GP prediction function. Returns mean and variance of predictions
    """
    jitter = 1e-8 * jnp.eye(len(K_train))
    L = jnp.linalg.cholesky(K_train + jitter)
    alpha = cho_solve((L, True), train_y)
    #pdb.set_trace()
    mu = K_test.T @ alpha

    c = solve_triangular(L, K_test, lower=True)
    var = K_test_test - jnp.sum(jnp.square(c), axis=0)
    
    return mu, var


def get_force_matrices(test_x: jnp.ndarray, test_dx: jnp.ndarray, train_x: jnp.ndarray, train_dx: jnp.ndarray, train_y: jnp.ndarray, kernel_fn: Callable, **kernel_kwargs):
    """
    Returns K_train, K_test, K_test_test matrices assuming they will be used for a force calculation
    """
    K_train = get_full_K(kernel_fn, train_x, train_x, train_dx, train_dx, **kernel_kwargs)
    K_test = get_full_K(kernel_fn, test_x, train_x, test_dx, train_dx, **kernel_kwargs).T
    K_test_test = get_diag_K(kernel_fn, test_x, test_x, test_dx, test_dx, **kernel_kwargs)
    return K_train, K_test, K_test_test


def get_energy_matrices(test_x: jnp.ndarray, train_x: jnp.ndarray, train_dx: jnp.ndarray, train_y: jnp.ndarray, kernel_fn: Callable, **kernel_kwargs):	
    K_train = get_full_K(kernel_fn, train_x, train_x, train_dx, train_dx, **kernel_kwargs)
    K_test = get_jac_K(kernel_fn, test_x, train_x, train_dx, **kernel_kwargs).T
    diag_kernel_fn = partial(kernel_fn, **kernel_kwargs)
    diag_kernel = vmap(diag_kernel_fn, in_axes=(0, 0))
    K_test_test = diag_kernel(test_x, test_x)
    return K_train, K_test, K_test_test


def gp_predict(test_x: jnp.ndarray, test_dx: jnp.ndarray, train_x: jnp.ndarray, train_dx: jnp.ndarray, train_y: jnp.ndarray, kernel_fn: Callable, **kernel_kwargs):
    matrices = get_force_matrices(test_x, test_dx, train_x, train_dx, train_y, kernel_fn, **kernel_kwargs)
    return gp_predict_from_matrices(*matrices, train_y)


def gp_predict_energy(test_x: jnp.ndarray, test_dx: jnp.ndarray, train_x: jnp.ndarray, train_dx: jnp.ndarray, train_y: jnp.ndarray, kernel_fn: Callable, **kernel_kwargs):
    matrices = get_energy_matrices(test_x, train_x, train_dx, train_y, kernel_fn, **kernel_kwargs)
    mu, std = gp_predict_from_matrices(*matrices, train_y)
    return -mu, std

"""
def gp_predict(test_x: jnp.ndarray, test_dx: jnp.ndarray, train_x: jnp.ndarray, train_dx: jnp.ndarray, train_y: jnp.ndarray, kernel_fn: Callable, **kernel_kwargs):
    K_train = get_full_K(kernel_fn, train_x, train_x, train_dx, train_dx, **kernel_kwargs)
    jitter = 1e-8 * jnp.eye(len(K_train))
    L = jnp.linalg.cholesky(K_train + jitter)
    alpha = cho_solve((L, True), train_y) 
    K_test = get_full_K(kernel_fn, test_x, train_x, test_dx, train_dx, **kernel_kwargs)

    mu = K_test @ alpha

    # uncertainties

    K_test_test = get_diag_K(kernel_fn, test_x, test_x, test_dx, test_dx, **kernel_kwargs)
    #var = jnp.diag(K_test_test - K_test.T @ solve(K_train + jitter, K_test, assume_a='pos'))
    c = solve_triangular(L, K_test, lower=True)
    var = K_test_test - jnp.sum(jnp.square(c), axis=0)

    return mu, var


def gp_predict_energy(test_x: jnp.ndarray, test_dx: jnp.ndarray, train_x: jnp.ndarray, train_dx: jnp.ndarray, train_y: jnp.ndarray, kernel_fn: Callable, **kernel_kwargs):	
    K_train = get_full_K(kernel_fn, train_x, train_x, train_dx, train_dx, **kernel_kwargs)
    jitter = 1e-8 * jnp.eye(len(K_train))
    L = jnp.linalg.cholesky(K_train + jitter)
    alpha = cho_solve((L, True), train_y)
    K_test = get_jac_K(kernel_fn, test_x, train_x, train_dx, **kernel_kwargs)
	
    mu = K_test @ alpha

    
    #batch_K_fn = vmap(
    #    vmap(partial(kernel_fn, **kernel_kwargs), in_axes=(None, 0)),
    #    in_axes=(0, None)
    #)

    #K_test_test = batch_K_fn(test_x, test_x) 
    #var = jnp.diag(K_test_test - K_test @ cho_solve((L, True), K_test.T))
    
    c = solve_triangular(L, K_test.T, lower=True)
    diag_kernel_fn = partial(kernel_fn, **kernel_kwargs)
    diag_kernel = vmap(diag_kernel_fn, in_axes=(0, 0))
    var = diag_kernel(test_x, test_x) - jnp.sum(jnp.square(c), axis=0)  # TODO: WRONG IMPLEMENTATION

    return mu, var 
"""

@partial(jit, static_argnames=['kernel_fn'])
def gp_energy_force(test_x: jnp.ndarray, test_dx: jnp.ndarray, train_x: jnp.ndarray, train_dx: jnp.ndarray, train_y: jnp.ndarray, kernel_fn: Callable, **kernel_kwargs): 
    K_train = get_full_K(kernel_fn, train_x, train_x, train_dx, train_dx, **kernel_kwargs)
    jitter = 1e-8 * jnp.eye(len(K_train))
    L = jnp.linalg.cholesky(K_train + jitter)
    alpha = cho_solve((L, True), train_y)

    K_test_hess = get_full_K(kernel_fn, test_x, train_x, test_dx, train_dx, **kernel_kwargs)
    F_mu = K_test_hess @ alpha

    K_test_jac = get_jac_K(kernel_fn, test_x, train_x, train_dx, **kernel_kwargs)
    E_mu = -K_test_jac @ alpha

    c_F = solve_triangular(L, K_test_hess.T, lower=True)
    K_test_test_diag = get_diag_K(kernel_fn, test_x, test_x, test_dx, test_dx, **kernel_kwargs)
    #pdb.set_trace()
    F_var = K_test_test_diag - jnp.sum(jnp.square(c_F), axis=0)

    c_E = solve_triangular(L, K_test_jac.T, lower=True)
    diag_kernel_fn = partial(kernel_fn, **kernel_kwargs)
    diag_kernel = vmap(diag_kernel_fn, in_axes=(0, 0))
    #pdb.set_trace()
    E_var = diag_kernel(test_x, test_x) - jnp.sum(jnp.square(c_E), axis=0)  # TODO: WRONG IMPLEMENTATION

    return (E_mu, E_var), (F_mu, F_var)


def gp_correct_energy(E_predict, E_ref):
    # finds integration constant and returns energy
    #pdb.set_trace()
    c = jnp.mean(E_predict - E_ref)
    return E_predict - c


@partial(jit, static_argnames=['kernel_fn'])
def neg_mll(train_x: jnp.ndarray, train_dx: jnp.ndarray, train_y: jnp.ndarray, kernel_fn: Callable, **kernel_kwargs):
    K = get_full_K(kernel_fn, train_x, train_x, train_dx, train_dx, **kernel_kwargs)
    return neg_mll_from_K(K, train_y)


@jit
def neg_mll_from_K(K: jnp.ndarray, train_y):
    jitter = 1e-8 * jnp.eye(len(K))
    L = jnp.linalg.cholesky(K + jitter)
    return jnp.sum(jnp.log(jnp.diag(L))) + 0.5 * jnp.dot(train_y, cho_solve((L, True), train_y))


def _optimize_kernel(
    loss_fn: Callable,
    init_kernel_kwargs: dict,
    optimizer_kwargs: dict,
    num_iterations: int = 100
    ):
    loss_and_grad_fn = jit(value_and_grad(loss_fn))

    optimizer = optax.sgd(**optimizer_kwargs)

    params = init_kernel_kwargs
    opt_state = optimizer.init(params)

    @jit
    def iteration(parameters, optimizer_state):
        # loss = loss_fn(parameters)
        # grad_loss = grad_loss_fn(parameters)
        loss, grad_loss = loss_and_grad_fn(parameters)
        updates, opt_state = optimizer.update(grad_loss, optimizer_state)
        new_params = optax.apply_updates(params, updates)
        return loss, opt_state, new_params

    with tqdm.trange(num_iterations) as pbar:
        for _ in pbar:
            loss, opt_state, params = iteration(params, opt_state)
            pbar.set_description('neg. MLL: %f' % loss)

        return loss, params


def optimize_kernel(
    train_x: jnp.ndarray, 
    train_dx: jnp.ndarray, 
    train_y: jnp.ndarray, 
    kernel_fn: Callable, 
    init_kernel_kwargs: dict,
    optimizer_kwargs: dict,
    num_iterations: int = 100
    ):
    loss_fn = lambda params: neg_mll(train_x, train_dx, train_y, kernel_fn, **params)
    return _optimize_kernel(loss_fn, init_kernel_kwargs, optimizer_kwargs, num_iterations)
