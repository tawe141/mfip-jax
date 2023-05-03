from kernels.multifidelity import perdikaris_kernel, get_full_K, get_diag_K, get_jac_K
from functools import partial
import jax.numpy as jnp
from typing import Callable, List
from jax.scipy.linalg import solve, cho_solve
from jax import vmap, jit


def gp_predict(test_x: jnp.ndarray, test_dx: jnp.ndarray, E_test: jnp.ndarray, train_x: jnp.ndarray, train_dx: jnp.ndarray, E_train: jnp.ndarray, train_y: jnp.ndarray, kernel_fn: Callable, **kernel_kwargs):
    # m1, d1, f1 = train_dx.shape
    # m2, d2, f2 = test_dx.shape
    K_train = get_full_K(kernel_fn, train_x, train_x, train_dx, train_dx, E_train, E_train, **kernel_kwargs)
    jitter = 1e-8 * jnp.eye(len(K_train))
    alpha = solve(K_train + jitter, train_y, assume_a='pos')
    K_test = get_full_K(kernel_fn, test_x, train_x, test_dx, train_dx, E_test, E_train, **kernel_kwargs)

    mu = K_test @ alpha

    # uncertainties

    K_test_test = get_full_K(kernel_fn, test_x, test_x, test_dx, test_dx, E_test, E_test, **kernel_kwargs)
    var = jnp.diag(K_test_test - K_test.T @ solve(K_train + jitter, K_test, assume_a='pos'))

    return mu, var


def gp_predict_energy(test_x: jnp.ndarray, test_dx: jnp.ndarray, E_test: jnp.ndarray, train_x: jnp.ndarray, train_dx: jnp.ndarray, E_train: jnp.ndarray, train_y: jnp.ndarray, kernel_fn: Callable, **kernel_kwargs):	
    K_train = get_full_K(kernel_fn, train_x, train_x, train_dx, train_dx, E_train, E_train, **kernel_kwargs)
    jitter = 1e-8 * jnp.eye(len(K_train))
    alpha = solve(K_train + jitter, train_y, assume_a='pos')
    K_test = get_jac_K(kernel_fn, test_x, train_x, train_dx, E_test, E_train, **kernel_kwargs)
	
    L = jnp.linalg.cholesky(K_train + jitter)
    mu = K_test @ cho_solve((L, True), train_y)

    batch_K_fn = vmap(
        vmap(partial(kernel_fn, **kernel_kwargs), in_axes=(None, 0)),
        in_axes=(0, None)
    )

    K_test_test = batch_K_fn(test_x, test_x) 
    var = jnp.diag(K_test_test - K_test @ cho_solve((L, True), K_test.T))

    return mu, var 


def gp_energy_force(test_x: jnp.ndarray, test_dx: jnp.ndarray, E_test: jnp.ndarray, train_x: jnp.ndarray, train_dx: jnp.ndarray, E_train: jnp.ndarray, train_y: jnp.ndarray, kernel_fn: Callable, **kernel_kwargs): 
    K_train = get_full_K(kernel_fn, train_x, train_x, train_dx, train_dx, E_train, E_train, **kernel_kwargs)
    jitter = 1e-8 * jnp.eye(len(K_train))
    L = jnp.linalg.cholesky(K_train + jitter)
    alpha = cho_solve((L, True), train_y)

    K_test_hess = get_full_K(kernel_fn, test_x, train_x, test_dx, train_dx, E_test, E_train, **kernel_kwargs)
    F_mu = K_test_hess @ alpha

    K_test_jac = get_jac_K(kernel_fn, test_x, train_x, train_dx, E_test, E_train, **kernel_kwargs)
    E_mu = K_test_jac @ alpha

    c_F = cho_solve((L, True), K_test_hess)
    K_test_test_diag = get_diag_K(kernel_fn, test_x, test_x, test_dx, test_dx, E_test, E_test, **kernel_kwargs) 
    F_var = K_test_test_diag - jnp.sum(jnp.square(c_F), axis=-1)

    c_E = cho_solve((L, True), K_test_jac)
    mf_kernel = partial(perdikaris_kernel, kernel_fn, **kernel_kwargs)
    diag_mf_kernel = vmap(mf_kernel, in_axes=(0, 0, 0, 0))
    E_var = diag_mf_kernel(test_x, test_x, E_test, E_test) - jnp.sum(jnp.square(c_F), axis=-1)  # TODO: WRONG IMPLEMENTATION

    return (E_mu, E_var), (F_mu, F_var)


def mf_step_E(kernel_fn, train_x, train_dx, test_x, test_dx, E_mu_train, E_var_train, E_mu_test, E_var_test, train_y, **kernel_kwargs):
    E_train = E_mu_train
    E_test = E_mu_test
    E_inducing = E_mu_inducing
    
    # not a very efficient way of doing this. need a function that evaluates energy and forces at the same time. 
    E_mu, E_var = gp_predict_energy(test_x, test_dx, E_test, train_x, train_dx, E_train, train_y, kernel_fn, **kernel_kwargs)
    F_mu, F_var = gp_predict(test_x, test_dx, E_test, train_x, train_dx, E_train, train_y, kernel_fn, **kernel_kwargs)

    return (E_mu, E_var), (F_mu, F_var)


#def mf_predict(kernel_fn: Callable, train_x: List[jnp.ndarray], train_dx: List[jnp.ndarray], test_x: List[jnp.ndarray], test_dx: List[jnp.ndarray], train_y: List[jnp.ndarray], kernel_kwargs: dict):

