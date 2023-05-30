import pdb
import jax
from jax import jit
import jax.numpy as jnp
from typing import List, Callable
import models.exact as exact
import models.exact_mf as emf
import kernels.perdikaris_mf as pmf
from functools import partial


def _build_test(i: int, test_x: jnp.ndarray, test_dx: jnp.ndarray, train_x: List[jnp.ndarray], train_dx: List[jnp.ndarray]):
    if i == len(train_x):
        return test_x, test_dx
    else:
        new_test_x = jnp.concatenate([*train_x[i:], test_x])
        new_test_dx = jnp.concatenate([*train_dx[i:], test_dx])
        return new_test_x, new_test_dx


def gp_energy_force(
    test_x: jnp.ndarray,
    test_dx: jnp.ndarray,
    train_x: List[jnp.ndarray],
    train_dx: List[jnp.ndarray],
    train_y: List[jnp.ndarray],
    kernel_fn: Callable,
    kernel_kwargs: List[dict]
    ):
    # for now, assumes two fidelities 

    num_fidelities = len(train_x)
    num_coords = test_dx.shape[-1]
    num_training_points = jnp.array([len(i) for i in train_x])
    num_test = len(test_x)
    train_idx = jnp.cumsum(num_training_points)

    E_mu = jnp.zeros((num_fidelities, len(test_x)))
    E_var = jnp.zeros((num_fidelities, len(test_x)))
    F_mu = jnp.zeros((num_fidelities, len(test_x), num_coords))
    F_var = jnp.zeros((num_fidelities, len(test_x), num_coords))

    # get new test matrices
    # start at index 1 because there's no need to predict the training set on the first fidelity
    new_test_x, new_test_dx = _build_test(1, test_x, test_dx, train_x, train_dx)

    # initialize first fidelity with an exact GP
    #pdb.set_trace()
    (E_mu_, E_var_), (F_mu_, F_var_) = exact.gp_energy_force(new_test_x, new_test_dx, train_x[0], train_dx[0], train_y[0], kernel_fn, **kernel_kwargs[0])
    
    #pdb.set_trace()
    
    E_mu = E_mu.at[0].set(E_mu_[-num_test:])
    E_var = E_var.at[0].set(E_var_[-num_test:])
    F_mu = F_mu.at[0].set(F_mu_.reshape(-1, num_coords)[-num_test:])
    F_var = F_var.at[0].set(F_var_.reshape(-1, num_coords)[-num_test:])

    for i in range(1, num_fidelities):
        n = num_training_points[i]
        E_train, F_train = E_mu_[:n], F_mu_.reshape(-1, num_coords)[:n]
        E_test, F_test = E_mu_[n:], F_mu_.reshape(-1, num_coords)[n:]

        new_test_x, new_test_dx = _build_test(i+1, test_x, test_dx, train_x, train_dx)
        #pdb.set_trace()
        (E_mu_, E_var_), (F_mu_, F_var_) = emf.gp_energy_force(
                new_test_x, 
                new_test_dx,
                E_test,
                F_test,
                train_x[i], 
                train_dx[i],
                E_train,
                F_train,
                train_y[i], 
                kernel_fn, 
                **kernel_kwargs[i]
        )

        #pdb.set_trace()
        E_mu = E_mu.at[i].set(E_mu_[-num_test:])
        E_var = E_var.at[i].set(E_var_[-num_test:])
        F_mu = F_mu.at[i].set(F_mu_.reshape(-1, num_coords)[-num_test:])
        F_var = F_var.at[i].set(F_var_.reshape(-1, num_coords)[-num_test:])

    return (E_mu, E_var), (F_mu, F_var)


@partial(jit, static_argnames=['kernel_fn'])
def neg_mll(
    train_x: jnp.ndarray, 
    train_dx: jnp.ndarray, 
    train_y: jnp.ndarray, 
    E_train: jnp.ndarray, 
    F_train: jnp.ndarray, 
    kernel_fn: Callable,
    **kernel_kwargs
    ):
    K = pmf.get_full_K(kernel_fn, train_x, train_x, train_dx, train_dx, E_train, E_train, F_train, F_train, **kernel_kwargs)
    return exact.neg_mll_from_K(K, train_y)


def total_neg_mll(
    train_x: List[jnp.ndarray], 
    train_dx: List[jnp.ndarray], 
    train_y: List[jnp.ndarray], 
    kernel_fn: Callable,
    kernel_kwargs: List[dict]
    ):
    num_fidelities = len(train_x)
    num_data = [len(i) for i in train_x]

    res = exact.neg_mll(train_x[0], train_dx[0], train_y[0], kernel_fn, **kernel_kwargs[0])
    _train_x = jnp.concatenate(train_x[1:])
    _train_dx = jnp.concatenate(train_dx[1:])
    #pdb.set_trace()
    (pred_E_mu, pred_E_var), (pred_F_mu, pred_F_var) = exact.gp_energy_force(_train_x, _train_dx, train_x[0], train_dx[0], train_y[0], kernel_fn, **kernel_kwargs[0])

    for i in range(1, num_fidelities):
        E_train = jax.lax.dynamic_slice_in_dim(pred_E_mu, 0, num_data[i])
        F_train = jax.lax.dynamic_slice_in_dim(pred_F_mu, 0, num_data[i])

        #E_train, F_train = pred_E_mu[:num_data[i]], pred_F_mu[:num_data[i]]
        res += neg_mll(train_x[i], train_dx[i], train_y[i], E_train, F_train, kernel_fn, **kernel_kwargs[i])
        if i != num_fidelities-1:
            _train_x = jnp.concatenate(train_x[i+1:])
            _train_dx = jnp.concatenate(train_dx[i+1:])
            #E_test, F_test = pred_E_mu[num_data[i+1]:], pred_F_mu[num_data[i+1]:]
            E_test = jax.lax.dynamic_slice_in_dim(pred_E_mu, num_data[i], -1)
            F_test = jax.lax.dynamic_slice_in_dim(pred_F_mu, num_data[i], -1)

            (pred_E_mu, pred_E_var), (pred_F_mu, pred_F_var) = emf.gp_energy_force(_train_x, _train_dx, E_test, F_test, train_x[i], train_dx[i], E_train, F_train, train_y[i], kernel_fn, **kernel_kwargs[i])

    return res


def optimize_kernel(
    train_x: List[jnp.ndarray], 
    train_dx: List[jnp.ndarray], 
    train_y: List[jnp.ndarray], 
    kernel_fn: Callable,
    init_kernel_kwargs: List[dict],
    optimizer_kwargs: dict,
    num_iterations: int = 100
    ):
    loss_fn = lambda params: total_neg_mll(train_x, train_dx, train_y, kernel_fn, params)
    return exact._optimize_kernel(loss_fn, init_kernel_kwargs, optimizer_kwargs, num_iterations)


def _optimize_kernel(
    train_x: jnp.ndarray, 
    train_dx: jnp.ndarray, 
    train_y: jnp.ndarray, 
    E_train: jnp.ndarray, 
    F_train: jnp.ndarray, 
    kernel_fn: Callable,
    init_kernel_kwargs: dict,
    optimizer_kwargs: dict,
    num_iterations: int = 100
    ):
    loss_fn = lambda params: neg_mll(train_x, train_dx, train_y, E_train, F_train, kernel_fn, **params)
    return exact._optimize_kernel(loss_fn, init_kernel_kwargs, optimizer_kwargs, num_iterations)

"""
def optimize_kernel(
    train_x: List[jnp.ndarray], 
    train_dx: List[jnp.ndarray], 
    train_y: List[jnp.ndarray], 
    kernel_fn: Callable,
    init_kernel_kwargs: List[dict],
    optimizer_kwargs: dict,
    num_iterations: int = 100
    ):
    num_fidelities = len(train_x)

    list_new_params = []
    all_loss = jnp.zeros((num_fidelities, ))
    # optimize the first fidelity as an exact gp
    loss, new_params = exact.optimize_kernel(train_x[0], train_dx[0], train_y[0], rbf, init_kernel_kwargs[0], optimizer_kwargs, num_iterations)
    list_new_params.append(new_params)
    all_loss = all_loss.at[0].set(loss)
    (prev_E_mu, prev_E_var), (prev_F_mu, prev_F_var) = exact.gp_energy_force(train_x[1], train_dx[1], train_x[0], train_dx[0], train_y[0], kernel_fn, **new_params)
    for f in range(1, num_fidelities):
        prev_F_mu = prev_F_mu.reshape(len(prev_E_mu), -1)
        loss, new_params = pmf._optimize_kernel(train_x[f], train_dx[f], train_y[f], pref_E_mu, prev_F_mu, kernel_fn, init_kernel_kwargs[f], optimizer_kwargs, num_iterations)
        list_new_params.append(new_params)
        all_loss = all_loss.at[f].set(loss)
        if f != num_fidelities-1:
            (prev_E_mu, prev_E_var), (prev_F_mu, prev_F_var) = emf.gp_energy_force(train_x[f+1], train_dx[f+1], prev_E_mu, prev_F_mu, 

"""
