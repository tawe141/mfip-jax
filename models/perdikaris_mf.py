import pdb
import jax.numpy as jnp
from typing import List, Callable
import models.exact as exact
import models.exact_mf as emf


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
