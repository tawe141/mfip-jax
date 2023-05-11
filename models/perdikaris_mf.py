from .sparse import variational_posterior, vposterior_from_matrices_energy_forces 
from kernels.multifidelity import perdikaris_kernel, get_full_K, get_diag_K, get_jac_K
from typing import List
import jax
import jax.numpy as jnp
from jax import vmap, jit
from functools import partial
import pdb


def _normal_sample(mu, var, rng_key):
    new_key, subkey = jax.random.split(rng_key)
    return mu + jax.random.normal(subkey, mu.shape) * jnp.sqrt(var), new_key


def get_kernel_matrices(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, test_x, test_dx, E_train, E_inducing, E_test, **kernel_kwargs):
    """
    Obtains the needed kernel matrices for variational posterior evaluation for forces
    """
    K_mm = get_full_K(kernel_fn, inducing_x, inducing_x, inducing_dx, inducing_dx, E_inducing, E_inducing, **kernel_kwargs)
    K_mn = get_full_K(kernel_fn, inducing_x, train_x, inducing_dx, train_dx, E_inducing, E_train, iterative=True, **kernel_kwargs)
    K_test_m = get_full_K(kernel_fn, inducing_x, test_x, inducing_dx, test_dx, E_inducing, E_test, iterative=True, **kernel_kwargs).T
    K_test_diag = get_diag_K(kernel_fn, test_x, test_x, test_dx, test_dx, E_test, E_test, **kernel_kwargs)

    return K_mm, K_mn, K_test_m, K_test_diag


def get_kernel_matrices_energy_force(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, test_x, test_dx, E_train, E_inducing, E_test, **kernel_kwargs):
    K_mm, K_mn, K_test_m_F, K_test_diag_F = get_kernel_matrices(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, test_x, test_dx, E_train, E_inducing, E_test, **kernel_kwargs)
    K_test_m_E = get_jac_K(kernel_fn, test_x, inducing_x, inducing_dx, E_test, E_inducing, **kernel_kwargs)
    kernel_fn_diag = vmap(partial(perdikaris_kernel, kernel_fn, **kernel_kwargs), in_axes=(0, 0, 0, 0))
    K_test_diag_E = kernel_fn_diag(test_x, test_x, E_test, E_test)
    return K_mm, K_mn, K_test_m_E, K_test_m_F, K_test_diag_E, K_test_diag_F


def vposterior(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, test_x, test_dx, E_train, E_inducing, E_test, train_y, sigma_y, **kernel_kwargs):
    K_matrices = get_kernel_matrices_energy_force(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, test_x, test_dx, E_train, E_inducing, E_test, **kernel_kwargs)
    pdb.set_trace()
    return vposterior_from_matrices_energy_forces(*K_matrices, train_y, sigma_y)


def mf_step_E(kernel_fn, train_x, train_dx, test_x, test_dx, inducing_x, inducing_dx, E_mu_train, E_var_train, E_mu_test, E_var_test, E_mu_inducing, E_var_inducing, train_y, sigma_y, **kernel_kwargs):
    """
    Samples the energies from the last fidelity and uses it as part of the kernel a la Perdikaris et al. 2017
    """
    
    """
    #sample from the posterior of the last step
    #TODO: energy variance calculation is broken due to a rather difficult integral... going to just use the mean for now. 
    E_train, rng_key = _normal_sample(E_mu_train, E_var_train, rng_key)
    E_test, rng_key = _normal_sample(E_mu_test, E_var_test, rng_key)
    E_inducing, rng_key = _normal_sample(E_mu_inducing, E_var_inducing, rng_key)
    """
    E_train = E_mu_train
    E_test = E_mu_test
    E_inducing = E_mu_inducing

    fn = partial(vposterior, kernel_fn=kernel_fn, train_x=train_x, train_dx=train_dx, inducing_x=inducing_x, inducing_dx=inducing_dx, E_train=E_train, E_inducing=E_inducing, E_test=E_test, train_y=train_y, sigma_y=sigma_y, **kernel_kwargs)

    #TODO: this is a pretty terrible way of evaluating all the necessary energies and can be made much more efficient
    #for instance, this will compute the same Cholesky decomps multiple times
    #also disgustingly written :(

    E_mu_train, E_var_train, F_mu_train, F_var_train = vposterior(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, train_x, train_dx, E_train, E_inducing, E_train, train_y, sigma_y, **kernel_kwargs)  # should these use train_y as forces...?
    E_mu_test, E_var_test, F_mu_test, F_var_test = vposterior(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, test_x, test_dx, E_train, E_inducing, E_test, train_y, sigma_y, **kernel_kwargs)
    #pdb.set_trace()
    E_mu_inducing, E_var_inducing, F_mu_inducing, F_var_inducing = vposterior(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, inducing_x, inducing_dx, E_train, E_inducing, E_inducing, train_y, sigma_y, **kernel_kwargs)

    return (
        (E_mu_train, E_var_train, E_mu_test, E_var_test, E_mu_inducing, E_var_inducing),
        (F_mu_train, F_var_train, F_mu_test, F_var_test, F_mu_inducing, F_var_inducing),
    )

# def mf():
