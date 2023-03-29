from .sparse import variational_posterior
from kernels.multifidelity import perdikaris_kernel
from typing import List
import jax
import jax.numpy as jnp


def get_kernel_matrices(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, test_x, test_dx, **kernel_kwargs):
    """
    Obtains the needed kernel matrices for variational posterior evaluation for forces
    """
    K_mm = get_full_K(kernel_fn, inducing_x, inducing_x, inducing_dx, inducing_dx, **kernel_kwargs)
    K_mn = get_full_K_iterative(kernel_fn, inducing_x, train_x, inducing_dx, train_dx, **kernel_kwargs)
    K_test_m = get_full_K_iterative(kernel_fn, inducing_x, test_x, inducing_dx, test_dx, **kernel_kwargs).T
    K_test_diag = get_diag_K(kernel_fn, test_x, test_x, test_dx, test_dx, **kernel_kwargs)

    return K_mm, K_mn, K_test_m, K_test_diag


def get_kernel_matrices_energy_force(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, test_x, test_dx, **kernel_kwargs):
    K_mm, K_mn, K_test_m_F, K_test_diag_F = get_kernel_matrices(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, test_x, test_dx, **kernel_kwargs)
    K_test_m_E = get_jac_K(kernel_fn, test_x, inducing_x, inducing_dx, **kernel_kwargs)
    kernel_fn_diag = vmap(partial(kernel_fn, **kernel_kwargs), in_axes=(0, 0))
    K_test_diag_E = kernel_fn_diag(test_x, test_x)
    return K_mm, K_mn, K_test_m_E, K_test_m_F, K_test_diag_E, K_test_diag_F


def mf_step_E(kernel_fn, train_x, train_dx, test_x, test_dx, inducing_x, inducing_dx, E_mu_x1, E_var_x1, E_mu_x2, E_var_x2, rng_key: jax.random.PRNGKey, **kernel_kwargs):
	"""
    Samples the energies from the last fidelity and uses it as part of the kernel a la Perdikaris et al. 2017
    """
    #sample from the posterior of the last step
    new_key, subkey = jax.random.split(rng_key)
    E_sample_x1 = jax.random.normal(subkey, shape=E_mu_x1.shape) * jnp.sqrt(E_var_x1) + E_mu_x1
    new_key, subkey = jax.random.split(rng_key)
    E_sample_x2 = jax.random.normal(subkey, shape=E_mu_x2.shape) * jnp.sqrt(E_var_x2) + E_mu_x2

    
	#construct covariance matrix from previous fidelity's cov matrix a la Perdikaris et al. 2017
	#evaluate variational posterior using newly constructed covariance matrix
    


# def mf():
