from .sparse import variational_posterior
from kernels.multifidelity import perdikaris_kernel
from typing import List
import jax
import jax.numpy as jnp


def mf_step_E(kernel_fn, train_x, train_dx, test_x, test_dx, inducing_x, inducing_dx, E_mu, E_var, rng_key: jax.random.PRNGKey, **kernel_kwargs):
	"""
    Samples the energies from the last fidelity and uses it as part of the kernel a la Perdikaris et al. 2017
    """
    #sample from the posterior of the last step
    new_key, subkey = jax.random.split(rng_key)
    E_sample = jax.random.normal(subkey, shape=E_mu.shape) * jnp.sqrt(E_var) + E_mu
    
	#construct covariance matrix from previous fidelity's cov matrix a la Perdikaris et al. 2017
	#evaluate variational posterior using newly constructed covariance matrix
    


# def mf():
