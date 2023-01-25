import jax.numpy as jnp


def safe_cholesky(A: jnp.ndarray):
    # Computes Cholesky decomp., and checks for any nans
    L = jnp.linalg.cholesky(A)
    assert not jnp.any(jnp.isnan(L)), "Found NaNs in Cholesky decomposition. Is this matrix positive definite?"
    return L
