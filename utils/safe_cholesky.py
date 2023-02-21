import jax.numpy as jnp
import jax


def safe_cholesky(A: jnp.ndarray):
    # Computes Cholesky decomp., and checks for any nans
    L = jnp.linalg.cholesky(A)
    with jax.disable_jit():
        assert not jnp.any(jnp.isnan(L)), "Found NaNs in Cholesky decomposition. Is this matrix positive definite?"
    return L
