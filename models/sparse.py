import jax.numpy as jnp
from utils.safe_cholesky import safe_cholesky
from jax.scipy.linalg import solve, solve_triangular
from kernels.hess import get_full_K


def neg_elbo(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, train_y, sigma_y, **kernel_kwargs):
    n = len(train_x)

    K_nn = get_full_K(kernel_fn, train_x, train_x, train_dx, train_dx, **kernel_kwargs)
    K_mn = get_full_K(kernel_fn, inducing_x, train_x, inducing_dx, train_dx, **kernel_kwargs)
    K_mm = get_full_K(kernel_fn, inducing_x, inducing_x, inducing_dx, inducing_dx, **kernel_kwargs)

    # hard-coded jitter matrix...
    jitter_mm = 1e-8 * jnp.eye(len(K_mm))
    L = safe_cholesky(K_mm + jitter_mm)

    A = solve_triangular(L, K_mn, lower=True) / sigma_y
    B = A @ A.T + jnp.eye(len(A))
    L_B = safe_cholesky(B)
    c = solve_triangular(L_B, A, lower=True) @ train_y / sigma_y   # solve_triangular(L_B, A @ y) could be more efficient

    logdet_B = 2 * jnp.trace(L_B)

    # super naive interpretation. could definitely be optimized, for example by only computing 
    # the diagonal terms of the trace of matrices
    elbo = -0.5 * n * jnp.log(2*jnp.pi)
    elbo -= 0.5 * logdet_B
    elbo -= 0.5 * n * jnp.log(sigma_y**2)
    elbo -= 0.5 / sigma_y**2 * jnp.dot(train_y, train_y)
    elbo += 0.5 * jnp.dot(c, c)
    elbo -= 0.5 / sigma_y**2 * jnp.trace(K_nn)
    elbo += 0.5 * jnp.trace(A @ A.T)

    return -elbo
