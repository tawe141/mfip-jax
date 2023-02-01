import jax.numpy as jnp
from jax import grad, jit
from utils.safe_cholesky import safe_cholesky
from jax.scipy.linalg import solve, solve_triangular, cho_solve
from kernels.hess import get_full_K
import tqdm
import optax


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
    c = solve_triangular(L_B, A.dot(train_y), lower=True) / sigma_y

    logdet_B = 2 * jnp.sum(jnp.log(jnp.diag(L_B)))

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


def neg_elbo_from_coords(descriptor_fn, kernel_fn, train_coords, inducing_coords, train_y, sigma_y, **kernel_kwargs):
    train_x, train_dx = descriptor_fn(train_coords)
    inducing_x, inducing_dx = descriptor_fn(inducing_coords)
    return neg_elbo(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, train_y, sigma_y, **kernel_kwargs)


def variational_posterior(descriptor_fn, kernel_fn, test_coords, train_coords, inducing_coords, train_y, sigma_y, **kernel_kwargs):
    test_x, test_dx = descriptor_fn(test_coords)
    train_x, train_dx = descriptor_fn(train_coords)
    inducing_x, inducing_dx = descriptor_fn(inducing_coords)
    K_mm = get_full_K(kernel_fn, inducing_x, inducing_x, inducing_dx, inducing_dx, **kernel_kwargs)
    K_mn = get_full_K(kernel_fn, inducing_x, train_x, inducing_dx, train_dx, **kernel_kwargs)
    K_test_m = get_full_K(kernel_fn, test_x, inducing_x, test_dx, inducing_dx, **kernel_kwargs)
    K_test = get_full_K(kernel_fn, test_x, test_x, test_dx, test_dx, **kernel_kwargs)

    jitter = 1e-8 * jnp.eye(len(K_mm))

    # TODO: write a function that takes in the pre-computed cholesky decomps as inputs instead
    # this way, this function will truly scale with O(nm^2) rather than O(m^3) as it stands now
    # not useful for training, but will be useful for predictions

    L = safe_cholesky(K_mm + jitter)

    A = solve_triangular(L, K_mn, lower=True) / sigma_y
    B = A @ A.T + jnp.eye(len(A))
    L_B = safe_cholesky(B)
    c = solve_triangular(L_B, A.dot(train_y), lower=True) / sigma_y

    mu = K_test_m @ solve_triangular(
        L.T, solve_triangular(L_B.T, c, lower=False), lower=False
    )

    L_inv_K_m_test = solve_triangular(L, K_test_m.T, lower=True)
    cov = K_test - K_test_m @ solve_triangular(L.T, (L_inv_K_m_test - cho_solve((L_B, True), L_inv_K_m_test)))


    # sig_inv = K_mm + (K_mn @ K_mn.T) / sigma_y**2 + (1e-5 * jnp.eye(len(K_mm)))
    # mu_m = K_mm @ solve(sig_inv, K_mn.dot(train_y), assume_a='pos') / sigma_y**2
    # A_m = K_mm @ solve(sig_inv, K_mm, assume_a='pos')

    # K_test = get_full_K(kernel_fn, test_x, test_x, test_dx, test_dx, **kernel_kwargs)
    # K_test_m = get_full_K(kernel_fn, test_x, inducing_x, test_dx, inducing_dx, **kernel_kwargs)

    # # jitter = 1e-8 * jnp.eye(len(K_mm))
    # mu = K_test_m @ solve(K_mm + jitter, mu_m, assume_a='pos')
    # K_mm_inv_K_m_test = solve(K_mm + jitter, K_test_m.T, assume_a='pos')
    # cov = K_test - K_test_m @ K_mm_inv_K_m_test + K_test_m @ solve(K_mm + jitter, A_m @ K_mm_inv_K_m_test, assume_a='pos')

    return mu, cov


def optimize_variational_params(
    descriptor_fn, 
    kernel_fn, 
    train_coords, 
    train_y, 
    sigma_y, 
    init_inducing_coords, 
    init_kernel_kwargs, 
    optimizer_kwargs,
    num_iterations: int = 100
    ):
    loss_fn = lambda params: neg_elbo_from_coords(descriptor_fn, kernel_fn, train_coords, train_y=train_y, sigma_y=sigma_y, **params)
    grad_loss_fn = grad(loss_fn)
    optimizer = optax.adam(**optimizer_kwargs)
    params = {**init_kernel_kwargs, 'inducing_coords': init_inducing_coords}
    opt_state = optimizer.init(params)

    with tqdm.trange(num_iterations) as pbar:   # hardcoded number of epochs
        for _ in pbar:  
            loss = loss_fn(params)
            grad_loss = grad_loss_fn(params)
            updates, opt_state = optimizer.update(grad_loss, opt_state)
            params = optax.apply_updates(params, updates)
            pbar.set_description('neg. ELBO: %f' % loss)

        return loss, params
