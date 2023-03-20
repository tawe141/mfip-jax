import jax.numpy as jnp
from jax import grad, jit, value_and_grad
from utils.safe_cholesky import safe_cholesky
from jax.scipy.linalg import solve, solve_triangular, cho_solve
from kernels.hess import get_full_K, get_diag_K, get_full_K_iterative
from functools import partial
import pdb
import tqdm
import optax
import jaxopt
# from memory_profiler import profile


# @profile
def neg_elbo(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, train_y, sigma_y, **kernel_kwargs):
    n = len(train_x)
    output_dims = train_dx.shape[-1]

    K_nn_trace = jnp.sum(get_diag_K(kernel_fn, train_x, train_x, train_dx, train_dx, **kernel_kwargs))
    K_mn = get_full_K_iterative(kernel_fn, inducing_x, train_x, inducing_dx, train_dx, **kernel_kwargs)
    K_mm = get_full_K(kernel_fn, inducing_x, inducing_x, inducing_dx, inducing_dx, **kernel_kwargs)

    # hard-coded jitter matrix...
    jitter_mm = 1e-8 * jnp.eye(len(K_mm))
    # L = safe_cholesky(K_mm + jitter_mm)
    L = jnp.linalg.cholesky(K_mm + jitter_mm)

    A = solve_triangular(L, K_mn, lower=True) / sigma_y
    B = A @ A.T + jnp.eye(len(A))
    # L_B = safe_cholesky(B)
    L_B = jnp.linalg.cholesky(B)
    c = solve_triangular(L_B, A.dot(train_y), lower=True) / sigma_y

    logdet_B = 2 * jnp.sum(jnp.log(jnp.diag(L_B)))

    # super naive interpretation. could definitely be optimized, for example by only computing 
    # the diagonal terms of the trace of matrices
    elbo = -0.5 * n * jnp.log(2*jnp.pi) * output_dims
    elbo -= 0.5 * logdet_B * output_dims
    elbo -= 0.5 * n * jnp.log(sigma_y**2) * output_dims
    elbo -= 0.5 / sigma_y**2 * jnp.sum(jnp.square(train_y))  # jnp.dot(train_y, train_y)
    elbo += 0.5 * jnp.sum(jnp.square(c))  # jnp.dot(c, c)
    elbo -= 0.5 / sigma_y**2 * output_dims * K_nn_trace  #* jnp.trace(K_nn)
    elbo += 0.5 * output_dims * jnp.sum(jnp.square(A))  #jnp.trace(A @ A.T)

    return -elbo / n


@partial(jit, static_argnames=['descriptor_fn', 'kernel_fn'])
def neg_elbo_from_coords(descriptor_fn, kernel_fn, train_coords, inducing_coords, train_y, sigma_y, **kernel_kwargs):
    train_x, train_dx = descriptor_fn(train_coords)
    inducing_x, inducing_dx = descriptor_fn(inducing_coords)
    return neg_elbo(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, train_y, sigma_y, **kernel_kwargs)


def _normalize(x, mu, std):
    return (x - mu.reshape(1, -1)) / std.reshape(1, -1)


def get_kernel_matrices(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, test_x, test_dx, **kernel_kwargs):
    """
    Obtains the needed kernel matrices for variational posterior evaluation
    """
    K_mm = get_full_K(kernel_fn, inducing_x, inducing_x, inducing_dx, inducing_dx, **kernel_kwargs)
    K_mn = get_full_K_iterative(kernel_fn, inducing_x, train_x, inducing_dx, train_dx, **kernel_kwargs)
    K_test_m = get_full_K_iterative(kernel_fn, inducing_x, test_x, inducing_dx, test_dx, **kernel_kwargs).T
    K_test_diag = get_diag_K(kernel_fn, test_x, test_x, test_dx, test_dx, **kernel_kwargs)

    return K_mm, K_mn, K_test_m, K_test_diag


def vposterior_from_matrices(K_mm, K_mn, K_test_m, K_test_diag, train_y, sigma_y):
    jitter = 1e-8 * jnp.eye(len(K_mm))
    
    L = jnp.linalg.cholesky(K_mm + jitter)

    A = solve_triangular(L, K_mn, lower=True) / sigma_y
    B = A @ A.T + jnp.eye(len(A))
    L_B = jnp.linalg.cholesky(B)
    c = solve_triangular(L_B, A.dot(train_y), lower=True) / sigma_y

    mu = K_test_m @ solve_triangular(
        L.T, solve_triangular(L_B.T, c, lower=False), lower=False
    )

    L_inv_K_m_test = solve_triangular(L, K_test_m.T, lower=True)
    LB_inv_L_inv_K_m_test = solve_triangular(L_B, L_inv_K_m_test, lower=True)
    var = K_test_diag \
        + jnp.sum(jnp.square(LB_inv_L_inv_K_m_test), axis=0) \
            - jnp.sum(jnp.square(L_inv_K_m_test), axis=0)  # square function is known to produce nans
    
    return mu, var

# @profile
# @partial(jit, static_argnames=['descriptor_fn', 'kernel_fn'])
def variational_posterior(descriptor_fn, kernel_fn, test_coords, train_coords, inducing_coords, train_y, sigma_y, **kernel_kwargs):
    train_x, train_dx = descriptor_fn(train_coords)
    test_x, test_dx = descriptor_fn(test_coords)
    inducing_x, inducing_dx = descriptor_fn(inducing_coords)
    """
    K_mm = get_full_K(kernel_fn, inducing_x, inducing_x, inducing_dx, inducing_dx, **kernel_kwargs)
    K_mn = get_full_K_iterative(kernel_fn, inducing_x, train_x, inducing_dx, train_dx, **kernel_kwargs)
    K_test_m = get_full_K_iterative(kernel_fn, inducing_x, test_x, inducing_dx, test_dx, **kernel_kwargs).T
    K_test_diag = get_diag_K(kernel_fn, test_x, test_x, test_dx, test_dx, **kernel_kwargs)
    """
    K_mm, K_mn, K_test_m, K_test_diag = get_kernel_matrices(kernel_fn, train_x, train_dx, inducing_x, inducing_dx, test_x, test_dx, **kernel_kwargs)

    """
    jitter = 1e-8 * jnp.eye(len(K_mm))

    # TODO: write a function that takes in the pre-computed cholesky decomps as inputs instead
    # this way, this function will truly scale with O(nm^2) rather than O(m^3) as it stands now
    # not useful for training, but will be useful for predictions

    # L = safe_cholesky(K_mm + jitter)
    L = jnp.linalg.cholesky(K_mm + jitter)

    A = solve_triangular(L, K_mn, lower=True) / sigma_y
    B = A @ A.T + jnp.eye(len(A))
    # L_B = safe_cholesky(B)
    L_B = jnp.linalg.cholesky(B)
    c = solve_triangular(L_B, A.dot(train_y), lower=True) / sigma_y

    mu = K_test_m @ solve_triangular(
        L.T, solve_triangular(L_B.T, c, lower=False), lower=False
    )

    L_inv_K_m_test = solve_triangular(L, K_test_m.T, lower=True)
    LB_inv_L_inv_K_m_test = solve_triangular(L_B, L_inv_K_m_test, lower=True)
    # var = K_test_diag - jnp.diagonal(
    # var = K_test_diag - jnp.diagonal(
    #     K_test_m @ solve_triangular(L.T, (L_inv_K_m_test - cho_solve((L_B, True), L_inv_K_m_test)))
    # )
    var = K_test_diag \
        + jnp.sum(jnp.square(LB_inv_L_inv_K_m_test), axis=0) \
            - jnp.sum(jnp.square(L_inv_K_m_test), axis=0)  # square function is known to produce nans


    # sig_inv = K_mm + (K_mn @ K_mn.T) / sigma_y**2 + (1e-5 * jnp.eye(len(K_mm)))
    # mu_m = K_mm @ solve(sig_inv, K_mn.dot(train_y), assume_a='pos') / sigma_y**2
    # A_m = K_mm @ solve(sig_inv, K_mm, assume_a='pos')

    # K_test = get_full_K(kernel_fn, test_x, test_x, test_dx, test_dx, **kernel_kwargs)
    # K_test_m = get_full_K(kernel_fn, test_x, inducing_x, test_dx, inducing_dx, **kernel_kwargs)

    # # jitter = 1e-8 * jnp.eye(len(K_mm))
    # mu = K_test_m @ solve(K_mm + jitter, mu_m, assume_a='pos')
    # K_mm_inv_K_m_test = solve(K_mm + jitter, K_test_m.T, assume_a='pos')
    # cov = K_test - K_test_m @ K_mm_inv_K_m_test + K_test_m @ solve(K_mm + jitter, A_m @ K_mm_inv_K_m_test, assume_a='pos')
    """
    mu, var = vposterior_from_matrices(K_mm, K_mn, K_test_m, K_test_diag, train_y, sigma_y)

    return mu, var


# @profile
# def variational_posterior(descriptor_fn, kernel_fn, test_coords, train_coords, inducing_coords, train_y, sigma_y, **kernel_kwargs):
#     test_x, test_dx = descriptor_fn(test_coords)
#     train_x, train_dx = descriptor_fn(train_coords)
#     inducing_x, inducing_dx = descriptor_fn(inducing_coords)
#     K_mm = get_full_K(kernel_fn, inducing_x, inducing_x, inducing_dx, inducing_dx, **kernel_kwargs)
#     K_mn = get_full_K(kernel_fn, inducing_x, train_x, inducing_dx, train_dx, **kernel_kwargs)
#     K_test_m = get_full_K(kernel_fn, test_x, inducing_x, test_dx, inducing_dx, **kernel_kwargs)
#     K_test = get_full_K(kernel_fn, test_x, test_x, test_dx, test_dx, **kernel_kwargs)

#     jitter = 1e-8 * jnp.eye(len(K_mm)) 
#     L = jnp.linalg.cholesky(K_mm + jitter)
#     A = solve_triangular(L, K_mn, lower=True) / sigma_y
#     B = jnp.eye(len(K_mm)) + A @ A.T
#     L_B = jnp.linalg.cholesky(B)

#     beta = solve_triangular(L_B, A @ train_y * sigma_y, lower=True)
#     beta = solve_triangular(L.T, beta, lower=False)
#     mu = K_test_m @ beta / sigma_y**2

#     C = solve_triangular(L, K_test_m.T, lower=True)
#     D = solve_triangular(L_B, C, lower=True)

#     cov = K_test + sigma_y**2 * jnp.eye(len(K_test)) - C.T @ C + D.T @ D

#     return mu, cov


#@partial(jit, static_argnames=['descriptor_fn', 'kernel_fn', 'num_iterations', 'to_optimize'])
def optimize_variational_params(
    descriptor_fn, 
    kernel_fn, 
    train_coords, 
    train_y, 
    init_params: dict,
    to_optimize: list,
    optimizer_kwargs,
    num_iterations: int = 100,
    ):
    params = {i: init_params[i] for i in to_optimize}
    static_params = {i: init_params[i] for i in init_params.keys() if i not in to_optimize}
    loss_fn = lambda params: neg_elbo_from_coords(descriptor_fn, kernel_fn, train_coords, train_y=train_y, **static_params, **params)
    
    #pdb.set_trace()
    # grad_loss_fn = grad(loss_fn)
    # jit loss and grad loss
    # loss_fn = jit(loss_fn)
    # grad_loss_fn = jit(grad_loss_fn)
    loss_and_grad_fn = jit(value_and_grad(loss_fn))
    """
    opt = jaxopt.NonlinearCG(loss_and_grad_fn, value_and_grad=True, jit=False)
    with tqdm.trange(num_iterations) as pbar:
        for i in pbar:
            if i == 0:
                state = opt.init_state(params)
            params, state = opt.update(params, state)
            pbar.set_description('neg. ELBO: %.3f; sigma_y: %.3f' % (state.value, params['l']))
    #params, state = opt.run(params)
    """
    
    optimizer = optax.adam(**optimizer_kwargs)
    #params = {**init_kernel_kwargs}
    #if optimize_inducing:
    #    params['inducing_coords'] = init_inducing_coords
    #params = init_kernel_kwargs
    opt_state = optimizer.init(params)

    print('Optimizing the following variables: ')
    print(list(params.keys()))

    @jit
    def iteration(parameters, optimizer_state):
        # loss = loss_fn(parameters)
        # grad_loss = grad_loss_fn(parameters)
        loss, grad_loss = loss_and_grad_fn(parameters)
        updates, opt_state = optimizer.update(grad_loss, optimizer_state)
        new_params = optax.apply_updates(params, updates)
        return loss, opt_state, new_params

    with tqdm.trange(num_iterations) as pbar:   # hardcoded number of epochs
        for _ in pbar:  
            # grad_loss = grad_loss_fn(params)
            # updates, opt_state = optimizer.update(grad_loss, opt_state)
            # params = optax.apply_updates(params, updates)
            # pbar.set_description('neg. ELBO: %f' % loss)
            loss, opt_state, params = iteration(params, opt_state)
            # l = params['l']
            # pbar.set_description('neg. ELBO: %f; l = %.3f' % (loss, l))
            pbar.set_description('neg. ELBO: %f' % loss)

        return loss, params
