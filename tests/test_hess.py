from kernels.hess import *
import jax 
import jax.numpy as jnp
import pytest


@pytest.fixture
def random_vec():
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, shape=(16,))


@pytest.fixture
def random_batch():
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, shape=(4, 16))


def test_explicit_hess(random_vec):
    H = explicit_hess(rbf, random_vec, random_vec, 1.0)
    assert jnp.allclose(H, 2.0 * jnp.eye(len(random_vec)))


def test_explicit_hess_batch(random_batch):
    # explicit_hess_partial = partial(explicit_hess, l=1.0)
    func = vmap(
        vmap(explicit_hess, in_axes=(None, None, 0, None), out_axes=0),
        in_axes=(None, 0, None, None),
        out_axes=0
    )
    H = func(rbf, random_batch, random_batch, 1.0)
    n, d = random_batch.shape
    assert jnp.allclose(H[0, 0], 2.0 * jnp.eye(d))
    assert jnp.all(jnp.linalg.eigvalsh(H.transpose(0,2,1,3).reshape(n*d, n*d)) > 0.0)


def test_hvp(random_vec):
    key = jax.random.PRNGKey(41)
    dx2 = jax.random.normal(key, shape=(16, 8))
    hess_vec_product = hvp(rbf, random_vec, random_vec, dx2, l=1.0)
    assert jnp.allclose(hess_vec_product, 2 * jnp.eye(16) @ dx2)

    K_ij = get_K(rbf, random_vec, random_vec, dx2, dx2, l=1.0)
    assert K_ij.shape == (8, 8)


def test_batched_K(random_batch):
    key = jax.random.PRNGKey(41)
    dx = jax.random.normal(key, shape=(len(random_batch), 16, 8))
    get_K_partial = partial(get_K, l=1.0)
    batch_over_x2 = vmap(get_K_partial, in_axes=(None, None, 0, None, 0), out_axes=0)
    res = batch_over_x2(rbf, random_batch[0], random_batch, dx[0], dx)
    assert res.shape == (4, 8, 8)

    batch_over_x1x2 = vmap(batch_over_x2, in_axes=(None, 0, None, 0, None), out_axes=0)
    res = batch_over_x1x2(rbf, random_batch, random_batch, dx, dx)
    assert res.shape == (4, 4, 8, 8)

    assert jnp.allclose(get_full_K(rbf, random_batch, random_batch, dx, dx, l=1.0), res)


# def test_batch_hvp(random_batch):
#     key = jax.random.PRNGKey(41)
#     dx2 = jax.random.normal(key, shape=(len(random_batch), 16, 8))
#     with jax.disable_jit():
#         hess_vec_product = hvp(rbf, random_batch, random_batch, dx2, l=1.0)
