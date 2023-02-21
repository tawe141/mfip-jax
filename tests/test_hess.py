from kernels.hess import rbf, explicit_hess, _get_full_K, _get_full_K_iterative, get_K, hvp, bilinear_hess, get_diag_K, get_full_K
from jax import vmap, jvp, vjp
from functools import partial
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

    assert jnp.allclose(_get_full_K(rbf, random_batch, random_batch, dx, dx, l=1.0), res)


def test_diag_K(random_vec):
    # single vector in a batch
    a = jnp.expand_dims(random_vec, 0)
    key = jax.random.PRNGKey(41)
    da = jax.random.normal(key, shape=(1, 16, 8))
    K_diag = get_diag_K(rbf, a, a, da, da, l=1.0)
    K = get_full_K(rbf, a, a, da, da, l=1.0)

    assert jnp.allclose(K_diag, jnp.diagonal(K))

    # multiple vectors in a batch
    key = jax.random.PRNGKey(11)
    a = jax.random.normal(key, (4, 16))
    da = jax.random.normal(key, (4, 16, 8))
    K_diag = get_diag_K(rbf, a, a, da, da, l=1.0)
    K = get_full_K(rbf, a, a, da, da, l=1.0)

    assert jnp.allclose(K_diag, jnp.diagonal(K))



def test_bilinear_hess(random_vec):
    a = random_vec
    # assume there is some map from R^16 to R^8
    key = jax.random.PRNGKey(11)
    da = jax.random.normal(key, shape=(16, 8))
    _, new_key = jax.random.split(key)
    b = jax.random.normal(new_key, shape=(16,))
    _, new_key = jax.random.split(new_key)
    db = jax.random.normal(new_key, shape=(16, 8))

    # x1=x2, dx1=dx2
    res = bilinear_hess(rbf, a, a, da, da, l=1.0)
    assert jnp.allclose(res, da.T @ (2*jnp.eye(16)) @ da)

    # x1!=x2
    res = bilinear_hess(rbf, a, b, da, db, l=1.0)
    H = explicit_hess(rbf, a, b, 1.0)
    assert jnp.allclose(res, da.T @ H @ db)


def test_iterative_K(random_batch):
    a = random_batch
    key = jax.random.PRNGKey(11)
    da = jax.random.normal(key, shape=(4, 16, 8))

    K = _get_full_K(rbf, a, a, da, da, l=1.0)
    iter_K = _get_full_K_iterative(rbf, a, a, da, da, l=1.0)

    assert jnp.allclose(K, iter_K)



# def test_batch_hvp(random_batch):
#     key = jax.random.PRNGKey(41)
#     dx2 = jax.random.normal(key, shape=(len(random_batch), 16, 8))
#     with jax.disable_jit():
#         hess_vec_product = hvp(rbf, random_batch, random_batch, dx2, l=1.0)
