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
    assert jnp.allclose(H, 2.0 * jnp.eye(16))


def test_hvp(random_vec):
    key = jax.random.PRNGKey(41)
    dx2 = jax.random.normal(key, shape=(16, 8))
    hess_vec_product = hvp(rbf, random_vec, random_vec, dx2, l=1.0)
    assert jnp.allclose(hess_vec_product, 2 * jnp.eye(16) @ dx2)

