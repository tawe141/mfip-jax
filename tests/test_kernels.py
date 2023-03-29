from kernels.hess import rbf, matern52, explicit_hess, _get_full_K, _get_full_K_iterative, get_K, hvp, bilinear_hess, get_diag_K, get_full_K, jac_K, get_jac_K
import kernels.multifidelity as mf
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


def test_matern52(random_vec, random_batch):
    cov = matern52(random_vec, random_vec, 1.0)
    assert jnp.allclose(cov, jax.nn.softplus(1.0))

    vmap_matern = vmap(vmap(matern52, in_axes=(0, None, None), out_axes=0), in_axes=(None, 0, None), out_axes=1)
    cov = vmap_matern(random_batch, random_batch, 1.0) + 1e-4 * jnp.eye(len(random_batch))
    assert jnp.all(jnp.linalg.eigvalsh(cov) > 0.0)


def test_explicit_hess(random_vec):
    H = explicit_hess(rbf, random_vec, random_vec, 1.0)
    assert jnp.allclose(H, 1.1596512 * jnp.eye(len(random_vec)))


def test_explicit_hess_batch(random_batch):
    # explicit_hess_partial = partial(explicit_hess, l=1.0)
    func = vmap(
        vmap(explicit_hess, in_axes=(None, None, 0, None), out_axes=0),
        in_axes=(None, 0, None, None),
        out_axes=0
    )
    H = func(rbf, random_batch, random_batch, 1.0)
    n, d = random_batch.shape
    assert jnp.allclose(H[0, 0], 1.1596512 * jnp.eye(d))
    assert jnp.all(jnp.linalg.eigvalsh(H.transpose(0,2,1,3).reshape(n*d, n*d)) > 0.0)


def test_hvp(random_vec):
    key = jax.random.PRNGKey(41)
    dx2 = jax.random.normal(key, shape=(16, 8))
    hess_vec_product = hvp(rbf, random_vec, random_vec, dx2, l=1.0)
    assert jnp.allclose(hess_vec_product, 1.1596512 * jnp.eye(16) @ dx2)

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
    assert jnp.allclose(res, da.T @ (1.1596512*jnp.eye(16)) @ da)

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


def test_jac_K(random_vec):
    a = random_vec
    key = jax.random.PRNGKey(11)
    da = jax.random.normal(key, shape=(16, 8))

    j = jac_K(rbf, a, a, da, l=1.0)
    assert jnp.allclose(j, jnp.zeros(8))


def test_jac_K_batch(random_batch):
    a = random_batch
    key = jax.random.PRNGKey(11)
    da = jax.random.normal(key, shape=(4, 16, 8))

    JK = get_jac_K(rbf, a, a, da, l=1.0)
    assert JK.shape == (4, 4*8)


def test_perdikaris(random_vec):
    a = random_vec
    key = jax.random.PRNGKey(11)
    da = jax.random.normal(key, shape=(16, 8))
    k = mf.perdikaris_kernel(rbf, a, a, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert jnp.allclose(k, 2.0)
    
    k_fn = partial(mf.perdikaris_kernel, rbf, lp=1.0, lf=1.0, ld=1.0)
     
    K_hess = get_K(k_fn, a, a, da, da, f_x1=1.0, f_x2=1.0)
    assert K_hess.shape == (8, 8)
    assert jnp.all(jnp.linalg.eigvalsh(K_hess + 1e-8 * jnp.eye(8, 8)) >= 0.0)


def test_multifidelity_get_K_batch(random_batch):
    a = random_batch
    key = jax.random.PRNGKey(11)
    da = jax.random.normal(key, shape=(4, 16, 8))
    new_key, subkey = jax.random.split(key)
    E = jax.random.uniform(subkey, shape=(4,))

    # main K matrix function
    K = mf._get_full_K(rbf, a, a, da, da, E, E, lp=1.0, lf=1.0, ld=1.0)
    assert K.shape == (4, 4, 8, 8)

    # main K matrix function, with a reshape op
    K = mf.get_full_K(rbf, a, a, da, da, E, E, lp=1.0, lf=1.0, ld=1.0)
    assert K.shape == (32, 32)
    assert jnp.all(jnp.linalg.eigvalsh(K + 1e-8 * jnp.eye(32, 32)) > 0.0)

    # test diag
    Kd = mf.get_diag_K(rbf, a, a, da, da, E, E, lp=1.0, lf=1.0, ld=1.0)
    assert Kd.shape == (32, )

    

