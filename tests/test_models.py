from models.exact import gp_predict
from models.sparse import neg_elbo
import pytest
from data.md17 import get_molecules
from descriptors.inv_dist import inv_dist
from kernels.hess import rbf
import jax.numpy as jnp
from jax import vmap, disable_jit
from sklearn.model_selection import train_test_split


from jax.config import config
config.update("jax_enable_x64", True)


@pytest.fixture
def benzene():
    atoms, E, F, z = get_molecules('raw/benzene2017_dft.npz', n=100)
    pos = jnp.stack([a.get_positions() for a in atoms], axis=0)
    x = vmap(inv_dist, in_axes=0, out_axes=0)(pos)
    return x, F


def test_exact_predictions(benzene):
    desc, train_y = benzene
    train_x, train_dx = desc
    train_dx = train_dx.reshape(len(train_dx), train_dx.shape[1], -1)
    train_y = train_y.flatten()
    with disable_jit():
        mu, var = gp_predict(train_x, train_dx, train_x, train_dx, train_y, rbf, l=1.0)
        assert jnp.allclose(train_y, mu, atol=1e-2, rtol=1e-3)
        assert jnp.all(var >= 0.0)
        assert jnp.allclose(var, 0.0)


def test_variatonal_elbo(benzene):
    desc, train_y = benzene
    train_x, train_dx = desc
    train_dx = train_dx.reshape(len(train_dx), train_dx.shape[1], -1)
    train_y = train_y.flatten()
    with disable_jit():
        ne = neg_elbo(rbf, train_x, train_dx, train_x, train_dx, train_y, 0.01, l=1.0)
        assert ne > 0.0   # not sure what tests would be appropriate here...
