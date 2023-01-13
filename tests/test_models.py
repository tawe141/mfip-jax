from models.exact import gp_predict
import pytest
from data.md17 import get_molecules
from descriptors.inv_dist import inv_dist
from kernels.hess import rbf
import jax.numpy as jnp
from jax import vmap, disable_jit


@pytest.fixture
def benzene():
    atoms, E, F, z = get_molecules('raw/benzene2017_dft.npz', n=10)
    pos = jnp.stack([a.get_positions() for a in atoms], axis=0)
    x = vmap(inv_dist, in_axes=0, out_axes=0)(pos)
    return x, F


def test_exact_predictions(benzene):
    desc, train_y = benzene
    train_x, train_dx = desc
    train_dx = train_dx.reshape(len(train_dx), -1)
    train_y = train_y.reshape(len(train_y), -1)
    with disable_jit():
        mu, var = gp_predict(train_x, train_dx, train_x, train_dx, train_y, rbf, l=1.0)
        assert jnp.allclose(train_y, mu)
