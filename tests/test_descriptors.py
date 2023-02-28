import pytest
from descriptors.inv_dist import inv_dist, _inv_dist, get_descriptors
import jax.numpy as jnp
from jax import vmap
from data.md17 import get_molecules


@pytest.fixture
def benzene_coords():
    atoms, E, F, z = get_molecules('raw/benzene2017_dft.npz', n=10)
    pos = jnp.stack([a.get_positions() for a in atoms], axis=0)
    return pos, F


def test_get_descriptors(benzene_coords):
    pos, _ = benzene_coords
    x, dx = vmap(inv_dist)(pos)
    test_x, test_dx = get_descriptors(_inv_dist, pos, normalize=False)
    assert jnp.allclose(test_x, x)
    assert jnp.allclose(test_dx, dx)
