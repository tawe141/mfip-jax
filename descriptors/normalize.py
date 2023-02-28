import jax.numpy as jnp
from jax.lax import cond

def _identity(mu, std):
    return mu, std


def normalized_descriptors(descriptor_fn, pos, *args):
    descriptors = descriptor_fn(pos)
    mu, std = jnp.mean(descriptors, axis=0), jnp.std(descriptors, axis=0)
    normed_desc = (descriptors - mu.reshape(1, -1)) / std.reshape(1, -1)
    return normed_desc, (mu, std)


def normalize_descriptors_apply(descriptor_fn, pos, mu, std):
    descriptors = descriptor_fn(pos)
    return (descriptors - mu.reshape(1, -1)) / std.reshape(1, -1), (mu, std)

"""
def normalized_descriptors(descriptor_fn, coords):
    x = descriptor_fn(coords)
    mu_ = jnp.mean(x, axis=0)
    std_ = jnp.std(x, axis=0)
    normed = normalize_known_mu_std(x, mu_, std_)
    return normed, (mu_, std_)


def normalize_known_mu_std(x, mu, std):
    return (x - mu_.reshape(1, -1)) / std_.reshape(1, -1)


def normalized_descriptors_known_mu_std(descriptor_fn, coords, mu, std):
    x = descriptor_fn(coords)
    normed = normalize_known_mu_std(x, mu, std)
    return normed
"""
