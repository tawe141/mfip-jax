"""
Try training a GP with many benzene configurations
"""

from data.md17 import get_molecules
from descriptors.inv_dist import inv_dist
from models.sparse import optimize_variational_params, variational_posterior
from kernels.hess import rbf
import jax.numpy as jnp
from jax import vmap
import jax
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from jax.config import config
config.update("jax_enable_x64", True)




atoms, E, F, z = get_molecules('raw/benzene2017_dft.npz', n=1000)
pos = jnp.stack([a.get_positions() for a in atoms], axis=0)
train_pos, test_pos, train_F, test_F, train_ind, test_ind = train_test_split(pos, F, jnp.arange(len(pos)), test_size=0.5, shuffle=False)
train_y, test_y = train_F.flatten(), test_F.flatten()

# choose every 10 configurations to be inducing points
inducing_pos = train_pos[::10]
print('Using %i inducing points' % len(inducing_pos))

# get initial error metrics
# with jax.disable_jit():
mu_train, cov_train = variational_posterior(vmap(inv_dist), rbf, train_pos, train_pos, inducing_pos, train_y, 0.001, l=1.0)
mu_train.block_until_ready()
cov_train.block_until_ready()
jax.profiler.save_device_memory_profile('memory.prof')

# var_train = jnp.diagonal(cov_train)
# mu_train = mu_train.reshape(train_F.shape)
# rmse_train_forces = jnp.sqrt(jnp.mean((mu_train - train_F)**2, axis=(-1, -2)))

# mu_test, cov_test = variational_posterior(vmap(inv_dist), rbf, test_pos, train_pos, inducing_pos, train_y, 0.001, l=1.0)
# var_test = jnp.diagonal(cov_test)
# rmse = mean_squared_error(test_y, mu_test, squared=False)
# print(rmse)


# mu_test = mu_test.reshape(test_F.shape)
# rmse_test_forces = jnp.sqrt(jnp.mean((test_F - mu_test)**2, axis=(-1, -2)))
# plt.plot(train_ind, rmse_train_forces)
# plt.plot(test_ind, rmse_test_forces)
# plt.xlabel('MD step')
# plt.ylabel('RMSE forces (kcal/mol/A)')
# plt.show()

# # optimize inducing points and lengthscale
# new_neg_elbo, new_params = optimize_variational_params(
#     vmap(inv_dist),
#     rbf,
#     pos,
#     train_y,
#     0.01,
#     inducing_pos,
#     {'l': 1.0},
#     {'learning_rate': 0.001},
#     num_iterations=100
# )
