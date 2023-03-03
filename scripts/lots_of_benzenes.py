"""
Try training a GP with many benzene configurations
"""

from data.md17 import get_molecules
from descriptors.inv_dist import inv_dist 
from models.sparse import optimize_variational_params, variational_posterior, neg_elbo_from_coords
from kernels.hess import rbf, scaled_rbf
import jax.numpy as jnp
from jax import vmap, jit
import tqdm
import jax
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import pdb
from jax.config import config
from functools import partial
config.update("jax_enable_x64", True)


@partial(jit, static_argnames=['descriptor_fn', 'kernel_fn'])
def get_variational_rmse(descriptor_fn, kernel_fn, test_pos, train_pos, inducing_pos, train_F, test_F, sigma_y, **kernel_kwargs):
    train_y = train_F.flatten()
    mu, var = variational_posterior(descriptor_fn, scaled_rbf, test_pos, train_pos, inducing_pos, train_y, sigma_y, **kernel_kwargs)
    mu = mu.reshape(test_F.shape)
    mse = jnp.mean((mu - test_F)**2, axis=(-1, -2))
    rmse_overall = jnp.sqrt(jnp.mean(mse))
    rmse_forces = jnp.sqrt(mse)
    return rmse_overall, rmse_forces


atoms, E, F, z = get_molecules('raw/benzene2017_dft.npz', n=1000)
pos = jnp.stack([a.get_positions() for a in atoms], axis=0)
train_pos, test_pos, train_F, test_F, train_ind, test_ind = train_test_split(pos, F, jnp.arange(len(pos)), test_size=0.5, shuffle=False)
train_y, test_y = train_F.flatten(), test_F.flatten()

# choose every 10 configurations to be inducing points
inducing_pos = train_pos[::10]
print('Using %i inducing points' % len(inducing_pos))
descriptor_fn = jit(vmap(inv_dist))

rmse_train_overall, rmse_train_forces = get_variational_rmse(descriptor_fn, rbf, train_pos, train_pos, inducing_pos, train_F, train_F, 0.01, l=1.0, prefactor=1.0)
rmse_test_overall, rmse_test_forces = get_variational_rmse(descriptor_fn, rbf, test_pos, train_pos, inducing_pos, train_F, test_F, 0.01, l=1.0, prefactor=1.0)

print('Train RMSE: %.3f' % rmse_train_overall)
print('Test RMSE: %.3f' % rmse_test_overall)
"""
# get initial error metrics
# with jax.disable_jit():
mu_train, var_train = variational_posterior(descriptor_fn, rbf, train_pos, train_pos, inducing_pos, train_y, 0.001, l=1.0)

# var_train = jnp.diagonal(cov_train)
mu_train = mu_train.reshape(train_F.shape)
rmse_train_forces = jnp.sqrt(jnp.mean((mu_train - train_F)**2, axis=(-1, -2)))

mu_test, var_test = variational_posterior(descriptor_fn, rbf, test_pos, train_pos, inducing_pos, train_y, 0.001, l=1.0)
# var_test = jnp.diagonal(cov_test)
rmse = mean_squared_error(test_y, mu_test, squared=False)
print(rmse)
"""

# plt.plot(train_ind, rmse_train_forces)
# plt.plot(test_ind, rmse_test_forces)
# plt.xlabel('MD step')
# plt.ylabel('RMSE forces (kcal/mol/A)')
# plt.show()

## try a bunch of lengthscale values out and see what neg. ELBO they give
#elbos = []
#lengthscales = jnp.logspace(-1, 1, num=100)
#with tqdm.tqdm(lengthscales) as pbar:
#    for l in pbar:
#        elbos.append(neg_elbo_from_coords(descriptor_fn, rbf, train_pos, inducing_pos, train_y, 0.001, l=l))
#plt.figure()
#plt.plot(lengthscales, elbos)
#plt.xlabel('kernel lengthscale')
#plt.ylabel('neg. ELBO')
#plt.xscale('log')
#plt.yscale('log')
#plt.savefig('lengthscale_elbo.svg')

"""
Scan over a grid of values for sigma and l
"""
n_l = 10
n_prefactor = 10
l_list = jnp.linspace(0.1, 5, n_l)
prefactor_list = jnp.logspace(0, 2, n_prefactor)
rmse_grid = jnp.zeros((n_l, n_prefactor))
elbo_grid = jnp.zeros((n_l, n_prefactor))

for i, l in enumerate(l_list):
    for j, prefactor in enumerate(prefactor_list):
        rmse, _ = get_variational_rmse(descriptor_fn, scaled_rbf, test_pos, train_pos, inducing_pos, train_F, test_F, 1e-4, l=l, prefactor=prefactor)
        rmse_grid = rmse_grid.at[i, j].set(rmse)
        elbo = neg_elbo_from_coords(descriptor_fn, scaled_rbf, train_pos, inducing_pos, train_y, 1e-4, l=l, prefactor=prefactor)
        elbo_grid = elbo_grid.at[i, j].set(elbo)

fig, ax = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)
sns.heatmap(rmse_grid, ax=ax[0], xticklabels=prefactor_list, yticklabels=l_list, norm=LogNorm())
sns.heatmap(elbo_grid, ax=ax[1], xticklabels=prefactor_list, yticklabels=l_list, norm=LogNorm())
ax[0].set_title('RMSE')
ax[1].set_title('ELBO')
plt.xlabel('prefactor')
plt.ylabel('$l$')
plt.savefig('rmse_elbo.svg')

# optimize inducing points and lengthscale
n_atoms = train_pos.shape[1]
#pdb.set_trace()
new_neg_elbo, new_params = optimize_variational_params(
    descriptor_fn,
    scaled_rbf,
    train_pos,
    train_y,
    {
        'l': 1.0,
        #'l': jnp.ones(n_atoms * (n_atoms - 1) // 2), 
        'sigma_y': 0.0001, 
        'inducing_coords': inducing_pos,
        'prefactor': 1.0
    },
    ['l', 'inducing_coords', 'prefactor'],
    {'learning_rate': 1e-2},
    num_iterations=100,
)
    # new_neg_elbo.block_until_ready()
#inducing_pos, l = new_params['inducing_coords'], new_params['l']
l = new_params['l']
inducing_pos = new_params['inducing_coords']

new_rmse_train_overall, new_rmse_train_forces = get_variational_rmse(descriptor_fn, rbf, train_pos, train_pos, inducing_pos, train_F, train_F, 0.01, l=l)
new_rmse_test_overall, new_rmse_test_forces = get_variational_rmse(descriptor_fn, rbf, test_pos, train_pos, inducing_pos, train_F, test_F, 0.01, l=l)

print('Done optimizing neg. ELBO')
print('Train RMSE: %.3f' % new_rmse_train_overall)
print('Test RMSE: %.3f' % new_rmse_test_overall)

plt.figure()
plt.plot(train_ind, rmse_train_forces, color='tab:blue', linestyle='--')
plt.plot(test_ind, rmse_test_forces, color='tab:orange', linestyle='--')
plt.plot(train_ind, new_rmse_train_forces, color='tab:blue')
plt.plot(test_ind, new_rmse_test_forces, color='tab:orange')
plt.legend(['init. training set', 'init. test set', 'trained training set', 'trained test set'])
plt.xlabel('MD step')
plt.ylabel('Force RMSE (kcal/mol/A)')
plt.savefig('md_rmse.svg')
plt.show()

