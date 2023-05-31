import jax
import pdb
import argparse
from data.md17 import get_molecules
from descriptors.inv_dist import inv_dist
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import models.exact as exact
import models.perdikaris_mf as pmf
from kernels.hess import rbf
from jax import vmap, grad
import jax.numpy as jnp


def get_coords(filename, n):
    atoms, E, F, z = get_molecules(filename, n, shuffle=True)
    pos = jnp.stack([a.get_positions() for a in atoms])
    return pos, E, F


def get_data(num_dft: int, num_ccsd: int):
    dft_mols = get_coords('raw/benzene2017_dft.npz', n=num_dft)
    cc_mols = get_coords('raw/benzene_ccsd_t-train.npz', n=num_ccsd)
    return dft_mols, cc_mols


def setup_data(num_dft, num_ccsd):
    dft_mols, cc_mols = get_data(num_dft, num_ccsd)
    desc_fn = vmap(inv_dist)
    dft_x, dft_dx = desc_fn(dft_mols[0])
    cc_x, cc_dx = desc_fn(cc_mols[0])
    
    dft_y = dft_mols[-1].reshape(-1, 36)
    cc_y = cc_mols[-1].reshape(-1, 36)

    train_cc_x, test_cc_x, train_cc_dx, test_cc_dx, train_cc_y, test_cc_y = train_test_split(cc_x, cc_dx, cc_y, train_size=0.2, shuffle=False)

    return (train_cc_x, train_cc_dx, train_cc_y.flatten()), (test_cc_x, test_cc_dx, test_cc_y.flatten()), (dft_x, dft_dx, dft_y.flatten())


def make_predictions(num_dft, num_ccsd):
    (train_cc_x, train_cc_dx, train_cc_y), (test_cc_x, test_cc_dx, test_cc_y), (dft_x, dft_dx, dft_y) = setup_data(num_dft, num_ccsd)
        
    init_params = [{'l': -3.0}, {'lp': 1.0, 'lf': 3.0, 'ld': -3.0, 'w': -15.0}]
    (E_mu, E_var), (F_mu, F_var) = exact.gp_energy_force(test_cc_x, test_cc_dx, train_cc_x, train_cc_dx, train_cc_y, rbf, l=init_params[1]['ld'])
    F_rmse = mean_squared_error(test_cc_y, F_mu, squared=False)
    print('RMSE on just CC data:')
    print(F_rmse)

    with jax.profiler.start_trace('tmp/benchmarking'):

        (E_mu, E_var), (F_mu, F_var) = pmf.gp_energy_force(
                test_cc_x,
                test_cc_dx,
                [dft_x, train_cc_x],
                [dft_dx, train_cc_dx],
                [dft_y, train_cc_y],
                rbf,
                init_params,
        )
        F_mu.block_until_ready()

    F_rmse = mean_squared_error(test_cc_y, F_mu[-1].flatten(), squared=False)
    print('RMSE on unoptimized MF GP using PBE and CC data: ')
    print(F_rmse)

    init_mll = pmf.total_neg_mll(
            [dft_x, train_cc_x],
            [dft_dx, train_cc_dx],
            [dft_y, train_cc_y],
            rbf,
            init_params,
    )
    print('Initial MLL: %f' % init_mll)

    grad_mll = grad(lambda params: pmf.total_neg_mll(
            [dft_x, train_cc_x],
            [dft_dx, train_cc_dx],
            [dft_y, train_cc_y],
            rbf,
            params)
    )(init_params)
    print('Grad neg. MLL: ')
    print(grad_mll)

    """
    print('Optimizing all kernel hyperparameters...')
    _, new_params = pmf.optimize_kernel(
        [dft_x, train_cc_x],
        [dft_dx, train_cc_dx],
        [dft_y, train_cc_y],
        rbf,
        init_params,
        {'learning_rate': 1e-3}
    )
    print('Found hyperparameters')
    print(new_params)

    (E_mu, E_var), (F_mu, F_var) = pmf.gp_energy_force(
            test_cc_x,
            test_cc_dx,
            [dft_x, train_cc_x],
            [dft_dx, train_cc_dx],
            [dft_y, train_cc_y],
            rbf,
            new_params,
    )
    F_rmse = mean_squared_error(test_cc_y, F_mu[-1].flatten(), squared=False)
    print('RMSE on MF GP after optimizing')
    print(F_rmse)
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog='Benzene Multi-Fidelity Example',
            description='Run a sample multifidelity force prediction and print the error wrt the last fidelity')
    parser.add_argument('num_dft', type=int)
    parser.add_argument('num_ccsd', type=int)

    args = parser.parse_args()

    make_predictions(args.num_dft, args.num_ccsd)
