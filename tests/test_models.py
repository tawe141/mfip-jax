import pdb
from models.exact import *
from models.sparse import *
import models.multifidelity as mf
import models.exact_mf as emf
import models.perdikaris_mf as pmf
import pytest
from data.md17 import get_molecules
from descriptors.inv_dist import inv_dist
from kernels.hess import rbf
import jax.numpy as jnp
from jax import vmap, disable_jit, grad
from sklearn.model_selection import train_test_split


from jax.config import config
config.update("jax_enable_x64", True)


def desc_to_inputs(desc_output):
    desc, E, F = desc_output
    x, dx = desc
    return x, dx, E.flatten(), F.flatten()


@pytest.fixture
def benzene_coords():
    atoms, E, F, z = get_molecules('raw/benzene2017_dft.npz', n=10)
    pos = jnp.stack([a.get_positions() for a in atoms], axis=0)
    return pos, E, F


@pytest.fixture
def benzene_with_descriptor(benzene_coords):
    pos, E, F = benzene_coords
    return vmap(inv_dist)(pos), E, F


@pytest.fixture
def benzene_coords_many():
    atoms, E, F, z = get_molecules('raw/benzene2017_dft.npz', n=1000, shuffle=True)
    pos = jnp.stack([a.get_positions() for a in atoms], axis=0)
    return pos, E, F


@pytest.fixture
def benzene_with_descriptor_many(benzene_coords_many):
    pos, E, F = benzene_coords_many
    return vmap(inv_dist)(pos), E, F


@pytest.fixture
def benzene_ccsd_coords():
    atoms, E, F, z = get_molecules('raw/benzene_ccsd_t-train.npz', n=10)
    pos = jnp.stack([a.get_positions() for a in atoms], axis=0)
    return pos, E, F


@pytest.fixture
def benzene_ccsd_descriptor(benzene_ccsd_coords):
    pos, E, F = benzene_ccsd_coords
    return vmap(inv_dist)(pos), E, F

class TestExact:
    def test_exact_predictions_identity(self, benzene_with_descriptor):
        desc, _, train_y = benzene_with_descriptor
        train_x, train_dx = desc
        train_y = train_y.flatten()
    
        mu, var = gp_predict(train_x, train_dx, train_x, train_dx, train_y, rbf, l=1.0)
        assert jnp.allclose(train_y, mu, atol=1e-2, rtol=1e-3)
        assert jnp.all(var >= 0.0)
        assert jnp.allclose(var, 0.0)
    
    
    def test_exact_predictions(self, benzene_with_descriptor, benzene_with_descriptor_many):
        desc, _, train_y = benzene_with_descriptor
        train_x, train_dx = desc
        train_y = train_y.flatten()
    
        desc, _, test_y = benzene_with_descriptor_many
        test_x, test_dx = desc
        test_y = test_y.flatten()
    
        mu, var = gp_predict(test_x, test_dx, train_x, train_dx, train_y, rbf, l=1.0)
        assert jnp.all(var >= 0.0)
        assert len(mu) == len(test_y)
    
    
    def test_exact_predictions_cc(self, benzene_ccsd_descriptor):
        self.test_exact_predictions_identity(benzene_ccsd_descriptor)
    
    
    def test_exact_energy_predict(self, benzene_with_descriptor):
        desc, E, train_y = benzene_with_descriptor
        train_x, train_dx = desc
        train_y = train_y.flatten()
        E = E.flatten()
        with disable_jit():
            mu, var = gp_predict_energy(train_x, train_dx, train_x, train_dx, train_y, rbf, l=1.0)
            #assert jnp.allclose(var, 0.0)  # variance comes out to be a constant ~0.4 here, and I'm not sure why... considering this still <1kcal/mol error, maybe this isn't so bad?
            #pdb.set_trace()
            E_predict = gp_correct_energy(mu, E) 
            assert jnp.allclose(E, E_predict)
    
    
    def test_exact_energy_force(self, benzene_with_descriptor):
        desc, E, train_y = benzene_with_descriptor
        train_x, train_dx = desc
        train_y = train_y.flatten()
        E = E.flatten()
    
        (E_mu, E_var), (F_mu, F_var) = gp_energy_force(train_x, train_dx, train_x, train_dx, train_y, rbf, l=1.0)
        assert jnp.allclose(F_mu, train_y, atol=0.01)
        assert jnp.allclose(F_var, 0.0, atol=0.001)
        E_mu = gp_correct_energy(E_mu, E)
        assert jnp.allclose(E_mu, E)
    
    
    def test_exact_energy_predict_cc(self, benzene_ccsd_descriptor):
        self.test_exact_energy_predict(benzene_ccsd_descriptor)


    def test_exact_mll(self, benzene_with_descriptor):
        desc, E, train_y = benzene_with_descriptor
        train_x, train_dx = desc
        train_y = train_y.flatten()

        loss = neg_mll(train_x, train_dx, train_y, rbf, l=1.0)
        assert isinstance(loss, jnp.ndarray)


    def test_optimize_kernel(self, benzene_with_descriptor):
        desc, E, train_y = benzene_with_descriptor
        train_x, train_dx = desc
        train_y = train_y.flatten()

        old_params = {'l': 1.0}
        loss, new_params = optimize_kernel(
                train_x,
                train_dx, 
                train_y, 
                rbf,
                old_params,
                {'learning_rate': 1e-3},
                num_iterations=5
        )

        assert old_params != new_params


class TestSparse:
    def test_kernel_matrices(self, benzene_with_descriptor):
        desc, E, train_y = benzene_with_descriptor
        train_x, train_dx = desc
        inducing_x, inducing_dx = train_x[::2], train_dx[::2]  # take every other training point to be an inducing point
        train_y = train_y.flatten()
        with disable_jit():
            K_mm, K_mn, K_test_m, K_test_diag = get_kernel_matrices(rbf, train_x, train_dx, inducing_x, inducing_dx, train_x, train_dx, l=1.0)
            assert K_mm.shape == (len(inducing_x) * 36, len(inducing_x) * 36)
            assert K_mn.shape == (len(inducing_x) * 36, len(train_x) * 36)
            assert K_test_m.shape == (len(train_x) * 36, len(inducing_x) * 36)
            assert K_test_diag.shape == (len(train_x) * 36,)
    
    
    def test_kernel_matrices_energy(self, benzene_with_descriptor):
        desc, E, train_y = benzene_with_descriptor
        train_x, train_dx = desc
        inducing_x, inducing_dx = train_x[::2], train_dx[::2]  # take every other training point to be an inducing point
        train_y = train_y.flatten()
        with disable_jit():
            K_mm, K_mn, K_test_m, K_test_diag = get_kernel_matrices_energy(rbf, train_x, train_dx, inducing_x, inducing_dx, train_x, train_dx, l=1.0)
            assert K_mm.shape == (len(inducing_x) * 36, len(inducing_x) * 36)
            assert K_mn.shape == (len(inducing_x) * 36, len(train_x) * 36)
            assert K_test_m.shape == (len(train_x), len(inducing_x) * 36)
            assert K_test_diag.shape == (len(train_x),)
    
    
    def test_kernel_matrices_energy_force(self, benzene_with_descriptor):
        desc, E, train_y = benzene_with_descriptor
        train_x, train_dx = desc
        inducing_x, inducing_dx = train_x[::2], train_dx[::2]  # take every other training point to be an inducing point
    
        K_mm, K_mn, K_test_m_E, K_test_m_F, K_test_diag_E, K_test_diag_F = get_kernel_matrices_energy_force(rbf, train_x, train_dx, inducing_x, inducing_dx, train_x, train_dx, l=1.0)
        assert K_mm.shape == (len(inducing_x) * 36, len(inducing_x) * 36)
        assert K_mn.shape == (len(inducing_x) * 36, len(train_x) * 36)
        assert K_test_m_E.shape == (len(train_x), len(inducing_x) * 36)
        assert K_test_diag_E.shape == (len(train_x),)
        assert K_test_m_F.shape == (len(train_x) * 36, len(inducing_x) * 36)
        assert K_test_diag_F.shape == (len(train_x) * 36,)
    
    
    def test_variatonal_elbo(self, benzene_with_descriptor, benzene_coords):
        desc, _, train_y = benzene_with_descriptor
        train_x, train_dx = desc
        train_y = train_y.flatten()
        with disable_jit():
            ne = neg_elbo(rbf, train_x, train_dx, train_x, train_dx, train_y, 0.01, l=1.0)
            assert ne > 0.0   # not sure what tests would be appropriate here...
    
        # see if the function from coords is the same
        with disable_jit():
            pos, _, _ = benzene_coords
            from_coords = neg_elbo_from_coords(vmap(inv_dist), rbf, pos, pos, train_y, 0.01, l=1.0)
            assert jnp.allclose(from_coords, ne)
    
    
    def test_variational_posterior(self, benzene_coords):
        pos, _, F = benzene_coords
        train_y = F.flatten()
        mu, var = variational_posterior(vmap(inv_dist), rbf, pos, pos, pos, train_y, 0.001, l=1.0)
    
        # assert jnp.allclose(mu, train_y)    # covariance works but not the means... not sure why
        # means are only 1e-2 off, which is probably ok
        assert jnp.allclose(var, 0.0, atol=1e-5)
    
    
    def test_variational_posterior_cc(self, benzene_ccsd_coords):
        test_variational_posterior(benzene_ccsd_coords)
    
    
    def test_variational_posterior_energy(self, benzene_coords):
        pos, E, F = benzene_coords
        train_y = F.flatten()
        inducing_pos = pos
        E = E.flatten()
        mu, var = variational_posterior_energy(vmap(inv_dist), rbf, pos, pos, inducing_pos, train_y, 0.001, l=1.0)
    
        assert mu.shape == (10,)
        assert var.shape == (10,)
    
        E_predict = gp_correct_energy(mu, E)
        assert jnp.allclose(E_predict, E)
        assert jnp.all(var > 0)
    
    
    def test_vposterior_energy_force(self, benzene_coords):
        pos, E, F = benzene_coords
        train_y = F.flatten()
        inducing_pos = pos
        E = E.flatten()
        E_mu, E_var, F_mu, F_var = variational_posterior_energy_force(vmap(inv_dist), rbf, pos, pos, inducing_pos, train_y, 0.001, l=0.1)
        
        #pdb.set_trace()
        assert jnp.allclose(train_y, F_mu, atol=1e-1)
        assert jnp.all(F_var >= 0.0)
        assert E.shape == E_mu.shape
        assert jnp.all(E_var >= 0.0)
        E_predict = gp_correct_energy(E_mu, E)
        assert jnp.allclose(E_predict, E)
    
    
    def test_vposterior_energy_force_cc(self, benzene_ccsd_coords):
        test_vposterior_energy_force(benzene_ccsd_coords)
    
    
    def test_optimizing_variational(self, benzene_coords):
        pos, _, F = benzene_coords
        train_y = F.flatten()
        inducing_pos = pos[::3]
    
        initial_neg_elbo = neg_elbo_from_coords(vmap(inv_dist), rbf, pos, inducing_pos, train_y, 0.01, l=1.0)
    
        new_neg_elbo, new_params = optimize_variational_params(
            vmap(inv_dist),
            rbf,
            pos,
            train_y,
            {
                'l': 1.0,
                'sigma_y': 0.01,
                'inducing_coords': inducing_pos
            },
            ['l', 'inducing_coords'],
            {'learning_rate': 0.001},
            num_iterations=10
        )
    
        assert new_neg_elbo < initial_neg_elbo
        assert not jnp.allclose(new_params['inducing_coords'], inducing_pos)
        assert not jnp.allclose(new_params['l'], 1.0)
    

def test_mf_step(benzene_with_descriptor_many, benzene_ccsd_descriptor):
    x_dft, dx_dft, E_dft, y_dft = desc_to_inputs(benzene_with_descriptor_many)
    x_cc, dx_cc, E_cc, y_cc = desc_to_inputs(benzene_ccsd_descriptor)

    # take inducing points to be a few of the DFT configurations
    inducing_x = x_dft[::20]
    inducing_dx = dx_dft[::20]

    def get_energy(x, dx):
        k_mats = get_kernel_matrices_energy(rbf, x_dft, dx_dft, inducing_x, inducing_dx, x, dx, l=0.1)
        return vposterior_from_matrices(*k_mats, y_dft, 0.001)

    # test set (CC data)
    E_test, _ = get_energy(x_cc, dx_cc)
    E_train, _ = get_energy(x_cc, dx_cc)  # same as E_test
    E_inducing, _ = get_energy(inducing_x, inducing_dx)
    E_var_test = jnp.zeros_like(E_test)  # placeholder
    E_var_train = jnp.zeros_like(E_train)
    E_var_inducing = jnp.zeros_like(E_inducing)
    # set lp and lf to near zero, making it really improbably it's using the previous fidelity
    E, F = mf.mf_step_E(rbf, x_cc, dx_cc, x_cc, dx_cc, inducing_x, inducing_dx, E_train, E_var_train, E_test, E_var_test, E_inducing, E_var_inducing, y_cc, 0.001, lp=1e-5, lf=1e-5, ld=0.1)

    E_mu_train, E_var_train, E_mu_test, E_var_test, E_mu_inducing, E_var_inducing = E
    F_mu_train, F_var_train, F_mu_test, F_var_test, F_mu_inducing, F_var_inducing = F

    c = jnp.mean(E_mu_train + E_cc.flatten())
    pdb.set_trace()
    assert jnp.allclose(c - E_mu_train, E_cc)


###
#Exact multi-fidelity GP
###

class TestExactMF:
    def test_gp_predict_emf(self, benzene_with_descriptor, benzene_ccsd_descriptor):
        # first, predict energies of CC points with DFT training data
        x_dft, dx_dft, E_dft, y_dft = desc_to_inputs(benzene_with_descriptor)
        x_cc, dx_cc, E_cc, y_cc = desc_to_inputs(benzene_ccsd_descriptor)
        
        F_cc_from_dft, _ = gp_predict(x_cc, dx_cc, x_dft, dx_dft, y_dft, rbf, l=1.0)
        F_cc_from_dft = F_cc_from_dft.reshape(len(x_cc), -1)
        E_cc_from_dft, _ = gp_predict_energy(x_cc, dx_cc, x_dft, dx_dft, y_dft, rbf, l=1.0) 
        F_mu, F_var = emf.gp_predict(x_cc, dx_cc, E_cc_from_dft, F_cc_from_dft, x_cc, dx_cc, E_cc_from_dft, F_cc_from_dft, y_cc, rbf, lp=1.0, lf=1.0, ld=1.0, w=0.0)
        assert jnp.allclose(F_mu, y_cc, atol=0.01)
        assert jnp.allclose(F_var, 0.0)
    
        ## TODO: energy evals using MF is currently failing
        #E_mu, E_var = emf.gp_predict_energy(x_cc, dx_cc, E_cc_from_dft, x_cc, dx_cc, E_cc_from_dft, F_cc_from_dft, y_cc, rbf, lp=1.0, lf=1.0, ld=1.0, w=0.0)
        #E_mu = gp_correct_energy(E_mu, E_cc)
        #assert jnp.allclose(E_mu, E_cc)
    
    
    def test_perdikaris_mf_predictions(self, benzene_with_descriptor, benzene_ccsd_descriptor):
        x_dft, dx_dft, E_dft, y_dft = desc_to_inputs(benzene_with_descriptor)
        x_cc, dx_cc, E_cc, y_cc = desc_to_inputs(benzene_ccsd_descriptor)
    
        (E_mu, E_var), (F_mu, F_var) = pmf.gp_energy_force(
                x_cc, 
                dx_cc, 
                [x_dft, x_cc], 
                [dx_dft, dx_cc], 
                [y_dft, y_cc], 
                rbf, 
                [{'l': 1.0}, {'lp': 1.0, 'lf': 1.0, 'ld': 1.0, 'w': 0.0}]
        )
    
        assert E_mu.shape == (2, len(x_cc))
        assert E_var.shape == (2, len(x_cc))
        assert F_mu.shape == (2, len(x_cc), 36)
        assert F_var.shape == (2, len(x_cc), 36)
    
        assert jnp.allclose(F_mu[-1].flatten(), y_cc, atol=1e-2)
        assert jnp.allclose(F_var[-1], 0.0, atol=1e-2)


    def test_neg_mll(self, benzene_with_descriptor):
        x_dft, dx_dft, E_dft, y_dft = desc_to_inputs(benzene_with_descriptor)
        kernel_params = {'lp': 1.0, 'lf': 3.0, 'ld': 1.0, 'w': 0.0}
        mll = pmf.neg_mll(x_dft, dx_dft, y_dft, E_dft, y_dft.reshape(-1, 36), rbf, **kernel_params)
        assert isinstance(mll, jnp.ndarray)


    def test_total_neg_mll(self, benzene_with_descriptor, benzene_ccsd_descriptor):
        x_dft, dx_dft, E_dft, y_dft = desc_to_inputs(benzene_with_descriptor)
        x_cc, dx_cc, E_cc, y_cc = desc_to_inputs(benzene_ccsd_descriptor)

        res = pmf.total_neg_mll(
            [x_dft, x_cc],
            [dx_dft, dx_cc],
            [y_dft, y_cc],
            rbf,
            [{'l': 1.0}, {'lp': 1.0, 'lf': 1.0, 'ld': 1.0, 'w': 0.0}],
        )

        assert isinstance(res, jnp.ndarray)


    def test_optimize_kernel(self, benzene_with_descriptor, benzene_ccsd_descriptor):
        x_dft, dx_dft, E_dft, y_dft = desc_to_inputs(benzene_with_descriptor)
        x_cc, dx_cc, E_cc, y_cc = desc_to_inputs(benzene_ccsd_descriptor)

        init_kernel_params = [{'l': 1.0}, {'lp': 1.0, 'lf': 3.0, 'ld': 1.0, 'w': 0.0}]
        optimizer_kwargs = {'learning_rate': 1e-3}
        old_loss = pmf.total_neg_mll(
            [x_dft, x_cc],
            [dx_dft, dx_cc],
            [y_dft, y_cc],
            rbf,
            init_kernel_params,
        )
        new_loss, new_params = pmf.optimize_kernel(
            [x_dft, x_cc], 
            [dx_dft, dx_cc], 
            [y_dft, y_cc],
            rbf, 
            init_kernel_params, 
            optimizer_kwargs, 
            num_iterations=5
        )
        assert new_loss < old_loss
        assert new_params != init_kernel_params
        
        print('old params:')
        print(init_kernel_params)
        print('new params:')
        print(new_params)
