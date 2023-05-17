import pdb
from models.exact import gp_predict_energy, gp_predict
from kernels.hess import rbf, matern52, explicit_hess, _get_full_K, _get_full_K_iterative, get_K, hvp, bilinear_hess, get_diag_K, get_full_K, jac_K, jac_K_dx1, get_jac_K, flatten
from .test_models import benzene_with_descriptor, benzene_ccsd_descriptor, desc_to_inputs, benzene_coords, benzene_ccsd_coords
from descriptors.inv_dist import _inv_dist, inv_dist
from kernels.auto_hess import kernel_with_descriptor, finite_diff_dk_dx2, finite_diff_hess_k, fd_hess_batch
import kernels.multifidelity as mf
import kernels.perdikaris_mf as pmf
from jax import vmap, jvp, vjp, disable_jit, jacfwd, jacrev, grad, jit
from functools import partial
import jax 
import jax.numpy as jnp
import pytest


def check_psd(A):
    return jnp.all(jnp.linalg.eigvalsh(A + 1e-7 * jnp.eye(len(A))) > 0.0)


@pytest.fixture
def random_vec():
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, shape=(16,))


@pytest.fixture
def random_batch():
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, shape=(4, 16))


@pytest.fixture
def random_config():
    key = jax.random.PRNGKey(42)
    a = jax.random.normal(key, shape=(16,))
    new_key, subkey = jax.random.split(key)
    da = jax.random.normal(subkey, shape=(16, 8))
    new_key, subkey = jax.random.split(new_key)
    E = jax.random.normal(subkey)
    new_key, subkey = jax.random.split(new_key)
    F = jax.random.normal(subkey, shape=(8,))
    return a, da, E, F


@pytest.fixture
def random_config_batch():
    b = 4
    key = jax.random.PRNGKey(42)
    a = jax.random.normal(key, shape=(b, 16))
    new_key, subkey = jax.random.split(key)
    da = jax.random.normal(subkey, shape=(b, 16, 8))
    new_key, subkey = jax.random.split(new_key)
    E = jax.random.normal(subkey, shape=(b,))
    new_key, subkey = jax.random.split(new_key)
    F = jax.random.normal(subkey, shape=(b, 8))
    return a, da, E, F


@pytest.fixture
def pair_configs(benzene_ccsd_coords):
    i, j = 0, 1
    pos, E, F = benzene_ccsd_coords
    p1, p2 = pos[i], pos[j]
    E1, E2 = E[i], E[j]
    F1, F2 = F[i].flatten(), F[j].flatten()

    return (p1, E1, F1), (p2, E2, F2)


def test_flatten():
    batch = 10
    d = 20

    key = jax.random.PRNGKey(32)
    a = jax.random.normal(key, shape=(batch, batch, d, d))
    b = jnp.zeros((batch*d, batch*d))

    for i in range(batch):
        for j in range(batch):
            b = b.at[i*d:(i+1)*d, j*d:(j+1)*d].set(a[i, j]) 

    flat = flatten(a, batch, d, batch, d)
    
    assert jnp.allclose(flat, b)


def test_understanding():
    key = jax.random.PRNGKey(11)
    a, b = jax.random.normal(key, shape=(2, 8))
    kernel_fn = lambda x1,x2: x1**2 * x2**3

    hess_fn = jacfwd(grad(kernel_fn, argnums=1), argnums=0)
    vhess_fn = vmap(vmap(hess_fn, in_axes=(None, 0)), in_axes=(0, None))

    vhess_fn_res = vhess_fn(a, b)

    for i in range(8):
        for j in range(8):
            assert jnp.isclose(vhess_fn_res[i, j], 6 * a[i] * b[j]**2)


class TestKernel:
    def test_matern52(self, random_vec, random_batch):
        cov = matern52(random_vec, random_vec, 1.0)
        assert jnp.allclose(cov, jax.nn.softplus(1.0))
    
        vmap_matern = vmap(vmap(matern52, in_axes=(0, None, None), out_axes=0), in_axes=(None, 0, None), out_axes=1)
        cov = vmap_matern(random_batch, random_batch, 1.0) + 1e-4 * jnp.eye(len(random_batch))
        assert jnp.all(jnp.linalg.eigvalsh(cov) > 0.0)
    
    
    def test_explicit_hess(self, random_vec):
        H = explicit_hess(rbf, random_vec, random_vec, 1.0)
        assert jnp.allclose(H, 1.1596512 * jnp.eye(len(random_vec)))
    
    
    def test_explicit_hess_batch(self, random_batch):
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
    
    
    def test_hvp(self, random_vec):
        key = jax.random.PRNGKey(41)
        dx2 = jax.random.normal(key, shape=(16, 8))
        hess_vec_product = hvp(rbf, random_vec, random_vec, dx2, l=1.0)
        assert jnp.allclose(hess_vec_product, 1.1596512 * jnp.eye(16) @ dx2)
    
        K_ij = get_K(rbf, random_vec, random_vec, dx2, dx2, l=1.0)
        assert K_ij.shape == (8, 8)
    
   
    """
    def test_batched_K(self, random_batch):
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
        """
    
    def test_diag_K(self, random_vec):
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
    

    
    
    def test_bilinear_hess(self, random_vec):
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
    
    
    def test_bilinear_hess_real(self, pair_configs):
        (p1, E1, F1), (p2, E2, F2) = pair_configs
        x1, dx1 = inv_dist(p1)
        x2, dx2 = inv_dist(p2)
    
        K = bilinear_hess(rbf, x1, x2, dx1, dx2, l=1.0)
        partial_f = partial(kernel_with_descriptor, rbf, _inv_dist, l=1.0)
        fd_K = finite_diff_hess_k(partial_f, p1, p2).reshape(36, 36)
    
        autograd_K = jacfwd(grad(partial_f, argnums=1))(p1, p2).reshape(36, 36).T
    
        assert jnp.allclose(K, fd_K, rtol=1e-3)
        assert jnp.allclose(K, autograd_K)

"""
def test_bilinear_hess_desc(benzene_ccsd_descriptor, benzene_ccsd_coords):
    x, E, F = benzene_ccsd_coords
    desc, E, F = benzene_ccsd_descriptor
    a, da = desc

    res = get_full_K(rbf, a, a, da, da, l=1.0)
    assert jnp.all(jnp.linalg.eigvalsh(res + 1e-8 * jnp.eye(len(res))) > 0.0)
    # ground truth via autograd
    k_truth = partial(kernel_with_descriptor, rbf, _inv_dist)
    hess_truth = jacfwd(grad(k_truth, argnums=1), argnums=0)
    i, j = 0, 1
    res = bilinear_hess(rbf, a[i], a[i], da[i], da[i], l=1.0)
    truth = hess_truth(x[i], x[i], l=1.0)
    #pdb.set_trace()
    assert jnp.allclose(res, truth.reshape(36, 36))

    res = bilinear_hess(rbf, a[i], a[j], da[i], da[j], l=1.0)
    pdb.set_trace()
    truth = hess_truth(x[i], x[j], l=1.0).reshape(36, 36)
    assert jnp.all(jnp.linalg.eigvalsh(truth + 1e-8 * jnp.eye(len(truth))) > 0.0)
    assert jnp.all(jnp.linalg.eigvalsh(res + 1e-8 * jnp.eye(len(res))) > 0.0)
    assert jnp.allclose(res, truth)
"""

class TestBatchKernel:
    def test_iterative_K(self, random_batch):
        a = random_batch
        key = jax.random.PRNGKey(11)
        da = jax.random.normal(key, shape=(4, 16, 8))
    
        K = _get_full_K(rbf, a, a, da, da, l=1.0)
        iter_K = _get_full_K_iterative(rbf, a, a, da, da, l=1.0)
    
        assert jnp.allclose(K, iter_K)


    def test_get_full_K_block(self, random_batch):
        a = random_batch
        key = jax.random.PRNGKey(11)
        da = jax.random.normal(key, shape=(4, 16, 8))
        
        K = _get_full_K(rbf, a, a, da, da, l=1.0)

        hess_k_fn = jacfwd(grad(rbf, argnums=1))

        for i in range(4):
            for j in range(4):
                #pdb.set_trace()
                res = da[i].T @ hess_k_fn(a[i], a[j], l=1.0) @ da[j]
                res_eh = da[i].T @ explicit_hess(rbf, a[i], a[j], l=1.0) @ da[j]
                assert jnp.allclose(res, res_eh)
                assert jnp.allclose(K[i, j], res)
    
    
    def test_get_full_K(self, random_batch):
        a = random_batch
        key = jax.random.PRNGKey(11)
        da = jax.random.normal(key, shape=(4, 16, 8))
        
        K = get_full_K(rbf, a, a, da, da, l=1.0)
        assert check_psd(K)
    
    
    def test_get_full_K_real(self, benzene_with_descriptor):
        x, dx, E, y = desc_to_inputs(benzene_with_descriptor)
        k_fn = partial(rbf, l=1.0)
        """
        autograd_f = vmap(
            vmap(
                jacfwd(grad(k_fn, argnums=1), argnums=0),
                in_axes=(None, 0)
            ),
            in_axes=(0, None),
        )
        autograd_K = jnp.expand_dims(dx.transpose(0,2,1), 1) @ autograd_f(x, x) @ jnp.expand_dims(dx, 0)
        autograd_K = autograd_K.reshape(360, 360)
        pdb.set_trace()
        assert jnp.allclose(autograd_K, autograd_K.T)
        assert check_psd(autograd_K)
        """
        K = get_full_K(rbf, x, x, dx, dx, l=1.0)
        #pdb.set_trace()
        assert jnp.allclose(K, K.T)
        assert check_psd(K)
    
    
    def test_jac_K(self, random_vec):
        a = random_vec
        key = jax.random.PRNGKey(11)
        da = jax.random.normal(key, shape=(16, 8))
    
        j = jac_K(rbf, a, a, da, l=1.0)
        assert jnp.allclose(j, jnp.zeros(8))
    
        j_dx1 = jac_K_dx1(rbf, a, da, a, l=1.0)
        assert jnp.allclose(j, -j_dx1)
    
    
    def test_jac_K_real(self, pair_configs):
        (p1, E1, F1), (p2, E2, F2) = pair_configs
        x1, dx1 = inv_dist(p1)
        x2, dx2 = inv_dist(p2)
    
        K = jac_K(rbf, x1, x2, dx2, l=1.0)
        analytical_K = dx2.T @ (2 * rbf(x1, x2, l=1.0) * (x1 - x2) / jax.nn.softplus(1.0)**2)
        partial_f = partial(kernel_with_descriptor, rbf, _inv_dist, l=1.0)
        fd_K = finite_diff_dk_dx2(partial_f, p1, p2).flatten()
    
        assert jnp.allclose(analytical_K, fd_K)
        assert jnp.allclose(K, fd_K)
        assert jnp.allclose(K, fd_K, rtol=1e-3)
        
        K_dx1 = jac_K_dx1(rbf, x1, dx1, x2, l=1.0)
        analytical_K = dx1.T @ (-2 * rbf(x1, x2, l=1.0) * (x1 - x2) / jax.nn.softplus(1.0)**2)
        assert jnp.allclose(K_dx1, analytical_K)
    
    
    def test_jac_K_batch(self, random_batch):
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

    k_fn = partial(mf.perdikaris_kernel, rbf, lp=1e-5, lf=1e-5, ld=1.0)

    K = k_fn(a, a, 1.0, 1.0)
    assert jnp.allclose(K, 2.0)
    


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
    K_iter = mf.get_full_K(rbf, a, a, da, da, E, E, iterative=True, lp=1.0, lf=1.0, ld=1.0)

    assert jnp.allclose(K, K_iter)
    assert K.shape == (32, 32)
    assert jnp.all(jnp.linalg.eigvalsh(K + 1e-8 * jnp.eye(32, 32)) > 0.0)

    # test diag
    Kd = mf.get_diag_K(rbf, a, a, da, da, E, E, lp=1.0, lf=1.0, ld=1.0)
    assert Kd.shape == (32, )


def test_multifidelity_get_K_jac(random_batch):
    a = random_batch
    key = jax.random.PRNGKey(11)
    da = jax.random.normal(key, shape=(4, 16, 8))
    new_key, subkey = jax.random.split(key)
    E = jax.random.uniform(subkey, shape=(4,))

    jac_K = mf.get_jac_K(rbf, a, a, da, E, E, lp=1.0, lf=1.0, ld=1.0)
    assert jac_K.shape == (4, 32)


###
#Perdikaris multifidelity testing
###

class TestPerdikarisMF:
    def desc_perdikaris(self, p1, p2, x_train, dx_train, E_train, F_train, lp, lf, ld):
        x1, dx1 = inv_dist(p1)
        x1, dx1 = jnp.expand_dims(x1, 0), jnp.expand_dims(dx1, 0)
        x2, dx2 = inv_dist(p2)
        x2, dx2 = jnp.expand_dims(x2, 0), jnp.expand_dims(dx2, 0)
        
        # make an energy prediction. E must change with p1 and p2!

        #pdb.set_trace()
        E1, _ = gp_predict_energy(x1, dx1, x_train, dx_train, F_train.flatten(), rbf, l=1.0)
        E2, _ = gp_predict_energy(x2, dx2, x_train, dx_train, F_train.flatten(), rbf, l=1.0)
        return rbf(x1, x2, lp) * rbf(E1, E2, lf) + rbf(x1, x2, ld)

    def test_get_K(self, random_config):
        x, dx, E, F = random_config
        K = pmf.get_K(rbf, x, x, dx, dx, E, E, F, F, lp=1.0, lf=1.0, ld=1.0)
        assert K.shape == (8, 8)

    def test_get_K_real(self, pair_configs, benzene_ccsd_descriptor):
        (p1, E1, F1), (p2, E2, F2) = pair_configs

        (x, dx), E, F = benzene_ccsd_descriptor

        #k_fn = jit(partial(pmf.perdikaris_kernel, rbf, f_x1=E1, f_x2=E2, lp=1.0, lf=1.0, ld=1.0))
        #k_w_desc = partial(kernel_with_descriptor, pmf.perdikaris_kernel, _inv_dist) 
        k_fn = partial(self.desc_perdikaris, x_train=x, dx_train=dx, E_train=E, F_train=F.flatten(), lp=1.0, lf=1.0, ld=1.0)
        #fd_hess = finite_diff_hess_k(k_fn, p1, p2).reshape(36, 36).T
        autograd_jac = jacfwd(grad(k_fn, argnums=1), argnums=0)(p1, p2).reshape(36, 36).T
        #pdb.set_trace()

        #assert jnp.allclose(fd_hess, autograd_jac, rtol=1e-3)

        # sample guess energies and forces
        x1, dx1 = inv_dist(p1)
        x1_, dx1_ = jnp.expand_dims(x1, 0), jnp.expand_dims(dx1, 0)
        x2, dx2 = inv_dist(p2)
        x2_, dx2_ = jnp.expand_dims(x2, 0), jnp.expand_dims(dx2, 0)
        
        E_guess_1, _ = gp_predict_energy(x1_, dx1_, x, dx, F.flatten(), rbf, l=1.0)
        E_guess_2, _ = gp_predict_energy(x2_, dx2_, x, dx, F.flatten(), rbf, l=1.0)
        F_guess_1, _ = gp_predict(x1_, dx1_, x, dx, F.flatten(), rbf, l=1.0)
        F_guess_2, _ = gp_predict(x2_, dx2_, x, dx, F.flatten(), rbf, l=1.0)


        K = pmf.get_K(rbf, x1, x2, dx1, dx2, E_guess_1, E_guess_2, F_guess_1, F_guess_2, lp=1.0, lf=1.0, ld=1.0)

        assert jnp.allclose(K, autograd_jac)
    
    def test_get_K_jac(self, random_config):
        x, dx, E, F = random_config
        K = pmf.get_K_jac(rbf, x, x, dx, E, E, F, lp=1.0, lf=1.0, ld=1.0) 
        assert K.shape == (8,)

        # test against analytical solution
        K_analytical = pmf.get_K_jac_analytical(rbf, x, x, dx, E, E, F, lp={'l': 1.0}, lf={'l': 1.0}, ld={'l': 1.0})
        assert jnp.allclose(K, K_analytical)

        ## check via autograd
        #k_fn = partial(pmf.perdikaris_kernel, rbf, f_x1=E, f_x2=E, lp=1.0, lf=1.0, ld=1.0)
        #dk_dx2 = grad(k_fn, argnums=1)
        #auto_K = dk_dx2(x, x) @ dx
        #assert jnp.allclose(K, auto_K)

    def test_get_K_jac_real(self, pair_configs, benzene_ccsd_descriptor):
        (p1, E1, F1), (p2, E2, F2) = pair_configs

        (x, dx), E, F = benzene_ccsd_descriptor

        #k_fn = jit(partial(pmf.perdikaris_kernel, rbf, f_x1=E1, f_x2=E2, lp=1.0, lf=1.0, ld=1.0))
        #k_w_desc = partial(kernel_with_descriptor, pmf.perdikaris_kernel, _inv_dist)
        #pdb.set_trace()
        k_fn = partial(self.desc_perdikaris, x_train=x, dx_train=dx, E_train=E, F_train=F.flatten(), lp=1.0, lf=1.0, ld=1.0)
        fd_jac = finite_diff_dk_dx2(k_fn, p1, p2).flatten()
        autograd_jac = grad(k_fn, argnums=1)(p1, p2).flatten()

        assert jnp.allclose(fd_jac, autograd_jac, rtol=1e-3)
        

        x1, dx1 = inv_dist(p1)
        x1_, dx1_ = jnp.expand_dims(x1, 0), jnp.expand_dims(dx1, 0)
        x2, dx2 = inv_dist(p2)
        x2_, dx2_ = jnp.expand_dims(x2, 0), jnp.expand_dims(dx2, 0)
        
        E_guess_1, _ = gp_predict_energy(x1_, dx1_, x, dx, F.flatten(), rbf, l=1.0)
        E_guess_2, _ = gp_predict_energy(x2_, dx2_, x, dx, F.flatten(), rbf, l=1.0)
        F_guess_1, _ = gp_predict(x1_, dx1_, x, dx, F.flatten(), rbf, l=1.0)
        F_guess_2, _ = gp_predict(x2_, dx2_, x, dx, F.flatten(), rbf, l=1.0)
        
        # test against analytical solution
        K_analytical = pmf.get_K_jac_analytical(rbf, x1, x2, dx2, E_guess_1, E_guess_2, F_guess_2, lp={'l': 1.0}, lf={'l': 1.0}, ld={'l': 1.0})
        #pdb.set_trace()
        assert jnp.allclose(K_analytical, autograd_jac)

        K = pmf.get_K_jac(rbf, x1, x2, dx2, E_guess_1, E_guess_2, F_guess_2, lp=1.0, lf=1.0, ld=1.0)
        assert jnp.allclose(K, autograd_jac, rtol=1e-3)

    def test_get_full_K(self, random_config_batch):
        x, dx, E, F = random_config_batch
        K = pmf.get_full_K(rbf, x, x, dx, dx, E, E, F, F, lp=1.0, lf=1.0, ld=1.0)
        assert K.shape == (32, 32)
        assert jnp.all(jnp.linalg.eigvals(K + 1e-8 * jnp.eye(32)) > 0.0)
        #K_iter = pmf.get_full_K(rbf, x, x, dx, dx, E, E, F, F, iterative=True, lp={'l': 1.0}, lf={'l': 1.0}, ld={'l': 1.0})
        #assert jnp.allclose(K, K_iter)

    def test_get_full_K_block(self, benzene_with_descriptor, benzene_ccsd_descriptor, benzene_ccsd_coords):
        x_dft, dx_dft, E_dft, y_dft = desc_to_inputs(benzene_with_descriptor)
        x_cc, dx_cc, E_cc, y_cc = desc_to_inputs(benzene_ccsd_descriptor)
        p_cc, _, _ = benzene_ccsd_coords

        # assumes previous fidelity predicts energies and forces perfectly
        F_cc = y_cc.reshape(len(x_cc), -1)
        E_cc = jnp.expand_dims(E_cc, 1)

        k_fn = partial(self.desc_perdikaris, x_train=x_cc, dx_train=dx_cc, E_train=E_cc, F_train=F_cc, lp=1.0, lf=1.0, ld=1.0)
        autograd_k = jacfwd(grad(k_fn, argnums=1))

        # regular ol' GP to predict "previous fidelity" energy and forces
        # in reality, this is trained on E and F of the same fidelity
        E_guess, _ = gp_predict_energy(x_cc, dx_cc, x_cc, dx_cc, F_cc.flatten(), rbf, l=1.0)
        F_guess, _ = gp_predict(x_cc, dx_cc, x_cc, dx_cc, F_cc.flatten(), rbf, l=1.0)
        F_guess = F_guess.reshape(len(x_cc), -1)

        K = pmf._get_full_K(rbf, x_cc, x_cc, dx_cc, dx_cc, E_guess, E_guess, F_guess, F_guess, lp=1.0, lf=1.0, ld=1.0)

        for i in range(10):
            for j in range(10):
                assert jnp.allclose(K[i, j], autograd_k(p_cc[i], p_cc[j]).reshape(36, 36).T)

    def test_get_full_K_real(self, benzene_with_descriptor, benzene_ccsd_descriptor, benzene_ccsd_coords):
        x_dft, dx_dft, E_dft, y_dft = desc_to_inputs(benzene_with_descriptor)
        x_cc, dx_cc, E_cc, y_cc = desc_to_inputs(benzene_ccsd_descriptor)
        p_cc, _, _ = benzene_ccsd_coords

        # assumes previous fidelity predicts energies and forces perfectly
        F_cc = y_cc.reshape(len(x_cc), -1)
        E_cc = jnp.expand_dims(E_cc, 1)

        #k_fn = partial(self.desc_perdikaris, x_train=x_cc, dx_train=dx_cc, E_train=E_cc, F_train=F_cc, lp=1.0, lf=1.0, ld=1.0)
        #fd_hess = fd_hess_batch(k_fn, p_cc, p_cc).reshape(len(x_cc), len(x_cc), 36, 36).transpose(0,2,1,3).reshape(360, 360)  # not sure this is right...
        #assert check_psd(fd_hess)

        # regular ol' GP to predict "previous fidelity" energy and forces
        # in reality, this is trained on E and F of the same fidelity
        E_guess, _ = gp_predict_energy(x_cc, dx_cc, x_cc, dx_cc, F_cc.flatten(), rbf, l=1.0)
        F_guess, _ = gp_predict(x_cc, dx_cc, x_cc, dx_cc, F_cc.flatten(), rbf, l=1.0)
        F_guess = F_guess.reshape(len(x_cc), -1)
        #E_guess = jnp.expand_dims(E_guess, 1)

        K_ideal = pmf.get_full_K(rbf, x_cc, x_cc, dx_cc, dx_cc, E_guess, E_guess, F_guess, F_guess, lp=1.0, lf=1.0, ld=1.0)
        #assert jnp.allclose(fd_hess, K_ideal, rtol=1e-3)
        assert jnp.all(jnp.linalg.eigvalsh(K_ideal + 1e-8 * jnp.eye(len(K_ideal))) > 0.0)

        # now try to make actual predictions and make the kernel matrix

    def test_get_diag_K(self, random_config_batch):
        x, dx, E, F = random_config_batch
        K_diag = pmf.get_diag_K(rbf, x, x, dx, dx, E, E, F, F, lp=1.0, lf=1.0, ld=1.0)
        assert K_diag.shape == (32,)

    def test_get_jac_K(self, random_config_batch):
        x, dx, E, F = random_config_batch
        K_diag = pmf.get_jac_K(rbf, x, x, dx, E, E, F, lp=1.0, lf=1.0, ld=1.0)
        assert K_diag.shape == (4, 32)

