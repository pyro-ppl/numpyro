import jax.numpy as np_jax
import numpy as np
from jax import lax
def load_dataset(observations,features, batch_size=None, shuffle=True):

    arrays = (observations,features)
    num_records = observations.shape[0]
    idxs = np_jax.arange(num_records)
    if not batch_size:
        batch_size = num_records

    def init():
        return num_records // batch_size, np.random.permutation(idxs) if shuffle else idxs

    def get_batch(i=0, idxs=idxs):
        ret_idx = lax.dynamic_slice_in_dim(idxs, i * batch_size, batch_size)
        batch_data = np_jax.take(arrays[0], ret_idx, axis=0)
        batch_matrix =np_jax.take(np_jax.take(arrays[1], ret_idx, axis=0),ret_idx,axis=1)
        return (batch_data,batch_matrix)
    return init, get_batch
def svi_map(model, rng_key, feats,obs,num_epochs,batch_size):
    """
    MLE in numpy: https://medium.com/@rrfd/what-is-maximum-likelihood-estimation-examples-in-python-791153818030i
    Cost function: -log (likelihood(parameters|data)
    Calculate pdf of the parameter|data under the distribution
    """
    from jax import random, jit
    from numpyro import optim
    from numpyro.infer.elbo import RenyiELBO, ELBO
    from numpyro.infer.svi import SVI
    from numpyro.util import fori_loop
    import time
    import numpyro
    numpyro.set_platform("gpu")

    from autoguide_hmcecs import AutoDelta, AutoDiagonalNormal
    n, _ = feats.shape
    #guide = AutoDelta(model)
    guide = AutoDiagonalNormal(model)
    #loss = RenyiELBO(alpha=2, num_particles=1)
    loss = ELBO()
    svi = SVI(model, guide, optim.Adam(0.0003), loss=loss)
    svi_state = svi.init( rng_key,feats,obs)
    train_init, train_fetch = load_dataset(obs,feats, batch_size=batch_size)
    num_train, train_idx = train_init()

    @jit
    def epoch_train(svi_state):
        def body_fn(i, val):
            batch_obs = train_fetch(i, train_idx)[0]
            batch_feats = train_fetch(i, train_idx)[1]
            loss_sum, svi_state = val
            svi_state, loss = svi.update(svi_state, feats,obs)
            loss_sum += loss
            return loss_sum, svi_state

        return fori_loop(0, n, body_fn, (0., svi_state))

    for i in range(num_epochs):
        t_start = time.time()
        train_loss, svi_state = epoch_train(svi_state)
        print("Epoch {}: loss = {} ({:.2f} s.)".format(i, train_loss, time.time() - t_start))
    return svi.get_params(svi_state), svi, svi_state