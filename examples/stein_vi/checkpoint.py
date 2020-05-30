import jax

from examples.stein_vi.sm_dmm import model, guide
from numpyro.examples.datasets import load_dataset, JSBCHORALES
from numpyro.guides import WrappedGuide
from numpyro.infer import ELBO
from numpyro.infer.kernels import RBFKernel
from numpyro.infer.stein import SVGD
from numpyro.optim import ClippedAdam

if __name__ == '__main__':
    ##
    batch_size = 1
    init, get_batch = load_dataset(JSBCHORALES, batch_size=batch_size, split='train')
    ds_count, ds_indxs = init()

    ##
    learning_rate = 0.0003
    lr_decay = 0.99996

    optimizer = ClippedAdam(step_size=lambda i: learning_rate * lr_decay ** i, b1=0.96, b2=0.999, clip_norm=10)
    svgd = SVGD(model, WrappedGuide(guide, reinit_hide_fn=lambda site: site['name'].endswith('$params')),
                optimizer, ELBO(), RBFKernel(), num_particles=10,
                repulsion_temperature=batch_size)

    ##
    rng_key = jax.random.PRNGKey(seed=142)
    seqs, seqs_rev, lengths = get_batch(0, ds_indxs)
    state = svgd.init(rng_key, seqs, seqs_rev, lengths)

    svgd.store_checkout(state, 0)

    ##
    svgd = SVGD(model, WrappedGuide(guide, reinit_hide_fn=lambda site: site['name'].endswith('$params')),
                optimizer, ELBO(), RBFKernel(), num_particles=10,
                repulsion_temperature=batch_size)
    state, it = svgd.load_latest_checkout(rng_key)
    print(it)
