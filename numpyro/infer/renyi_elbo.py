from jax import random, vmap, lax
from jax.lax import stop_gradient
from jax.scipy.linalg import logsumexp
import jax.numpy as np
from numpyro.infer.elbo import ELBO
from numpyro.infer.util import log_density
from numpyro.handlers import replay, seed

# should i inherit here?
class RenyiELBO(ELBO):
    r"""
    An implementation of Renyi's :math:`\alpha`-divergence variational inference
    following reference [1].
    In order for the objective to be a strict lower bound, we require
    :math:`\alpha \ge 0`. Note, however, that according to reference [1], depending
    on the dataset :math:`\alpha < 0` might give better results. In the special case
    :math:`\alpha = 0`, the objective function is that of the important weighted
    autoencoder derived in reference [2].
    .. note:: Setting :math:`\alpha < 1` gives a better bound than the usual ELBO.
        For :math:`\alpha = 1`, it is better to use
        :class:`~pyro.infer.trace_elbo.Trace_ELBO` class because it helps reduce
        variances of gradient estimations.
    .. warning:: Mini-batch training is not supported yet.
    :param float alpha: The order of :math:`\alpha`-divergence. Here
        :math:`\alpha \neq 1`. Default is 0.
    :param num_particles: The number of particles/samples used to form the objective
        (gradient) estimator. Default is 2.
    References:
    [1] `Renyi Divergence Variational Inference`,
        Yingzhen Li, Richard E. Turner
    [2] `Importance Weighted Autoencoders`,
        Yuri Burda, Roger Grosse, Ruslan Salakhutdinov
    """

    def __init__(self, alpha=0, num_particles=2):
        #TODO: Handle case alpha == 0
        if alpha == 1:
            # Trace_ELBO is not implemented for numpyro, what to do here? 
            raise ValueError("The order alpha should not be equal to 1. Please use Trace_ELBO class"
                             "for the case alpha = 1.")
        self.alpha = alpha
        super(RenyiELBO, self).__init__(num_particles=num_particles)

    def loss(self, rng, param_map, model, guide, *args, **kwargs):
        r"""
        :returns: returns an estimate of the Renyi ELBO
        :rtype: float
        Evaluates the Renyi ELBO with an estimator that uses num_particles many samples/particles.
        """
        def single_particle_elbo(rng, stop_gradient=False):
            model_seed, guide_seed = random.split(rng)
            seeded_model = seed(model, model_seed)
            seeded_guide = seed(guide, guide_seed)
            guide_log_density, guide_trace = log_density(seeded_guide, args, kwargs, param_map)
            # NB: we only want to substitute params not available in guide_trace
            model_param_map = {k: v for k, v in param_map.items() if k not in guide_trace}
            seeded_model = replay(seeded_model, guide_trace)
            model_log_density, _ = log_density(seeded_model, args, kwargs, model_param_map)

            # log p(z) - log q(z)
            if stop_gradient:
                elbo = stop_gradient(model_log_density) - stop_gradient(guide_log_density)
            else:
                elbo = model_log_density - guide_log_density
            # Return (-elbo) since by convention we do gradient descent on a loss and
            # the ELBO is a lower bound that needs to be maximized.
            return -elbo

        def single_particle_renyi_elbo(rng):
            elbo = - single_particle_elbo(rng)
            # not sure if this is numerically stable
            renyi_elbo = np.exp((1. - self.alpha) * elbo) / (1. - self.alpha)
            return -renyi_elbo

        rng_keys = random.split(rng, self.num_particles)
        return np.mean(vmap(single_particle_renyi_elbo)(rng_keys))

    def loss_and_grads(self, rng, param_map, model, guide, *args, **kwargs):
        r"""
        :returns: returns an estimate of the ELBO
        :rtype: float
        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        rng_keys = random.split(rng, self.num_particles)
        stop_gradients = [True for i in range(self.num_particles)]
        # not sure how to pass multiple args into vmap
        elbo_particles = vmap(single_particle_elbo)(rng_keys, stop_gradients)
        surrogate_elbo_particles = vmap(single_particle_renyi_elbo)(rng_keys)
        log_weights = (1. - self.alpha) * elbo_particles
        log_mean_weight = logsumexp(log_weights, axis=0) - math.log(self.num_particles)
        elbo = np.sum(log_mean_weight) / (1. - self.alpha)

