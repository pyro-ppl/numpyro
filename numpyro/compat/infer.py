import math

import numpyro
from numpyro import mcmc
import numpyro.distributions as dist


class HMC(mcmc.HMC):
    def __init__(self,
                 model=None,
                 potential_fn=None,
                 step_size=1,
                 adapt_step_size=True,
                 adapt_mass_matrix=True,
                 full_mass=False,
                 use_multinomial_sampling=True,
                 transforms=None,
                 max_plate_nesting=None,
                 jit_compile=False,
                 jit_options=None,
                 ignore_jit_warnings=False,
                 trajectory_length=2 * math.pi,
                 target_accept_prob=0.8):
        super(HMC, self).__init__(model=model,
                                  potential_fn=potential_fn,
                                  step_size=step_size,
                                  adapt_step_size=adapt_step_size,
                                  adapt_mass_matrix=adapt_mass_matrix,
                                  dense_mass=full_mass,
                                  target_accept_prob=target_accept_prob,
                                  trajectory_length=trajectory_length)


class NUTS(mcmc.NUTS):
    def __init__(self,
                 model=None,
                 potential_fn=None,
                 step_size=1,
                 adapt_step_size=True,
                 adapt_mass_matrix=True,
                 full_mass=False,
                 use_multinomial_sampling=True,
                 transforms=None,
                 max_plate_nesting=None,
                 jit_compile=False,
                 jit_options=None,
                 ignore_jit_warnings=False,
                 trajectory_length=2 * math.pi,
                 target_accept_prob=0.8,
                 max_tree_depth=10):
        if potential_fn is not None:
            raise ValueError('Only `model` argument is supported in generic module;'
                             ' `potential_fn` is not supported.')
        super(NUTS, self).__init__(model=model,
                                   potential_fn=potential_fn,
                                   step_size=step_size,
                                   adapt_step_size=adapt_step_size,
                                   adapt_mass_matrix=adapt_mass_matrix,
                                   dense_mass=full_mass,
                                   target_accept_prob=target_accept_prob,
                                   trajectory_length=trajectory_length,
                                   max_tree_depth=max_tree_depth)


class MCMC(object):
    def __init__(self,
                 kernel,
                 num_samples,
                 warmup_steps=None,
                 initial_params=None,
                 num_chains=1,
                 hook_fn=None,
                 mp_context=None,
                 disable_progbar=False,
                 disable_validation=True,
                 transforms=None):
        if warmup_steps is None:
            warmup_steps = num_samples
        self._initial_params = initial_params
        self._mcmc = mcmc.MCMC(kernel,
                               warmup_steps,
                               num_samples,
                               num_chains=num_chains,
                               progress_bar=(not disable_progbar))

    def run(self, *args, rng=None, **kwargs):
        if rng is None:
            rng = numpyro.sample('mcmc.run', dist.PRNGIdentity())
        self._mcmc.run(rng, *args, init_params=self._initial_params, **kwargs)

    def get_samples(self, num_samples=None, group_by_chain=False):
        if num_samples is not None:
            raise ValueError('`num_samples` arg unsupported in NumPyro.')
        return self._mcmc.get_samples(group_by_chain=group_by_chain)

    def summary(self, prob=0.9):
        self._mcmc.print_summary()
