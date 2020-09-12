# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import namedtuple
import inspect

from jax import random, vmap
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from numpyro.infer import init_to_uniform
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import initialize_model
from numpyro.util import identity


TFPKernelState = namedtuple('TFPKernelState', ['z', 'kernel_results', 'rng_key'])


def _extract_kernel_functions(kernel):

    def init_fn(z, rng_key):
        z_flat, unravel_fn = ravel_pytree(z)
        results = kernel.bootstrap_results(z_flat)
        return TFPKernelState(z, results, rng_key)

    def sample_fn(state, model_args=(), model_kwargs=None):
        rng_key, rng_key_transition = random.split(state.rng_key)
        z_flat, unravel_fn = ravel_pytree(state.z)
        z_new_flat, results = kernel.one_step(z_flat, state.kernel_results, seed=rng_key_transition)
        return TFPKernelState(unravel_fn(z_new_flat), results, rng_key)

    return init_fn, sample_fn


def _make_log_prob_fn(potential_fn, unravel_fn):
    def log_prob_fn(x):
        print(unravel_fn(x))
        return - potential_fn(unravel_fn(x))

    return log_prob_fn


class TFPKernel(MCMCKernel):
    """
    A thin wrapper for TensorFlow Probability MCMC transition kernels.

    :param model: Python callable containing Pyro :mod:`~numpyro.primitives`.
        If model is provided, `potential_fn` will be inferred using the model.
    :param target_log_prob_fn: Python callable that computes the target log
        probability (the negative of potential energy)
        given input parameters. The input parameters to `target_log_prob_fn`
        can be any python collection type, provided that `init_params` argument to
        :meth:`init` has the same type.
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    :param kernel_kwargs: other arguments to be passed to TFP kernel constructor.
    """
    kernel_class = None

    def __init__(self, model=None, target_log_prob_fn=None, init_strategy=init_to_uniform,
                 **kernel_kwargs):
        if not (model is None) ^ (target_log_prob_fn is None):
            raise ValueError('Only one of `model` or `target_log_prob_fn` must be specified.')
        self._model = model
        self._target_log_prob_fn = target_log_prob_fn
        self._kernel_kwargs = kernel_kwargs
        self._init_strategy = init_strategy
        # Set on first call to init
        self._init_fn = None
        self._sample_fn = None

    def _init_state(self, rng_key, model_args, model_kwargs, init_params):
        if self._model is not None:
            init_params, potential_fn, postprocess_fn, model_trace = initialize_model(
                rng_key,
                self._model,
                init_strategy=self._init_strategy,
                dynamic_args=True,
                model_args=model_args,
                model_kwargs=model_kwargs)
            init_params = init_params.z
            if self._init_fn is None:
                _, unravel_fn = ravel_pytree(init_params)
                kernel = self.kernel_class(
                    _make_log_prob_fn(potential_fn(*model_args, **model_kwargs), unravel_fn),
                    **self._kernel_kwargs)
                self._init_fn, self._sample_fn = _extract_kernel_functions(kernel)
            self._postprocess_fn = postprocess_fn
        elif self._kernel is None:
            kernel = self.kernel_class(self._target_log_prob_fn, **self._kernel_kwargs)
            self._init_fn, self._sample_fn = _extract_kernel_functions(kernel)
        return init_params

    @property
    def model(self):
        return self._model

    @property
    def sample_field(self):
        return 'z'

    @property
    def default_fields(self):
        return ('z',)

    def get_diagnostics_str(self, state):
        """
        Given the current `state`, returns the diagnostics string to
        be added to progress bar for diagnostics purpose.
        """
        return ''

    def init(self, rng_key, num_warmup, init_params=None, model_args=(), model_kwargs={}):
        # non-vectorized
        if rng_key.ndim == 1:
            rng_key, rng_key_init_model = random.split(rng_key)
        # vectorized
        else:
            rng_key, rng_key_init_model = jnp.swapaxes(vmap(random.split)(rng_key), 0, 1)
        init_params = self._init_state(rng_key_init_model, model_args, model_kwargs, init_params)
        if self._target_log_prob_fn and init_params is None:
            raise ValueError('Valid value of `init_params` must be provided with'
                             ' `target_log_prob_fn`.')

        if rng_key.ndim == 1:
            init_state = self._init_fn(init_params, rng_key)
        else:
            # XXX it is safe to run hmc_init_fn under vmap despite that hmc_init_fn changes some
            # nonlocal variables: momentum_generator, wa_update, trajectory_len, max_treedepth,
            # wa_steps because those variables do not depend on traced args: init_params, rng_key.
            init_state = vmap(self._init_fn)(init_params, rng_key)
            sample_fn = vmap(self._sample_fn, in_axes=(0, None, None))
            self._sample_fn = sample_fn
        return init_state

    def postprocess_fn(self, args, kwargs):
        if self._postprocess_fn is None:
            return identity
        return self._postprocess_fn(*args, **kwargs)

    def sample(self, state, model_args, model_kwargs):
        """
        Run the kernel from the given :data:`~numpyro.contrib.tfp.mcmc.TFPKernelState`
        and return the resulting :data:`~numpyro.contrib.tfp.mcmc.TFPKernelState`.

        :param TFPKernelState state: Represents the current state.
        :param model_args: Arguments provided to the model.
        :param model_kwargs: Keyword arguments provided to the model.
        :return: Next `state` after running the kernel.
        """
        return self._sample_fn(state, model_args, model_kwargs)


__all__ = []
for _name, _Kernel in tfp.mcmc.__dict__.items():
    if not isinstance(_Kernel, type):
        continue
    if not issubclass(_Kernel, tfp.mcmc.TransitionKernel):
        continue
    if 'target_log_prob_fn' not in inspect.getfullargspec(_Kernel).args:
        continue

    try:
        _PyroKernel = locals()[_name]
    except KeyError:
        _PyroKernel = type(_name, (TFPKernel,), {})
        _PyroKernel.__module__ = __name__
        _PyroKernel.kernel_class = _Kernel
        locals()[_name] = _PyroKernel

    _PyroKernel.__doc__ = '''
    Wraps `{}.{} <https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/mcmc/{}>`_
    with :class:`~numpyro.contrib.tfp.distributions.TFPDistributionMixin`.
    '''.format(_Kernel.__module__, _Kernel.__name__, _Kernel.__name__)

    __all__.append(_name)


# Create sphinx documentation.
__doc__ = '\n\n'.join([

    '''
    {0}
    ----------------------------------------------------------------
    .. autoclass:: numpyro.contrib.tfp.mcmc.{0}
    '''.format(_name)
    for _name in sorted(__all__)
])
