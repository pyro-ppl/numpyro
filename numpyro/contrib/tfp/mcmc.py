# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABCMeta
from collections import namedtuple
import inspect

from jax import random, tree_map, vmap
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
        z_flat, _ = ravel_pytree(z)
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
        # we deal with batched x in case the kernel is ReplicaExchangeMC
        batch_shape = jnp.shape(x)[:-1]
        if batch_shape:
            flatten_result = vmap(lambda a: -potential_fn(unravel_fn(a)))(
                jnp.reshape(x, (-1,) + jnp.shape(x)[-1:]))
            return tree_map(lambda a: jnp.reshape(a, batch_shape + jnp.shape(a)[1:]),
                            flatten_result)
        else:
            return - potential_fn(unravel_fn(x))

    return log_prob_fn


class _TFPKernelMeta(ABCMeta):
    def __getitem__(cls, kernel_class):
        assert issubclass(kernel_class, tfp.mcmc.TransitionKernel)
        assert 'target_log_prob_fn' in inspect.getfullargspec(kernel_class).args, \
            f"the first argument of {kernel_class} must be `target_log_prob_fn`"

        _PyroKernel = type(kernel_class.__name__, (TFPKernel,), {})
        _PyroKernel.kernel_class = kernel_class
        return _PyroKernel


class TFPKernel(MCMCKernel, metaclass=_TFPKernelMeta):
    """
    A thin wrapper for TensorFlow Probability (TFP) MCMC transition kernels.
    The argument `target_log_prob_fn` in TFP is replaced by either `model`
    or `potential_fn` (which is the negative of `target_log_prob_fn`).

    This class can be used to convert a TFP kernel to a NumPyro-compatible one
    as follows::

        kernel = TFPKernel[tfp.mcmc.NoUTurnSampler](model, step_size=1.)

    .. note:: By default, uncalibrated kernels will be inner kernels of the
        :class:`~tensorflow_probability.substrates.jax.mcmc.MetropolisHastings` kernel.

    .. note:: For :class:`~numpyro.contrib.tfp.mcmc.ReplicaExchangeMC`, TFP requires
        that the shape of `step_size` of the inner kernel must be
        `[len(inverse_temperatures), 1]` or `[len(inverse_temperatures), latent_size]`.

    :param model: Python callable containing Pyro :mod:`~numpyro.primitives`.
        If model is provided, `potential_fn` will be inferred using the model.
    :param potential_fn: Python callable that computes the target potential energy
        given input parameters. The input parameters to `potential_fn`
        can be any python collection type, provided that `init_params` argument to
        :meth:`init` has the same type.
    :param callable init_strategy: a per-site initialization function.
        See :ref:`init_strategy` section for available functions.
    :param kernel_kwargs: other arguments to be passed to TFP kernel constructor.
    """
    kernel_class = None

    def __init__(self, model=None, potential_fn=None, init_strategy=init_to_uniform,
                 **kernel_kwargs):
        if not (model is None) ^ (potential_fn is None):
            raise ValueError('Only one of `model` or `potential_fn` must be specified.')
        self._model = model
        self._potential_fn = potential_fn
        self._kernel_kwargs = kernel_kwargs
        self._init_strategy = init_strategy
        # Set on first call to init
        self._init_fn = None
        self._postprocess_fn = None
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
                # Uncalibrated... kernels have to used inside MetropolisHastings, see
                # https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/mcmc/UncalibratedLangevin
                if self.kernel_class.__name__.startswith("Uncalibrated"):
                    kernel = tfp.mcmc.MetropolisHastings(kernel)
                self._init_fn, self._sample_fn = _extract_kernel_functions(kernel)
            self._postprocess_fn = postprocess_fn
        elif self._init_fn is None:
            _, unravel_fn = ravel_pytree(init_params)
            kernel = self.kernel_class(
                _make_log_prob_fn(self._potential_fn, unravel_fn),
                **self._kernel_kwargs)
            if self.kernel_class.__name__.startswith("Uncalibrated"):
                kernel = tfp.mcmc.MetropolisHastings(kernel)
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
        if self._potential_fn and init_params is None:
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


__all__ = ['TFPKernel']
for _name, _Kernel in tfp.mcmc.__dict__.items():
    if not isinstance(_Kernel, type):
        continue
    if not issubclass(_Kernel, tfp.mcmc.TransitionKernel):
        continue
    if 'target_log_prob_fn' not in inspect.getfullargspec(_Kernel).args:
        continue

    _PyroKernel = TFPKernel[_Kernel]
    _PyroKernel.__module__ = __name__
    locals()[_name] = _PyroKernel

    _PyroKernel.__doc__ = '''
    Wraps `{}.{} <https://www.tensorflow.org/probability/api_docs/python/tfp/substrates/jax/mcmc/{}>`_
    with :class:`~numpyro.contrib.tfp.mcmc.TFPKernel`. The first argument `target_log_prob_fn`
    in TFP kernel construction is replaced by either `model` or `potential_fn`.
    '''.format(_Kernel.__module__, _Kernel.__name__, _Kernel.__name__)

    __all__.append(_name)


# Create sphinx documentation.
__doc__ = '\n\n'.join([

    '''
    {0}
    ----------------------------------------------------------------
    .. autoclass:: numpyro.contrib.tfp.mcmc.{0}
    '''.format(_name)
    for _name in __all__[:1] + sorted(__all__[1:])
])
