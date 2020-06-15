Reparameterizers
================

.. automodule:: numpyro.infer.reparam

The :mod:`numpyro.infer.reparam` module contains reparameterization strategies for
the :class:`numpyro.handlers.reparam` effect. These are useful for altering
geometry of a poorly-conditioned parameter space to make the posterior better
shaped. These can be used with a variety of inference algorithms, e.g.
``Auto*Normal`` guides and MCMC.

.. autoclass:: numpyro.infer.reparam.Reparam
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
    :special-members: __call___

Neural Transport
----------------
.. autoclass:: numpyro.infer.reparam.NeuTraReparam
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
    :special-members: __call__

Transformed Distributions
-------------------------
.. autoclass:: numpyro.infer.reparam.TransformReparam
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
    :special-members: __call__
