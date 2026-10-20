Automatic Guide Generation
==========================

We provide a brief overview of the automatically generated guides available in NumPyro:

* `AutoNormal <https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoNormal>`_ and `AutoDiagonalNormal <https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoDiagonalNormal>`_ are our basic mean-field guides. If the latent space is non-euclidean (due to e.g. a positivity constraint on one of the sample sites) an appropriate bijective transformation is automatically used under the hood to map between the unconstrained space (where the Normal variational distribution is defined) to the corresponding constrained space (note this is true for all automatic guides). These guides are a great place to start when trying to get variational inference to work on a model you are developing.
* `AutoMultivariateNormal <https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoMultivariateNormal>`_ and `AutoLowRankMultivariateNormal <https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoLowRankMultivariateNormal>`_ also construct Normal variational distributions but offer more flexibility, as they can capture correlations in the posterior. Note that these guides may be difficult to fit in the high-dimensional setting.
* `AutoDelta <https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoDelta>`_ is used for computing point estimates via MAP (maximum a posteriori estimation). See `here <https://github.com/pyro-ppl/numpyro/blob/bbe1f879eede79eebfdd16dfc49c77c4d1fc727c/examples/zero_inflated_poisson.py#L101>`_ for example usage.
* `AutoBNAFNormal <https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoBNAFNormal>`_ and `AutoIAFNormal <https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoIAFNormal>`_ offer flexible variational distributions parameterized by normalizing flows.
* `AutoDAIS <https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoDAIS>`_ is a powerful variational inference algorithm that leverages HMC. It can be a good choice for dealing with highly correlated posteriors but may be computationally expensive depending on the nature of the model.
* `AutoSurrogateLikelihoodDAIS <https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoSurrogateLikelihoodDAIS>`_ is a powerful variational inference algorithm that leverages HMC and that supports data subsampling.
* `AutoSemiDAIS <https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoSemiDAIS>`_ constructs a posterior approximation like `AutoDAIS <https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoDAIS>`_ for local latent variables but provides support for data subsampling during ELBO training by utilizing a parametric guide for global latent variables. 
* `AutoLaplaceApproximation <https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoLaplaceApproximation>`_ can be used to compute a Laplace approximation.
* `AutoGuideList <https://num.pyro.ai/en/latest/autoguide.html#numpyro.infer.autoguide.AutoGuideList>`_ can be used to combine multiple automatic guides.

.. automodule:: numpyro.infer.autoguide

AutoGuide
---------
.. autoclass:: numpyro.infer.autoguide.AutoGuide
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

AutoGuideList
-------------
.. autoclass:: numpyro.infer.autoguide.AutoGuideList
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

AutoContinuous
--------------
.. autoclass:: numpyro.infer.autoguide.AutoContinuous
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

AutoBNAFNormal
--------------
.. autoclass:: numpyro.infer.autoguide.AutoBNAFNormal
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

AutoDiagonalNormal
------------------
.. autoclass:: numpyro.infer.autoguide.AutoDiagonalNormal
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

AutoMultivariateNormal
----------------------
.. autoclass:: numpyro.infer.autoguide.AutoMultivariateNormal
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

AutoIAFNormal
-------------
.. autoclass:: numpyro.infer.autoguide.AutoIAFNormal
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

AutoLaplaceApproximation
------------------------
.. autoclass:: numpyro.infer.autoguide.AutoLaplaceApproximation
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

AutoLowRankMultivariateNormal
-----------------------------
.. autoclass:: numpyro.infer.autoguide.AutoLowRankMultivariateNormal
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

AutoNormal
----------
.. autoclass:: numpyro.infer.autoguide.AutoNormal
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

AutoDelta
---------
.. autoclass:: numpyro.infer.autoguide.AutoDelta
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

AutoDAIS
--------
.. autoclass:: numpyro.infer.autoguide.AutoDAIS
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

AutoSemiDAIS
------------
.. autoclass:: numpyro.infer.autoguide.AutoSemiDAIS
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

AutoSurrogateLikelihoodDAIS
---------------------------
.. autoclass:: numpyro.infer.autoguide.AutoSurrogateLikelihoodDAIS
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
