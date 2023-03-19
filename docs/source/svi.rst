Stochastic Variational Inference (SVI)
======================================

We offer a brief overview of the three most commonly used ELBO implementations in NumPyro:

* `Trace_ELBO <https://num.pyro.ai/en/latest/svi.html#numpyro.infer.elbo.Trace_ELBO>`_ is our basic ELBO implementation.
* `TraceMeanField_ELBO <https://num.pyro.ai/en/latest/svi.html#numpyro.infer.elbo.TraceMeanField_ELBO>`_ is like ``Trace_ELBO`` but computes part of the ELBO analytically if doing so is possible.
* `TraceGraph_ELBO <https://num.pyro.ai/en/latest/svi.html#numpyro.infer.elbo.TraceGraph_ELBO>`_ offers variance reduction strategies for models with discrete latent variables. Generally speaking, this ELBO should always be used for models with discrete latent variables.
* `TraceEnum_ELBO <https://num.pyro.ai/en/latest/svi.html#numpyro.infer.elbo.TraceEnum_ELBO>`_ offers variable enumeration strategies for models with discrete latent variables. Generally speaking, this ELBO should always be used for models with discrete latent variables when enumeration is possible.

.. autoclass:: numpyro.infer.svi.SVI
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

.. autodata:: numpyro.infer.svi.SVIState

.. autodata:: numpyro.infer.svi.SVIRunResult

ELBO
----

.. autoclass:: numpyro.infer.elbo.ELBO
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Trace_ELBO
----------

.. autoclass:: numpyro.infer.elbo.Trace_ELBO
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource


TraceEnum_ELBO
---------------

.. autoclass:: numpyro.infer.elbo.TraceEnum_ELBO
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource


TraceGraph_ELBO
---------------

.. autoclass:: numpyro.infer.elbo.TraceGraph_ELBO
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource


TraceMeanField_ELBO
-------------------

.. autoclass:: numpyro.infer.elbo.TraceMeanField_ELBO
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

RenyiELBO
---------

.. autoclass:: numpyro.infer.elbo.RenyiELBO
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
