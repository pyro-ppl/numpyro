Runtime Utilities
=================

enable_validation
-----------------
.. autofunction:: numpyro.distributions.distribution.enable_validation

validation_enabled
------------------
.. autofunction:: numpyro.distributions.distribution.validation_enabled

.. automodule:: numpyro.util

enable_x64
----------
.. autofunction:: numpyro.util.enable_x64

set_platform
------------
.. autofunction:: numpyro.util.set_platform

set_host_device_count
---------------------
.. autofunction:: numpyro.util.set_host_device_count

Inference Utilities
===================

.. automodule:: numpyro.infer.util

Predictive
----------
.. autoclass:: numpyro.infer.util.Predictive
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

log_density
-----------
.. autofunction:: numpyro.infer.util.log_density

transform_fn
------------
.. autofunction:: numpyro.infer.util.transform_fn

constrain_fn
------------
.. autofunction:: numpyro.infer.util.constrain_fn

potential_energy
----------------
.. autofunction:: numpyro.infer.util.potential_energy

log_likelihood
--------------
.. autofunction:: numpyro.infer.util.log_likelihood

find_valid_initial_params
-------------------------
.. autofunction:: numpyro.infer.util.find_valid_initial_params

.. _init_strategy:

Initialization Strategies
-------------------------

init_to_median
^^^^^^^^^^^^^^
.. autofunction:: numpyro.infer.util.init_to_median

init_to_prior
^^^^^^^^^^^^^
.. autofunction:: numpyro.infer.util.init_to_prior

init_to_uniform
^^^^^^^^^^^^^^^
.. autofunction:: numpyro.infer.util.init_to_uniform

init_to_feasible
^^^^^^^^^^^^^^^^
.. autofunction:: numpyro.infer.util.init_to_feasible

init_to_value
^^^^^^^^^^^^^
.. autofunction:: numpyro.infer.util.init_to_value

Tensor Indexing
---------------

.. automodule:: numpyro.contrib.indexing
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
