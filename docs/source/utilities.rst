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

compute_log_probs
-----------------
.. autofunction:: numpyro.infer.util.compute_log_probs

get_transforms
--------------
.. autofunction:: numpyro.infer.util.get_transforms

transform_fn
------------
.. autofunction:: numpyro.infer.util.transform_fn

constrain_fn
------------
.. autofunction:: numpyro.infer.util.constrain_fn

unconstrain_fn
--------------
.. autofunction:: numpyro.infer.util.unconstrain_fn

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

init_to_feasible
^^^^^^^^^^^^^^^^
.. autofunction:: numpyro.infer.initialization.init_to_feasible

init_to_mean
^^^^^^^^^^^^
.. autofunction:: numpyro.infer.initialization.init_to_mean

init_to_median
^^^^^^^^^^^^^^
.. autofunction:: numpyro.infer.initialization.init_to_median

init_to_sample
^^^^^^^^^^^^^^
.. autofunction:: numpyro.infer.initialization.init_to_sample

init_to_uniform
^^^^^^^^^^^^^^^
.. autofunction:: numpyro.infer.initialization.init_to_uniform

init_to_value
^^^^^^^^^^^^^
.. autofunction:: numpyro.infer.initialization.init_to_value

Tensor Indexing
---------------

.. automodule:: numpyro.ops.indexing
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Model Inspection
----------------

get_dependencies
^^^^^^^^^^^^^^^^
.. autofunction:: numpyro.infer.inspect.get_dependencies

get_model_relations
^^^^^^^^^^^^^^^^^^^
.. autofunction:: numpyro.infer.inspect.get_model_relations

Visualization Utilities
=======================

render_model
------------
.. autofunction:: numpyro.infer.inspect.render_model

Trace Inspection
----------------
.. autofunction:: numpyro.util.format_shapes
