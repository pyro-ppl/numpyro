Base Distribution
=================

Distribution
------------
.. autoclass:: numpyro.distributions.distribution.Distribution
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

TransformedDistribution
-----------------------
.. autoclass:: numpyro.distributions.distribution.TransformedDistribution
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource


Continuous Distributions
========================

Beta
----
.. autoclass:: numpyro.distributions.continuous.Beta
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Cauchy
------
.. autoclass:: numpyro.distributions.continuous.Cauchy
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Chi2
----
.. autoclass:: numpyro.distributions.continuous.Chi2
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Dirichlet
---------
.. autoclass:: numpyro.distributions.continuous.Dirichlet
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Exponential
-----------
.. autoclass:: numpyro.distributions.continuous.Exponential
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Gamma
-----
.. autoclass:: numpyro.distributions.continuous.Gamma
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

GaussianRandomWalk
------------------
.. autoclass:: numpyro.distributions.continuous.GaussianRandomWalk
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

HalfCauchy
----------
.. autoclass:: numpyro.distributions.continuous.HalfCauchy
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

HalfNormal
----------
.. autoclass:: numpyro.distributions.continuous.HalfNormal
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

InverseGamma
------------
.. autoclass:: numpyro.distributions.continuous.InverseGamma
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

LKJCholesky
-----------
.. autoclass:: numpyro.distributions.continuous.LKJCholesky
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

LogNormal
---------
.. autoclass:: numpyro.distributions.continuous.LogNormal
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Normal
------
.. autoclass:: numpyro.distributions.continuous.Normal
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Pareto
------
.. autoclass:: numpyro.distributions.continuous.Pareto
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

StudentT
--------
.. autoclass:: numpyro.distributions.continuous.StudentT
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

TruncatedCauchy
---------------
.. autoclass:: numpyro.distributions.continuous.TruncatedCauchy
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

TruncatedNormal
---------------
.. autoclass:: numpyro.distributions.continuous.TruncatedNormal
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Uniform
-------
.. autoclass:: numpyro.distributions.continuous.Uniform
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource


Discrete Distributions
======================

Bernoulli
---------
.. autofunction:: numpyro.distributions.discrete.Bernoulli

BernoulliLogits
---------------
.. autoclass:: numpyro.distributions.discrete.BernoulliLogits
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

BernoulliProbs
--------------
.. autoclass:: numpyro.distributions.discrete.BernoulliProbs
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Binomial
---------
.. autofunction:: numpyro.distributions.discrete.Binomial

BinomialLogits
---------------
.. autoclass:: numpyro.distributions.discrete.BinomialLogits
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

BinomialProbs
-------------
.. autoclass:: numpyro.distributions.discrete.BinomialProbs
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Categorical
-----------
.. autofunction:: numpyro.distributions.discrete.Categorical

CategoricalLogits
-----------------
.. autoclass:: numpyro.distributions.discrete.CategoricalLogits
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

CategoricalProbs
----------------
.. autoclass:: numpyro.distributions.discrete.CategoricalProbs
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Delta
-----
.. autoclass:: numpyro.distributions.discrete.Delta
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Multinomial
-----------
.. autofunction:: numpyro.distributions.discrete.Multinomial

MultinomialLogits
-----------------
.. autoclass:: numpyro.distributions.discrete.MultinomialLogits
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

MultinomialProbs
----------------
.. autoclass:: numpyro.distributions.discrete.MultinomialProbs
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Poisson
-------
.. autoclass:: numpyro.distributions.discrete.Poisson
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource


Constraints
===========

biject_to
---------
.. autofunction:: numpyro.distributions.constraints.biject_to

boolean
-------
.. autodata:: numpyro.distributions.constraints.boolean

corr_cholesky
-------------
.. autodata:: numpyro.distributions.constraints.corr_cholesky

dependent
---------
.. autodata:: numpyro.distributions.constraints.dependent

greater_than
------------
.. autofunction:: numpyro.distributions.constraints.greater_than

integer_interval
----------------
.. autofunction:: numpyro.distributions.constraints.integer_interval

integer_greater_than
--------------------
.. autofunction:: numpyro.distributions.constraints.integer_greater_than

interval
--------
.. autofunction:: numpyro.distributions.constraints.interval

lower_cholesky
--------------
.. autodata:: numpyro.distributions.constraints.lower_cholesky

multinomial
-----------
.. autofunction:: numpyro.distributions.constraints.multinomial

nonnegative_integer
-------------------
.. autodata:: numpyro.distributions.constraints.nonnegative_integer

positive
--------
.. autodata:: numpyro.distributions.constraints.positive

positive_definite
-----------------
.. autodata:: numpyro.distributions.constraints.positive_definite

positive_integer
-----------------
.. autodata:: numpyro.distributions.constraints.positive_integer

real
----
.. autodata:: numpyro.distributions.constraints.real

real_vector
-----------
.. autodata:: numpyro.distributions.constraints.real_vector

simplex
-------
.. autodata:: numpyro.distributions.constraints.simplex

unit_interval
-------------
.. autodata:: numpyro.distributions.constraints.unit_interval


Transforms
==========

Transform
---------
.. autoclass:: numpyro.distributions.constraints.Transform
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

AbsTransform
------------
.. autoclass:: numpyro.distributions.constraints.AbsTransform
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

AffineTransform
---------------
.. autoclass:: numpyro.distributions.constraints.AffineTransform
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

ComposeTransform
----------------
.. autoclass:: numpyro.distributions.constraints.ComposeTransform
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

CorrCholeskyTransform
---------------------
.. autoclass:: numpyro.distributions.constraints.CorrCholeskyTransform
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

ExpTransform
------------
.. autoclass:: numpyro.distributions.constraints.ExpTransform
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

IdentityTransform
-----------------
.. autoclass:: numpyro.distributions.constraints.IdentityTransform
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

LowerCholeskyTransform
----------------------
.. autoclass:: numpyro.distributions.constraints.LowerCholeskyTransform
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

PermuteTransform
----------------
.. autoclass:: numpyro.distributions.constraints.PermuteTransform
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

PowerTransform
--------------
.. autoclass:: numpyro.distributions.constraints.PowerTransform
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

SigmoidTransform
----------------
.. autoclass:: numpyro.distributions.constraints.SigmoidTransform
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

StickBreakingTransform
----------------------
.. autoclass:: numpyro.distributions.constraints.StickBreakingTransform
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource


Flows
=====

InverseAutoregressiveTransform
------------------------------
.. autoclass:: numpyro.distributions.flows.InverseAutoregressiveTransform
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource
