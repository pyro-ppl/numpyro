# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Conditional Variational Autoencoder in Flax
====================================================

This example trains a *Conditional Variational Autoencoder* (CVAE) [1] on the MNIST data
using Flax' neural network API. The implementation can be found here:
https://github.com/pyro-ppl/numpyro/tree/master/examples/cvae-flax

The model is a port of Pyro's excellent CVAE example which describes the model as well as the data in detail:
https://pyro.ai/examples/cvae.html

The model first trains a baseline to predict an entire MNIST image from a single quadrant of it
(i.e., input is one quadrant of an image, output is the entire image (not the other three quadrants)).
Then, in a second model, the generation/prior/recognition nets of the CVAE are trained while keeping the model
parameters of the baseline fixed/frozen. We use Optax' `multi_transform` to apply different gradient transformations
to the trainable parameters and the frozen parameters.


.. image:: ../_static/img/examples/cvae.png
    :align: center

**References:**

    1. Kihyuk Sohn, Xinchen Yan, Honglak Lee (2015), "Learning Structured Output Representation using Deep
       Conditional Generative Models
       (https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models)

"""
