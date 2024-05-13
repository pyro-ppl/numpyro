## Conditional Variational Autoencoder in Flax

Trains a *Conditional Variational Autoencoder* (CVAE) on the MNIST data using Flax' neural network API.

The model first trains a baseline to predict an entire MNIST image from a single quadrant of it (i.e., input is one quadrant of an image, output is the entire image (not the other three quadrants)).
Then, in a second model, the generation/prior/recognition nets of the CVAE are trained while keeping the model parameters of the baseline fixed/frozen.
We use Optax' `multi_transform` to apply different gradient transformations to the trainable parameters and the frozen parameters.

Running `main.py` trains the model(s) and plots a figure in the end comparing the baseline prediction with the CVAE prediction like this one:

![CVAE prediction](https://github.com/pyro-ppl/numpyro/tree/master/docs/source/_static/img/examples/cvae.png)
