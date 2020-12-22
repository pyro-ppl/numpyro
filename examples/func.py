from functools import partial

import numpy as np

from jax import grad, vmap, jit, jvp, vjp, custom_jvp
import jax.numpy as jnp
import jax.random as random
from jax.lax import while_loop, stop_gradient, dynamic_slice_in_dim


#def forward(alpha, x):
#    return x + (2.0 / 3.0) * alpha * jnp.power(x, 3.0) + 0.2 * jnp.square(alpha) * jnp.power(x, 5.0)
#def forward(alpha, beta, x):
#    return x + (2.0 / 3.0) * alpha * jnp.power(x, 3.0) + 0.2 * jnp.square(alpha) * jnp.power(x, 5.0)


def forward(alpha, beta, x):
    return x + (1.0 / 3.0) * jnp.square(alpha) * jnp.power(x, 3.0) + \
           0.5 * alpha * beta * jnp.power(x, 4.0) + \
           0.2 * jnp.square(beta) * jnp.power(x, 5.0)


def cond_fn(val):
    return (val[2] > 1.0e-6) & (val[3] < 200)


def body_fn(alpha, beta, val):
    x, y, _, i = val
    f = partial(forward, alpha, beta)
    df = grad(f)
    delta = (f(x) - y) / df(x)
    x = x - delta
    return (x, y, jnp.fabs(delta), i + 1)


@custom_jvp
def inverse(alpha, beta, y):
    return while_loop(cond_fn, partial(body_fn, alpha, beta), (y, y, 9.9e9, 0))[0]


@inverse.defjvp
def inverse_jvp(primals, tangents):
  alpha, beta, y = primals
  alpha_dot, beta_dot, y_dot = tangents
  primal_out = inverse(alpha, beta, y)
  y_tilde = primal_out
  denominator = 1.0 + jnp.square(alpha * y_tilde + beta * jnp.square(y_tilde))
  dalpha = (2.0 / 3.0) * alpha * jnp.power(y_tilde, 3.0) + 0.5 * beta * jnp.power(y_tilde, 4.0)
  dbeta = 0.5 * alpha * jnp.power(y_tilde, 4.0) + 0.4 * beta * jnp.power(y_tilde, 5.0)
  numerator = dalpha * alpha_dot + dbeta * beta_dot
  tangent_out = -numerator / denominator
  return primal_out, tangent_out


def jacobian_and_inverse(alpha, beta, Y):
    Y_tilde = vmap(lambda y: inverse(alpha, beta, y))(Y)
    Y_tilde_stop = stop_gradient(Y_tilde)
    log_det_jacobian = -jnp.sum(jnp.log(1.0 + jnp.square(alpha * Y_tilde_stop + beta * jnp.square(Y_tilde_stop))))
    return Y_tilde, log_det_jacobian


# tests
if __name__ == '__main__':
    Y = np.random.randn(2)
    alpha = np.random.randn(1)[0]
    beta = np.zeros(1)[0]
    Y_tilde, jac = jacobian_and_inverse(alpha, beta, Y)
    ffY = forward(alpha, beta, Y_tilde)
    delta = np.fabs(Y - ffY)
    print("Ytilde", Y_tilde, "Y", Y, "ffY", ffY, "delta", delta)

    def Yt(Y, alpha):
        return jacobian_and_inverse(alpha, beta, Y)[0][0]

    g = grad(Yt, argnums=1)(Y, alpha)
    print("dYtilde/dalpha", g)

    def Yt(Y, alpha):
        return jacobian_and_inverse(alpha, beta, Y)[1]

    g = grad(Yt, argnums=1)(Y, alpha)
    print("dlogjac/dalpha", g)

    def YYt(Y, alpha, beta):
        return jnp.sum(forward(alpha, beta, jacobian_and_inverse(alpha, beta, Y)[0]))

    g = grad(YYt, argnums=1)(Y, alpha, beta)
    print("didentity/dalpha", g)
    g = grad(YYt, argnums=2)(Y, alpha, beta)
    print("didentity/dbeta", g)
