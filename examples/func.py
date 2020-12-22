from functools import partial

import numpy as np

from jax import grad, vmap, jit, jvp, vjp, custom_jvp, value_and_grad
import jax.numpy as jnp
import jax.random as random
from jax.lax import while_loop, stop_gradient, dynamic_slice_in_dim


def forward(alpha, beta, x):
    return x + (1.0 / 3.0) * jnp.square(alpha) * jnp.power(x, 3.0) + \
           0.5 * alpha * beta * jnp.power(x, 4.0) + \
           0.2 * jnp.square(beta) * jnp.power(x, 5.0)


def cond_fn(val):
    return (val[2] > 1.0e-6) & (val[3] < 100)


def body_fn(alpha, beta, val):
    x, y, _, i = val
    f = partial(forward, alpha, beta)
    fx, dfx = value_and_grad(f)(x)
    delta = (fx - y) / dfx
    x = x - delta
    return (x, y, jnp.fabs(delta), i + 1)


@partial(custom_jvp, nondiff_argnums=(2,))
def inverse(alpha, beta, y):
    return while_loop(cond_fn, partial(body_fn, alpha, beta), (y, y, 9.9e9, 0))[0]


@inverse.defjvp
@jit
def inverse_jvp(y, primals, tangents):
  alpha, beta = primals
  alpha_dot, beta_dot = tangents
  primal_out = inverse(alpha, beta, y)
  y_tilde = primal_out
  denominator = 1.0 + jnp.square(alpha * y_tilde + beta * jnp.square(y_tilde))
  dalpha = (2.0 / 3.0) * alpha * jnp.power(y_tilde, 3.0) + 0.5 * beta * jnp.power(y_tilde, 4.0)
  dbeta = 0.5 * alpha * jnp.power(y_tilde, 4.0) + 0.4 * beta * jnp.power(y_tilde, 5.0)
  numerator = dalpha * alpha_dot + dbeta * beta_dot
  tangent_out = -numerator / denominator
  return primal_out, tangent_out



@jit
def jacobian_and_inverse(alpha, beta, Y):
    Y_tilde = vmap(inverse)(jnp.broadcast_to(alpha, Y.shape), jnp.broadcast_to(beta, Y.shape), Y)
    Y_tilde_stop = stop_gradient(Y_tilde)
    log_det_jacobian = -jnp.sum(jnp.log(1.0 + jnp.square(alpha * Y_tilde_stop + beta * jnp.square(Y_tilde_stop))))
    return Y_tilde, log_det_jacobian


# tests
if __name__ == '__main__':
    Y = 3 * np.random.randn(2)
    alpha = 0.5 + 0.5 * np.random.rand(1)[0]
    beta = 0.5 + 0.5 * np.random.rand(1)[0]
    print("alpha, beta", alpha, beta)
    Y_tilde, jac = jacobian_and_inverse(alpha, beta, Y)
    ffY = vmap(partial(forward, alpha, beta))(Y_tilde)
    delta = np.fabs(Y - ffY)
    print("Ytilde", Y_tilde, "Y", Y, "ffY", ffY, "delta", delta)

    def Yt(alpha, beta):
        return jacobian_and_inverse(alpha, beta, Y)[0][0]

    g = grad(Yt, argnums=0)(alpha, beta)
    print("dYtilde/dalpha", g)
    g = grad(Yt, argnums=1)(alpha, beta)
    print("dYtilde/dbeta", g)

    def Yt(alpha, beta):
        return jacobian_and_inverse(alpha, beta, Y)[1]

    g = grad(Yt, argnums=0)(alpha, beta)
    print("dlogjac/dalpha", g)
    g = grad(Yt, argnums=1)(alpha, beta)
    print("dlogjac/dbeta", g)

    def YYt(alpha, beta):
        return jnp.sum(forward(alpha, beta, jacobian_and_inverse(alpha, beta, Y)[0]))

    g = grad(YYt, argnums=0)(alpha, beta)
    print("didentity/dalpha", g)
    g = grad(YYt, argnums=1)(alpha, beta)
    print("didentity/dbeta", g)

    #def YYt(alpha, beta):
    #    return jnp.sum(jacobian_and_inverse(alpha, beta, forward(alpha, beta, Y))[0])

    #g = grad(YYt, argnums=0)(alpha, beta)
    #print("didentity/dalpha", g)
    #g = grad(YYt, argnums=1)(alpha, beta)
    #print("didentity/dbeta", g)
