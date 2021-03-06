from functools import partial

import numpyro

import numpy as np
import jax
from jax import ops
import jax.numpy as jnp
from jax.lax import scan



def convolve(history, serial, tau):
    def scan_body(carry, x):
        output, h, t = carry
        output_t = (serial * h).sum()
        output = ops.index_update(output, t, output_t)
        h = jnp.concatenate([h[1:], output_t[None]])
        return (output, h, t + 1), None

    output = scan(scan_body, (jnp.zeros(tau), history, 0), None, length=tau)[0][0]

    return output

#def _convolve(history, serial, tau):
#    output = jnp.zeros(tau)
#    for t in range(tau):
#        output_t = (serial * history).sum()
#        output = ops.index_update(output, t, output_t)
#        history = jnp.concatenate([history[1:], output[t:t+1]])
#    return output


@partial(jax.jit, static_argnums=(1,))
def compute_serial_matrix(serial, tau):
    SI_CUT = serial.shape[0]
    serial_matrix = jnp.zeros((tau, SI_CUT))

    for t in range(SI_CUT):
        history = jnp.zeros(SI_CUT)
        history = ops.index_update(history, t, 1.0)
        serial_matrix = ops.index_update(serial_matrix, ops.index[:, t], convolve(history, serial, tau))

    return serial_matrix


def compute_serial_matrix_rho(serial, tau):
    SI_CUT = serial.shape[0]
    rho = jnp.array(0.0)
    serial_matrix_rho = jnp.zeros((tau, tau, SI_CUT))

    f = lambda rho: compute_serial_matrix(rho * serial, tau)

    gs = [jax.jacfwd(f)]
    Z = 1

    for t in range(tau):
        Z *= (t + 1)
        g_rho = gs[t](rho) / Z
        serial_matrix_rho = ops.index_update(serial_matrix_rho, ops.index[t, :, :], g_rho)
        gs.append(jax.jacfwd(gs[t]))

    return serial_matrix_rho


# test compute_serial_matrix
if __name__ == "__main__":
    numpyro.enable_x64()

    for tau in [2, 3]:
        for SI_CUT in [4, 5]:
            rho = np.random.rand(1).item()
            serial = np.random.RandomState(tau + 10 * SI_CUT).rand(SI_CUT)
            serial /= serial.sum()

            m = compute_serial_matrix(rho * serial, tau)
            m_rho = np.array(compute_serial_matrix_rho(serial, tau))
            rho_powers = np.power(rho, np.arange(tau) + 1)
            m_rho = rho_powers[:, None, None] * m_rho
            m_rho = m_rho.sum(0)

            delta = np.max(np.fabs(m - m_rho))
            print("tau", tau, "SI_CUT", SI_CUT, "delta", delta)
            assert delta < 1.0e-13

            if tau == 3:
                serial *= rho
                expected = np.zeros((tau, SI_CUT))
                expected[0] = serial
                expected[1, 1:] = serial[:-1]
                expected[1, :] += serial * serial[-1]
                expected[2, 2:] = serial[:-2]
                expected[2, :] += serial * serial[-2]
                expected[2, :] += expected[1, :] * serial[-1]

                delta = np.max(np.fabs(expected - m))
                assert delta < 1.0e-13
