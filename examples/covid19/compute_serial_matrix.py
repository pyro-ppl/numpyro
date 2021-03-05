import numpyro

import numpy as np
import jax
from jax import ops
import jax.numpy as jnp



def compute_serial_matrix(serial, tau):
    SI_CUT = serial.shape[0]
    serial_matrix = jnp.zeros((tau, SI_CUT))

    def convolve(history):
        output = jnp.zeros(tau)
        for t in range(tau):
            output_t = (serial * history).sum()
            output = ops.index_update(output, t, output_t)
            history = jnp.concatenate([history[1:], output[t:t+1]])
        return output

    for t in range(SI_CUT):
        history = jnp.zeros(SI_CUT)
        history = ops.index_update(history, t, 1.0)
        serial_matrix = ops.index_update(serial_matrix, ops.index[:, t], convolve(history))

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

    #for tau in [2, 3, 4]:
    #    for SI_CUT in [3, 6, 7, 12]:
    for tau in [5]:
        for SI_CUT in [10]:
            serial = jnp.array(np.random.rand(SI_CUT))

            m = compute_serial_matrix(serial, tau)
            m_rho = compute_serial_matrix_rho(serial, tau).sum(0)

            delta = np.max(np.fabs(m - m_rho))
            print(delta, tau, SI_CUT)
            assert delta < 1.0e-13

            if tau == 3:
                expected = np.zeros((tau, SI_CUT))
                expected[0] = serial
                expected[1, 1:] = serial[:-1]
                expected[1, :] += serial * serial[-1]
                expected[2, 2:] = serial[:-2]
                expected[2, :] += serial * serial[-2]
                expected[2, :] += expected[1, :] * serial[-1]

                delta = np.max(np.fabs(expected - m))
                assert delta < 1.0e-6
