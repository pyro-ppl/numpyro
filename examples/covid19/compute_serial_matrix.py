import numpy as np


def compute_serial_matrix(serial, tau):
    SI_CUT = serial.shape[0]
    serial_matrix = np.zeros((tau, SI_CUT))

    def convolve(history):
        output = np.zeros(tau)
        for t in range(tau):
            output[t] = (serial * history).sum()
            history = np.concatenate([history[1:], output[t:t+1]])
        return output

    for t in range(SI_CUT):
        history = np.zeros(SI_CUT)
        history[t] = 1.0
        serial_matrix[:, t] = convolve(history)

    return serial_matrix


# test compute_serial_matrix
if __name__ == "__main__":
    tau = 3

    for SI_CUT in [3, 4, 5, 9]:
        serial = np.random.rand(SI_CUT)

        m = compute_serial_matrix(serial, tau)

        expected = np.zeros((tau, SI_CUT))
        expected[0] = serial
        expected[1, 1:] = serial[:-1]
        expected[1, :] += serial * serial[-1]
        expected[2, 2:] = serial[:-2]
        expected[2, :] += serial * serial[-2]
        expected[2, :] += expected[1, :] * serial[-1]

        delta = np.max(np.fabs(expected - m))
        assert delta < 1.0e-10
