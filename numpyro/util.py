from __future__ import division


def dual_averaging(t0=10, kappa=0.75, gamma=0.05):
    # TODO: add docs
    def init_fn(prox_center=0.):
        x_t = 0.
        x_avg = 0.  # average of primal sequence
        g_avg = 0.  # average of dual sequence
        t = 0
        return (x_t, x_avg, g_avg, t, prox_center)

    def update_fn(g, state):
        x_t, x_avg, g_avg, t, prox_center = state
        t = t + 1
        # g_avg = (g_1 + ... + g_t) / t
        g_avg = (1 - 1 / (t + t0)) * g_avg + g / (t + t0)
        # According to formula (3.4) of [1], we have
        #     x_t = argmin{ g_avg . x + loc_t . |x - x0|^2 },
        # where loc_t := beta_t / t, beta_t := (gamma/2) * sqrt(t)
        x_t = prox_center - (t ** 0.5) / gamma * g_avg
        # weight for the new x_t
        weight_t = t ** (-kappa)
        x_avg = (1 - weight_t) * x_avg + weight_t * x_t
        return (x_t, x_avg, g_avg, t, prox_center)

    return init_fn, update_fn
