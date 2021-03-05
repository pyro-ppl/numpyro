import numpy as np

import jax.numpy as jnp

import numpyro

from functions import country_EcasesByAge

import jax
import jax.ops as ops

from jax.lax import scan, cond, switch
from numpyro.util import identity

from compute_serial_matrix import compute_serial_matrix


# direct implementation of country_EcasesByAge in NumPy with for loops
def country_EcasesByAge_direct(
    tau,
    # parameters
    R0_local: float,
    e_cases_N0_local: float,
    log_relsusceptibility_age: jnp.float32,  # A
    impact_intv_children_effect: float,
    impact_intv_onlychildren_effect: float,
    impact_intv: jnp.float32,  # N2 x A
    # data
    N0: int,
    elementary_school_reopening_idx_local: int,
    N2: int,
    SCHOOL_STATUS_local: jnp.float32,  # N2
    A: int,
    A_CHILD: int,
    SI_CUT: int,
    wkend_idx_local,  # boolean array
    avg_cntct_local: float,
    cntct_weekends_mean_local: jnp.float32,  # A x A
    cntct_weekdays_mean_local: jnp.float32,  # A x A
    cntct_school_closure_weekends_local: jnp.float32,  # A x A
    cntct_school_closure_weekdays_local: jnp.float32,  # A x A
    cntct_elementary_school_reopening_weekends_local: jnp.float32,  # A x A
    cntct_elementary_school_reopening_weekdays_local: jnp.float32,  # A x A
    rev_serial_interval: jnp.float32,  # SI_CUT
    popByAge_abs_local: jnp.float32,  # A
    N_init_A: int,
    init_A: jnp.int32,  # N_init_A
) -> jnp.float32:  # N2 x A

    # probability of infection given contact in location m
    rho0 = R0_local / avg_cntct_local

    # expected new cases by calendar day, age, and location under self-renewal model
    # and a container to store the precomputed cases by age
    E_casesByAge = np.zeros((N2, A))

    # init expected cases by age and location in first N0 days
    E_casesByAge[:N0, init_A] = e_cases_N0_local / N_init_A

    for t in range(N0, N2):
        start_idx_rev_serial = max(0, SI_CUT - t)
        start_idx_E_casesByAge = max(0, t - SI_CUT)

        #t_round = t - (t - N0) % tau
        #print("t, t_round, tau", t, t_round, tau)
        #prop_susceptibleByAge = 1.0 - E_casesByAge[:t_round].sum(0) / popByAge_abs_local
        prop_susceptibleByAge = 1.0
        #prop_susceptibleByAge = np.maximum(0.0, prop_susceptibleByAge)
        #print(t, prop_susceptibleByAge)
        #print("E_casesByAge[:{}].sum(0)".format(t_round),E_casesByAge[:t_round].sum(0))

        tmp_row_vector_A = (rev_serial_interval[start_idx_rev_serial:SI_CUT][:, None]
                            * E_casesByAge[start_idx_E_casesByAge:t]).sum(0)
        #print("start_idx_rev_serial:SI_CUT", start_idx_rev_serial,SI_CUT, "start_idx_E_casesByAge:t", start_idx_E_casesByAge,t)
        #tmp_row_vector_A *= rho0
        #tmp_row_vector_A_no_impact_intv = tmp_row_vector_A.copy()
        #tmp_row_vector_A *= impact_intv[t - N0]

        """
        # choose weekend/weekday contact matrices
        weekend = wkend_idx_local[t - N0]  # this is a boolean
        cntct_mean_local = cntct_weekends_mean_local if weekend else cntct_weekdays_mean_local
        cntct_elementary_school_reopening_local = cntct_elementary_school_reopening_weekends_local if weekend \
            else cntct_elementary_school_reopening_weekdays_local
        cntct_school_closure_local = cntct_school_closure_weekends_local if weekend \
            else cntct_school_closure_weekdays_local

        if SCHOOL_STATUS_local[t - N0] == 0.0 and t < elementary_school_reopening_idx_local:  # school open
            col1 = (tmp_row_vector_A_no_impact_intv[:, None] * cntct_mean_local[:, :A_CHILD]).sum(0)
            col2 = (tmp_row_vector_A_no_impact_intv[:A_CHILD, None] * cntct_mean_local[:A_CHILD, A_CHILD:]).sum(0) +\
                (tmp_row_vector_A[A_CHILD:, None] *
                 cntct_mean_local[A_CHILD:, A_CHILD:]).sum(0) * impact_intv[t - N0, A_CHILD:]
            E_casesByAge[t] = np.concatenate([col1, col2])
        elif SCHOOL_STATUS_local[t - N0] == 0.0 and t >= elementary_school_reopening_idx_local:  # school reopen
            tmp_row_vector_A_with_children_impact_intv = tmp_row_vector_A.copy()
            tmp_row_vector_A_with_children_impact_intv[:A_CHILD] *= impact_intv_children_effect
            tmp_row_vector_A_with_children_and_childrenchildren_impact_intv = \
                tmp_row_vector_A_with_children_impact_intv.copy()
            tmp_row_vector_A_with_children_and_childrenchildren_impact_intv[:A_CHILD] *= impact_intv_onlychildren_effect

            col1 = (tmp_row_vector_A_with_children_and_childrenchildren_impact_intv[:, None] *
                    cntct_elementary_school_reopening_local[:, :A_CHILD]).sum(0)
            col2 = (tmp_row_vector_A_with_children_impact_intv[:, None] *
                    cntct_elementary_school_reopening_local[:, A_CHILD:]).sum(0)
            E_casesByAge[t] = np.concatenate([col1, col2])

            E_casesByAge[t, :A_CHILD] *= impact_intv_children_effect
            E_casesByAge[t, A_CHILD:] *= impact_intv[t - N0, A_CHILD:]
        else:  # school closed
            col1 = (tmp_row_vector_A_no_impact_intv[:, None] * cntct_school_closure_local[:, :A_CHILD]).sum(0)
            col2 = (tmp_row_vector_A_no_impact_intv[:A_CHILD, None] *
                    cntct_school_closure_local[:A_CHILD, A_CHILD:]).sum(0) +\
                (tmp_row_vector_A[A_CHILD:, None] * cntct_school_closure_local[A_CHILD:, A_CHILD:]).sum(0) * \
                impact_intv[t - N0, A_CHILD:]
            E_casesByAge[t] = np.concatenate([col1, col2])

        """
        E_casesByAge[t] = tmp_row_vector_A
        #E_casesByAge[t] *= prop_susceptibleByAge
        #E_casesByAge[t] *= np.exp(log_relsusceptibility_age)

    return E_casesByAge


def country_EcasesByAge(
    tau,
    # parameters
    R0_local: float,
    e_cases_N0_local: float,
    log_relsusceptibility_age: np.float64,  # A
    impact_intv_children_effect: float,
    impact_intv_onlychildren_effect: float,
    impact_intv: np.float64,  # N2 x A
    # data
    N0: int,
    N2: int,
    A: int,
    A_CHILD: int,
    SI_CUT: int,
    wkend_mask_local: np.bool,  # 1D
    avg_cntct_local: float,
    cntct_weekends_mean_local: np.float64,  # A x A
    cntct_weekdays_mean_local: np.float64,  # A x A
    cntct_school_closure_weekends_local: np.float64,  # A x A
    cntct_school_closure_weekdays_local: np.float64,  # A x A
    cntct_elementary_school_reopening_weekends_local: np.float64,  # A x A
    cntct_elementary_school_reopening_weekdays_local: np.float64,  # A x A
    rev_serial_interval_matrix: np.float64,  # tau SI_CUT
    popByAge_abs_local: np.float64,  # A
    N_init_A: int,
    init_A: np.int64,  # N_init_A
    school_switch: np.int64,  # N2 - N0
) -> np.float64:  # N2 x A

    # probability of infection given contact in location m
    rho0 = R0_local / avg_cntct_local
    #rev_serial_interval_matrix *= rho0

    relsusceptibility_age = jnp.exp(log_relsusceptibility_age)

    """
    impact_intv_children_effect_padded = jnp.concatenate([impact_intv_children_effect * jnp.ones(A_CHILD),
                                                          jnp.ones(A - A_CHILD)])
    impact_intv_onlychildren_effect_padded = jnp.concatenate([impact_intv_onlychildren_effect * jnp.ones(A_CHILD),
                                                              jnp.ones(A - A_CHILD)])
    impact_intv_children_onlychildren_padded = impact_intv_children_effect_padded * impact_intv_onlychildren_effect_padded
    """

    cntct_mean = jnp.stack([cntct_weekdays_mean_local, cntct_weekends_mean_local])
    cntct_school_closure = jnp.stack([cntct_school_closure_weekdays_local, cntct_school_closure_weekends_local])
    cntct_elementary_school_reopening = jnp.stack([cntct_elementary_school_reopening_weekdays_local,
                                                   cntct_elementary_school_reopening_weekends_local])

    # define body of main for loop
    def scan_body(carry, x):
        impact_intv_t, weekend_t, school_switch_t = x
        E_casesByAge_sum, E_casesByAge_SI_CUT = carry

        assert E_casesByAge_sum.shape == (A,)
        assert E_casesByAge_SI_CUT.shape == (SI_CUT, A)

        #prop_susceptibleByAge = 1.0 - E_casesByAge_sum / popByAge_abs_local
        #prop_susceptibleByAge = jnp.maximum(0.0, prop_susceptibleByAge)
        prop_susceptibleByAge = jnp.ones(A)
        assert prop_susceptibleByAge.shape == (A,)

        # this convolution is effectively zero padded on the left
        tmp_row_vector_A_no_impact_intv = rev_serial_interval_matrix @ E_casesByAge_SI_CUT  # tau A
        assert tmp_row_vector_A_no_impact_intv.shape == (tau, A)
        tmp_row_vector_A = impact_intv_t * tmp_row_vector_A_no_impact_intv

        # choose weekend/weekday contact matrices
        assert weekend_t.shape == (tau,)

        cntct_mean_t = jnp.take_along_axis(cntct_mean, weekend_t[:, None, None], 0)  # tau A A
        cntct_school_closure_t = jnp.take_along_axis(cntct_school_closure, weekend_t[:, None, None], 0)
        cntct_elementary_school_reopening_t = jnp.take_along_axis(cntct_elementary_school_reopening,
                                                                  weekend_t[:, None, None], 0)

        """
        def school_open(dummy):
            return tmp_row_vector_A_no_impact_intv[:, None], cntct_mean_local[:, :A_CHILD],\
                   tmp_row_vector_A_no_impact_intv[:A_CHILD, None], cntct_mean_local[:A_CHILD, A_CHILD:],\
                   tmp_row_vector_A[A_CHILD:, None], cntct_mean_local[A_CHILD:, A_CHILD:],\
                   impact_intv_t[A_CHILD:]

        def school_reopen(dummy):
            return ((tmp_row_vector_A * impact_intv_children_onlychildren_padded)[:, None],
                    cntct_elementary_school_reopening_local[:, :A_CHILD] * impact_intv_children_effect,
                    tmp_row_vector_A[:A_CHILD, None] * impact_intv_children_effect,
                    cntct_elementary_school_reopening_local[:A_CHILD, A_CHILD:] * impact_intv_t[A_CHILD:],
                    tmp_row_vector_A[A_CHILD:, None],
                    cntct_elementary_school_reopening_local[A_CHILD:, A_CHILD:] * impact_intv_t[A_CHILD:],
                    jnp.ones(A - A_CHILD))

        def school_closed(dummy):
            return tmp_row_vector_A_no_impact_intv[:, None], cntct_school_closure_local[:, :A_CHILD],\
                   tmp_row_vector_A_no_impact_intv[:A_CHILD, None], cntct_school_closure_local[:A_CHILD, A_CHILD:],\
                   tmp_row_vector_A[A_CHILD:, None], cntct_school_closure_local[A_CHILD:, A_CHILD:],\
                   impact_intv_t[A_CHILD:]
        """

        # school_switch_t controls which of the three school branches we should follow in this iteration
        # 0 => school_open     1 => school_reopen     2 => school_closed
        """
        contact_inputs = switch(school_switch_t, [school_open, school_reopen, school_closed], None)
        (col1_left, col1_right, col2_topleft, col2_topright,
         col2_bottomleft, col2_bottomright, impact_intv_adult) = contact_inputs

        col1 = (col1_left.T @ col1_right)[0]
        col2 = (col2_topleft.T @ col2_topright)[0] + (col2_bottomleft.T @ col2_bottomright)[0] * impact_intv_adult
        E_casesByAge_t = jnp.concatenate([col1, col2])

        E_casesByAge_t *= prop_susceptibleByAge * relsusceptibility_age
        """
        E_casesByAge_t = tmp_row_vector_A_no_impact_intv
        #E_casesByAge_t *= prop_susceptibleByAge * relsusceptibility_age

        # add term to cumulative sum
        E_casesByAge_sum = E_casesByAge_sum + E_casesByAge_t.sum(0)
        # basically "roll left and append most recent time slice at right"
        E_casesByAge_SI_CUT = jnp.concatenate([E_casesByAge_SI_CUT[tau:], E_casesByAge_t])

        return (E_casesByAge_sum, E_casesByAge_SI_CUT), (E_casesByAge_sum, E_casesByAge_t)
        #return (E_casesByAge_sum, E_casesByAge_SI_CUT), E_casesByAge_t

    # init expected cases by age and location in first N0 days
    E_casesByAge_init = ops.index_update(jnp.zeros((N0, A)), ops.index[:N0, init_A], e_cases_N0_local / N_init_A,
                                         indices_are_sorted=True, unique_indices=True)

    # initialize carry variables
    E_casesByAge_sum = E_casesByAge_init.sum(0)
    # pad with zeros on left
    E_casesByAge_SI_CUT = jnp.concatenate([jnp.zeros((SI_CUT - N0, A)), E_casesByAge_init])

    init = (E_casesByAge_sum, E_casesByAge_SI_CUT)
    print("init E_casesByAge_sum",E_casesByAge_sum)
    #print("init E_casesByAge_sum",E_casesByAge_sum.shape, "E_casesByAge_SI_CUT",E_casesByAge_SI_CUT.shape)
    xs = (impact_intv, wkend_mask_local, school_switch)
    #for k, x in enumerate(xs):
    #    print("k", k, "x.shape", x.shape)

    # execute for loop using JAX primitive scan
    E_sum, E_t = scan(scan_body, init, xs, length=xs[0].shape[0])[1]
    print("E_sum\n",E_sum)
    print("E_t\n",E_t)
    import sys; sys.exit()
    E_casesByAge_N0_N2 = E_casesByAge_N0_N2.reshape((xs[0].shape[0] * tau, A))

    E_casesByAge = jnp.concatenate([E_casesByAge_init, E_casesByAge_N0_N2])

    return E_casesByAge




if __name__ == '__main__':
    numpyro.enable_x64()

    NUM_TESTS = 1

    A = 3
    A_CHILD = 2
    N_init_A = 2
    init_A = [1, 2]

    for test in range(NUM_TESTS):
        N0 = 4 + 3 * test
        N2 = 20 + test
        SI_CUT = 9 + test

        R0_local = np.random.rand(1).item()
        e_cases_N0_local = np.random.rand(1).item()
        log_relsusceptibility_age = np.random.randn(A)
        impact_intv_children_effect = np.random.rand(1).item()
        impact_intv_onlychildren_effect = np.random.rand(1).item()
        impact_intv = np.random.rand(N2 - N0, A)
        elementary_school_reopening_idx_local = np.random.randint(N0 + 2, N0 + 6)
        SCHOOL_STATUS_local = np.array(np.random.randint(0, 2, N2 - N0), dtype=np.float32)
        avg_cntct_local = np.random.rand(1).item()
        wkend_idx_local = np.array(np.random.randint(0, 2, N2 - N0), dtype=np.bool)
        cntct_weekends_mean_local = np.random.rand(A, A)
        cntct_weekdays_mean_local = np.random.rand(A, A)
        cntct_school_closure_weekends_local = np.random.rand(A, A)
        cntct_school_closure_weekdays_local = np.random.rand(A, A)
        cntct_elementary_school_reopening_weekends_local = np.random.rand(A, A)
        cntct_elementary_school_reopening_weekdays_local = np.random.rand(A, A)
        rev_serial_interval = np.random.rand(SI_CUT)
        rev_serial_interval /= rev_serial_interval.sum()
        popByAge_abs_local = 123.4 * np.random.rand(A)

        # which = 1
        # if which == 0:
        #    SCHOOL_STATUS_local = 0 * np.ones(N2)
        #    elementary_school_reopening_idx_local = 9999
        # elif which == 1:
        #    SCHOOL_STATUS_local = 0 * np.ones(N2)
        #    elementary_school_reopening_idx_local = 0
        # elif which == 2:
        #    SCHOOL_STATUS_local = np.ones(N2)

        school_switch = 2 * SCHOOL_STATUS_local.astype(np.int32) + \
            (np.arange(N0, N2) >= elementary_school_reopening_idx_local).astype(np.int32)
        print("school_switch",school_switch.shape)
        tau = 3
        rev_serial_interval_matrix = jnp.array(compute_serial_matrix(rev_serial_interval, tau))

        def reshape_and_pad(x, reshape=False):
            T = x.shape[0]
            pad = T % tau > 0
            pad = tau - (T - tau * (T // tau)) if pad else 0
            pad = np.zeros((pad,) + x.shape[1:])
            x = np.concatenate([x, pad], axis=0)
            if not reshape:
                return x
            else:
                return x.reshape((x.shape[0] // tau, tau) + x.shape[1:])

        impact_intv_pad = reshape_and_pad(impact_intv, reshape=False)
        SCHOOL_STATUS_local_pad = reshape_and_pad(SCHOOL_STATUS_local, reshape=False)
        wkend_idx_local_pad = reshape_and_pad(wkend_idx_local, reshape=False)
        N2_pad = N0 + impact_intv_pad.shape[0]

        print("N2, N2_pad", N2, N2_pad, "A", A, "N0", N0)
        print("tau", tau)
        print("impact_intv_pad", impact_intv.shape, impact_intv_pad.shape)
        print("SCHOOL_STATUS_local_pad", SCHOOL_STATUS_local.shape, SCHOOL_STATUS_local_pad.shape)
        print("wkend_idx_local_pad", wkend_idx_local.shape, wkend_idx_local_pad.shape)

        value_direct = country_EcasesByAge_direct(tau,
            R0_local,
            e_cases_N0_local,
            log_relsusceptibility_age,
            impact_intv_children_effect,
            impact_intv_onlychildren_effect,
            impact_intv_pad,
            N0,
            elementary_school_reopening_idx_local,
            N2_pad,
            SCHOOL_STATUS_local_pad,
            A,
            A_CHILD,
            SI_CUT,
            wkend_idx_local_pad,
            avg_cntct_local,
            cntct_weekends_mean_local,
            cntct_weekdays_mean_local,
            cntct_school_closure_weekends_local,
            cntct_school_closure_weekdays_local,
            cntct_elementary_school_reopening_weekends_local,
            cntct_elementary_school_reopening_weekdays_local,
            rev_serial_interval,
            popByAge_abs_local,
            N_init_A,
            init_A)

        print("value_direct",value_direct, "\n")

        print("impact_intv, wkend_idx_local", impact_intv.shape, wkend_idx_local.shape)
        impact_intv_pad = reshape_and_pad(impact_intv, reshape=True)
        wkend_idx_local_pad = reshape_and_pad(wkend_idx_local, reshape=True)
        print("school_switch",school_switch.shape)
        school_switch_pad = reshape_and_pad(school_switch, reshape=True)
        print("school_switch_pad",school_switch_pad.shape)
        print("impact_intv, wkend_idx_local", impact_intv_pad.shape, wkend_idx_local_pad.shape)

        value_scan = country_EcasesByAge(
            tau,
            R0_local,
            e_cases_N0_local,
            log_relsusceptibility_age,
            impact_intv_children_effect,
            impact_intv_onlychildren_effect,
            jnp.array(impact_intv_pad),
            N0,
            N2,
            A,
            A_CHILD,
            SI_CUT,
            jnp.array(wkend_idx_local_pad),
            avg_cntct_local,
            cntct_weekends_mean_local,
            cntct_weekdays_mean_local,
            cntct_school_closure_weekends_local,
            cntct_school_closure_weekdays_local,
            cntct_elementary_school_reopening_weekends_local,
            cntct_elementary_school_reopening_weekdays_local,
            rev_serial_interval_matrix,
            popByAge_abs_local,
            N_init_A,
            init_A,
            school_switch_pad)

        delta = value_direct - value_scan
        max_delta = np.max(np.fabs(delta))

        print("[Test {}] Max delta: {:.2e}".format(test, max_delta))

        print("value_direct\n",value_direct)
        print("value_scan\n", value_scan)

        assert value_direct.shape == (N2_pad, A)
        assert value_scan.shape == (N2_pad, A)
        assert max_delta < 1.0e-11
