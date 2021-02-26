import numpy as np

from jax.lax import scan, dynamic_slice_in_dim, cond
import jax.numpy as jnp
import jax.ops as ops

import numpyro
from numpyro.util import identity


# direct implementation of country_EcasesByAge in NumPy with for loops
def country_EcasesByAge_direct(
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
    # TODO: convert this to a boolean vector of length N2 - N0:
    #   [t in wken_idx_local for t in [N0, N2)]
    wkend_idx_local: jnp.int32,  # 1D
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
        start_idx_rev_serial = SI_CUT - t + 1;
        start_idx_E_casesByAge = t - SI_CUT - 1;

        prop_susceptibleByAge = 1.0 - E_casesByAge[:t-1].sum(0) / popByAge_abs_local

        start_idx_rev_serial = max(0, start_idx_rev_serial)
        start_idx_E_casesByAge = max(0, start_idx_E_casesByAge)
        prop_susceptibleByAge = np.maximum(0.0, prop_susceptibleByAge)

        print("start_idx_rev_serial:SI_CUT", start_idx_rev_serial, SI_CUT,
                "    start_idx_E_casesByAge:t-1", start_idx_E_casesByAge,t-1)

        tmp_row_vector_A = (rev_serial_interval[start_idx_rev_serial:SI_CUT][:, None] *
                            E_casesByAge[start_idx_E_casesByAge:t-1]).sum(0)
        tmp_row_vector_A *= rho0
        tmp_row_vector_A_no_impact_intv = tmp_row_vector_A.copy()

        weekend = wkend_idx_local[t - N0]  # this is a boolean
        cntct_mean_local = cntct_weekends_mean_local if weekend else cntct_weekdays_mean_local
        cntct_elementary_school_reopening_local = cntct_elementary_school_reopening_weekends_local if weekend \
                                             else cntct_elementary_school_reopening_weekdays_local
        cntct_school_closure_local = cntct_school_closure_weekends_local if weekend else cntct_school_closure_weekdays_local

        if SCHOOL_STATUS_local[t] == 0 and t < elementary_school_reopening_idx_local:  # school open
            col1 = (tmp_row_vector_A_no_impact_intv[:, None] * cntct_mean_local[:, :A_CHILD]).sum(0)
            col2 = (tmp_row_vector_A_no_impact_intv[:A_CHILD, None] * cntct_mean_local[:A_CHILD, A_CHILD:]).sum(0) +\
                    (tmp_row_vector_A[A_CHILD:, None] * cntct_mean_local[A_CHILD:, A_CHILD:]).sum(0) * impact_intv[t, A_CHILD:]
            E_casesByAge[t] = np.concatenate([col1, col2])
        elif SCHOOL_STATUS_local[t] == 0 and t >= elementary_school_reopening_idx_local:  # school reopen
            tmp_row_vector_A_with_children_impact_intv = tmp_row_vector_A.copy()
            tmp_row_vector_A_with_children_impact_intv[:A_CHILD] *= impact_intv_children_effect
            tmp_row_vector_A_with_children_and_childrenchildren_impact_intv = tmp_row_vector_A_with_children_impact_intv.copy()
            tmp_row_vector_A_with_children_and_childrenchildren_impact_intv[:A_CHILD] *= impact_intv_onlychildren_effect

            col1 = (tmp_row_vector_A_with_children_and_childrenchildren_impact_intv[:, None] * \
                    cntct_elementary_school_reopening_local[:, :A_CHILD]).sum(0)
            col2 = (tmp_row_vector_A_with_children_impact_intv[:, None] * \
                    cntct_elementary_school_reopening_local[:, A_CHILD:]).sum(0)
            E_casesByAge[t] = np.concatenate([col1, col2])

            E_casesByAge[t, :A_CHILD] *= impact_intv_children_effect
            E_casesByAge[t, A_CHILD:] *= impact_intv[t, A_CHILD:]
        else:  # school closed
            col1 = (tmp_row_vector_A_no_impact_intv[:, None] * cntct_school_closure_local[:, :A_CHILD]).sum(0)
            col2 = (tmp_row_vector_A_no_impact_intv[:A_CHILD, None] * cntct_school_closure_local[:A_CHILD, A_CHILD:]).sum(0) +\
                    (tmp_row_vector_A[A_CHILD:, None] * cntct_school_closure_local[A_CHILD:, A_CHILD:]).sum(0) * \
                     impact_intv[t, A_CHILD:]
            E_casesByAge[t] = np.concatenate([col1, col2])

        E_casesByAge[t] *= prop_susceptibleByAge
        E_casesByAge[t] *= np.exp(log_relsusceptibility_age)

    return E_casesByAge


# scan implementation of country_EcasesByAge in JAX
def country_EcasesByAge_scan(
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
    # TODO: convert this to a boolean vector of length N2 - N0:
    #   [t in wken_idx_local for t in [N0, N2)]
    wkend_idx_local: jnp.int32,  # 1D
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
    E_casesByAge = jnp.zeros((N2, A))

    # init expected cases by age and location in first N0 days
    E_casesByAge = ops.index_update(E_casesByAge, ops.index[:N0, init_A], e_cases_N0_local / N_init_A,
                                    indices_are_sorted=True, unique_indices=True)

    # pad with zeros on left
    E_casesByAge_SI_CUT = jnp.concatenate([jnp.zeros((SI_CUT - N0 + 1, A)), E_casesByAge[:N0 - 1]])

    def scan_body(carry, x):
        weekend_t, SCHOOL_STATUS_t = x
        E_casesByAge, E_casesByAge_sum, E_casesByAge_SI_CUT, t = carry

        prop_susceptibleByAge = 1.0 - E_casesByAge_sum / popByAge_abs_local
        prop_susceptibleByAge = jnp.maximum(0.0, prop_susceptibleByAge)

        tmp_row_vector_A = rev_serial_interval @ E_casesByAge_SI_CUT
        tmp_row_vector_A *= rho0
        tmp_row_vector_A_no_impact_intv = tmp_row_vector_A

        cntct_mean_local, cntct_elementary_school_reopening_local, cntct_school_closure_local = cond(weekend_t,
            (cntct_weekends_mean_local, cntct_elementary_school_reopening_weekends_local, cntct_school_closure_weekends_local), identity,
            (cntct_weekdays_mean_local, cntct_elementary_school_reopening_weekdays_local, cntct_school_closure_weekdays_local), identity)

        E_casesByAge_t = jnp.ones(A)

        # update current time slice of E_casesByAge
        E_casesByAge = ops.index_update(E_casesByAge, t,
                                        E_casesByAge_t * prop_susceptibleByAge * np.exp(log_relsusceptibility_age),
                                        indices_are_sorted=True, unique_indices=True)

        # update carry variables
        E_casesByAge_sum = E_casesByAge_sum + E_casesByAge[t].sum(0)
        # basically "roll left and append newest slice at right"
        E_casesByAge_SI_CUT = jnp.concatenate([E_casesByAge_SI_CUT[1:], E_casesByAge[t][None, :]])

        return (E_casesByAge, E_casesByAge_sum, E_casesByAge_SI_CUT, t + 1), None

    print("E_casesByAge before\n ", E_casesByAge)

    E_casesByAge_sum = E_casesByAge[:N0-1].sum(0)
    E_casesByAge_SI_CUT = jnp.concatenate([jnp.zeros((SI_CUT - N0 + 1, A)), E_casesByAge[:N0 - 1]])

    init = (E_casesByAge, E_casesByAge_sum, E_casesByAge_SI_CUT, N0)
    xs = (wkend_idx_local, SCHOOL_STATUS_local[N0:])
    E_casesByAge = scan(scan_body, init, xs, length=N2 - N0)[0][0]

    print("E_casesByAge after\n ", E_casesByAge)

    return E_casesByAge



N0 = 4
N2 = 10
A = 6
A_CHILD = 2
SI_CUT = 5
N_init_A = 2
init_A = [2, 4]

R0_local = np.random.rand(1).item()
e_cases_N0_local = np.random.rand(1).item()
log_relsusceptibility_age = np.random.randn(A)
impact_intv_children_effect = np.random.rand(1).item()
impact_intv_onlychildren_effect = np.random.rand(1).item()
impact_intv = np.random.rand(N2, A)
elementary_school_reopening_idx_local = 2
SCHOOL_STATUS_local = np.array([0, 1] * (N2 // 2))
avg_cntct_local = np.random.rand(1).item()
wkend_idx_local = np.array([True, False] * ((N2 - N0) // 2))
cntct_weekends_mean_local = np.random.rand(A, A)
cntct_weekdays_mean_local = np.random.rand(A, A)
cntct_school_closure_weekends_local = np.random.rand(A, A)
cntct_school_closure_weekdays_local = np.random.rand(A, A)
cntct_elementary_school_reopening_weekends_local = np.random.rand(A, A)
cntct_elementary_school_reopening_weekdays_local = np.random.rand(A, A)
rev_serial_interval = np.random.rand(SI_CUT)
popByAge_abs_local = 1234.5 * np.random.rand(A)


value = country_EcasesByAge_direct(
    R0_local,
    e_cases_N0_local,
    log_relsusceptibility_age,
    impact_intv_children_effect,
    impact_intv_onlychildren_effect,
    impact_intv,
    N0,
    elementary_school_reopening_idx_local,
    N2,
    SCHOOL_STATUS_local,
    A,
    A_CHILD,
    SI_CUT,
    wkend_idx_local,
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

print("country_EcasesByAge_direct() =", value.shape)
assert value.shape == (N2, A)

value = country_EcasesByAge_scan(
    R0_local,
    e_cases_N0_local,
    log_relsusceptibility_age,
    impact_intv_children_effect,
    impact_intv_onlychildren_effect,
    impact_intv,
    N0,
    elementary_school_reopening_idx_local,
    N2,
    SCHOOL_STATUS_local,
    A,
    A_CHILD,
    SI_CUT,
    wkend_idx_local,
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
