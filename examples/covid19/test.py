import numpy as np

import jax.numpy as jnp
import jax.ops as ops

import numpyro


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
    # wkend_idx_local: jnp.int32,  # 1D
    avg_cntct_local: float,
    # cntct_weekends_mean_local: jnp.float32,  # A x A
    # cntct_weekdays_mean_local: jnp.float32,  # A x A
    cntct_mean_local: jnp.float32,  # A x A
    # cntct_school_closure_weekends_local: jnp.float32,  # A x A
    # cntct_school_closure_weekdays_local: jnp.float32,  # A x A
    cntct_school_closure_local: jnp.float32,  # A x A
    # cntct_elementary_school_reopening_weekends_local: jnp.float32,  # A x A
    # cntct_elementary_school_reopening_weekdays_local: jnp.float32,  # A x A
    cntct_elementary_school_reopening_local: jnp.float32,  # A x A
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

        tmp_row_vector_A = (rev_serial_interval[start_idx_rev_serial:SI_CUT][:, None] *
                            E_casesByAge[start_idx_E_casesByAge:t-1]).sum(0)
        tmp_row_vector_A *= rho0
        tmp_row_vector_A_no_impact_intv = tmp_row_vector_A.copy()
        tmp_row_vector_A *= impact_intv[t]

        col1 = (tmp_row_vector_A_no_impact_intv[:, None] * cntct_mean_local[:, :A_CHILD]).sum(0)
        col2 = (tmp_row_vector_A_no_impact_intv[:A_CHILD, None] * cntct_mean_local[:A_CHILD, A_CHILD:]).sum(0) +\
                (tmp_row_vector_A[A_CHILD:, None] * cntct_mean_local[A_CHILD:, A_CHILD:]).sum(0) * impact_intv[t, A_CHILD:]

        E_casesByAge[t] = np.concatenate([col1, col2])
        E_casesByAge[t] *= prop_susceptibleByAge
        E_casesByAge[t] *= np.exp(log_relsusceptibility_age)

    return E_casesByAge


N0 = 3
N2 = 9
A = 6
A_CHILD = 2
SI_CUT = 5
N_init_A = 2
init_A = [2, 4]

R0_local = 0.53
e_cases_N0_local = 0.67
log_relsusceptibility_age = np.random.randn(A)
impact_intv_children_effect = 0.42
impact_intv_onlychildren_effect = 0.38
impact_intv = np.random.rand(N2, A)
elementary_school_reopening_idx_local = 2
SCHOOL_STATUS_local = np.array([0, 1] * (N2 // 2))
avg_cntct_local = 1.42
cntct_mean_local = np.random.rand(A, A)
cntct_school_closure_local = np.random.rand(A, A)
cntct_elementary_school_reopening_local = np.random.rand(A, A)
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
    avg_cntct_local,
    cntct_mean_local,
    cntct_school_closure_local,
    cntct_elementary_school_reopening_local,
    rev_serial_interval,
    popByAge_abs_local,
    N_init_A,
    init_A)

print("country_EcasesByAge_direct() =", value.shape)
