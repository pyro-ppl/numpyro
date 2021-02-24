import numpy as np

import jax.numpy as jnp
import jax.ops as ops

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform


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
    E_casesByAge = jnp.zeros((N2, A))

    # init expected cases by age and location in first N0 days
    E_casesByAge = ops.index_update(E_casesByAge, ops.index[:N0, init_A], e_cases_N0_local / N_init_A,
                                    indices_are_sorted=True, unique_indices=True)

    # calculate expected cases by age and country under self-renewal model after first N0 days
    # and adjusted for saturation
    t = jnp.arange(N0, N2)
    start_idx_rev_serial = jnp.clip(SI_CUT - t, a_min=0)
    start_idx_E_casesByAge = jnp.clip(t - SI_CUT, a_min=0)

    tmp_row_vector_A = jnp.where(start_idx_rev_serial <= t & t < SI_CUT )

    # TODO: compute the remaining E_casesByAge, then using stick breaking transform
    prop_susceptibleByAge = jnp.zeros(1.0, A)
    prop_susceptibleByAge = jnp.clip(prop_susceptibleByAge, a_min=0.)
    return E_casesByAge


N0 = 3
N2 = 6
A = 5
A_CHILD = 2
SI_CUT = 3
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

print("value", value)
