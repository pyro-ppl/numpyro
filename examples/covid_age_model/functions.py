import numpy as np

import jax
import jax.numpy as jnp
import jax.ops as ops

from jax.lax import scan, cond, switch
from numpyro.util import identity

import numpyro.distributions as dist


# returns multiplier on the rows of the contact matrix over time for one country
def country_impact(
    beta: np.float64,  # 2
    dip_rdeff_local: float,
    upswing_timeeff_reduced_local: np.float64,  # 1D
    # N2: int,  # num of days
    # A: int,  # num of ages
    # A_CHILD: int,  # num of children ages
    AGE_CHILD: np.int64,  # A_CHILD
    # COVARIATES_N: int,
    covariates_local: np.float64,  # 3 x N2 x A
    upswing_age_rdeff_local: np.float64,  # A
    upswing_timeeff_map_local: np.int64,  # N2
) -> np.float64:  # N2 x A
    # scaling of contacts after intervention effect on day t in location m

    # define multipliers for contacts in each location
    impact_intv = (beta[1] + upswing_age_rdeff_local) * covariates_local[2]

    # expand upswing time effect
    impact_intv *= upswing_timeeff_reduced_local[upswing_timeeff_map_local][:, None]

    # add other coeff*predictors
    impact_intv += covariates_local[0]
    impact_intv += (beta[0] + dip_rdeff_local) * covariates_local[1]

    impact_intv = jnp.exp(impact_intv)

    # impact_intv set to 1 for children
    impact_intv = ops.index_update(impact_intv, ops.index[:, AGE_CHILD], 1.,
                                   indices_are_sorted=True, unique_indices=True)

    return impact_intv


def country_EcasesByAge(
    # parameters
    R0_local: float,
    e_cases_N0_local: float,
    log_relsusceptibility_age: np.float64,  # A
    impact_intv_children_effect: float,
    impact_intv_onlychildren_effect: float,
    impact_intv: np.float64,  # N2 x A
    # data
    N0: int,
    elementary_school_reopening_idx_local: int,
    N2: int,
    SCHOOL_STATUS_local: np.float64,  # N2
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
    rev_serial_interval: np.float64,  # SI_CUT
    popByAge_abs_local: np.float64,  # A
    N_init_A: int,
    init_A: np.int64,  # N_init_A
    school_switch: np.int64,  # N2 - N0
) -> np.float64:  # N2 x A

    # probability of infection given contact in location m
    rho0 = R0_local / avg_cntct_local
    rev_serial_interval *= rho0

    relsusceptibility_age = jnp.exp(log_relsusceptibility_age)

    impact_intv_children_effect_padded = jnp.concatenate([impact_intv_children_effect * jnp.ones(A_CHILD),
                                                          jnp.ones(A - A_CHILD)])
    impact_intv_onlychildren_effect_padded = jnp.concatenate([impact_intv_onlychildren_effect * jnp.ones(A_CHILD),
                                                              jnp.ones(A - A_CHILD)])
    impact_intv_children_onlychildren_padded = impact_intv_children_effect_padded * impact_intv_onlychildren_effect_padded

    # define body of main for loop
    def scan_body(carry, x):
        impact_intv_t, weekend_t, school_switch_t = x
        E_casesByAge_sum, E_casesByAge_SI_CUT = carry

        prop_susceptibleByAge = 1.0 - E_casesByAge_sum / popByAge_abs_local
        prop_susceptibleByAge = jnp.maximum(0.0, prop_susceptibleByAge)

        # this convolution is effectively zero padded on the left
        tmp_row_vector_A_no_impact_intv = rev_serial_interval @ E_casesByAge_SI_CUT
        tmp_row_vector_A = impact_intv_t * tmp_row_vector_A_no_impact_intv

        # choose weekend/weekday contact matrices
        cntct_mean_local, cntct_elementary_school_reopening_local, cntct_school_closure_local = cond(
            weekend_t,
            (cntct_weekends_mean_local,
             cntct_elementary_school_reopening_weekends_local,
             cntct_school_closure_weekends_local),
            identity,
            (cntct_weekdays_mean_local,
             cntct_elementary_school_reopening_weekdays_local,
             cntct_school_closure_weekdays_local),
            identity)

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

        # school_switch_t controls which of the three school branches we should follow in this iteration
        # 0 => school_open     1 => school_reopen     2 => school_closed
        contact_inputs = switch(school_switch_t, [school_open, school_reopen, school_closed], None)
        (col1_left, col1_right, col2_topleft, col2_topright,
         col2_bottomleft, col2_bottomright, impact_intv_adult) = contact_inputs

        col1 = (col1_left.T @ col1_right)[0]
        col2 = (col2_topleft.T @ col2_topright)[0] + (col2_bottomleft.T @ col2_bottomright)[0] * impact_intv_adult
        E_casesByAge_t = jnp.concatenate([col1, col2])

        E_casesByAge_t *= prop_susceptibleByAge * relsusceptibility_age

        # add term to cumulative sum
        E_casesByAge_sum = E_casesByAge_sum + E_casesByAge_t
        # basically "roll left and append most recent time slice at right"
        E_casesByAge_SI_CUT = jnp.concatenate([E_casesByAge_SI_CUT[1:], E_casesByAge_t[None, :]])

        return (E_casesByAge_sum, E_casesByAge_SI_CUT), E_casesByAge_t

    # init expected cases by age and location in first N0 days
    E_casesByAge_init = ops.index_update(jnp.zeros((N0, A)), ops.index[:N0, init_A], e_cases_N0_local / N_init_A,
                                         indices_are_sorted=True, unique_indices=True)

    # initialize carry variables
    E_casesByAge_sum = E_casesByAge_init.sum(0)
    # pad with zeros on left
    E_casesByAge_SI_CUT = jnp.concatenate([jnp.zeros((SI_CUT - N0, A)), E_casesByAge_init])

    init = (E_casesByAge_sum, E_casesByAge_SI_CUT)
    xs = (impact_intv[N0:], wkend_mask_local, school_switch)

    # execute for loop using JAX primitive scan
    E_casesByAge_N0_N2 = scan(scan_body, init, xs, length=N2 - N0)[1]

    E_casesByAge = jnp.concatenate([E_casesByAge_init, E_casesByAge_N0_N2])

    return E_casesByAge


# evaluate the line 232
# return C where C[t] = b[-t-1]...b[-1] * A[:t+1]
def circular_vecmat(b, A):
    # To construct the matrix
    #   b2  0  0
    #   b1 b2  0
    #   b0 b1 b2
    # first, we will flip + broadcasting + padding,
    #   b2 b1 b0  0  0
    #   b2 b1 b0  0  0
    #   b2 b1 b0  0  0
    # reshape(-1) + truncate + reshape,
    #   b2 b1 b0  |  0
    #    0 b2 b1  | b0
    #    0  0 b2  | b1
    # then we truncate and transpose the result.
    assert isinstance(b, np.ndarray)
    n = b.shape[0]
    B = np.pad(np.broadcast_to(b[::-1], (n, n)), ((0, 0), (0, n - 1)))
    B = B.reshape(-1)[:n * (2 * n - 2)].reshape((n, -1))[:, :n].T
    return B @ A


def country_EdeathsByAge(
    # parameters
    E_casesByAge_local: np.float64,  # N2 x A
    # data
    # N2: int,
    # A: int,
    rev_ifr_daysSinceInfection: np.float64,  # N2
    log_ifr_age_base: np.float64,  # A
    log_ifr_age_rnde_mid1_local: float,
    log_ifr_age_rnde_mid2_local: float,
    log_ifr_age_rnde_old_local: float,
) -> np.float64:  # N2 x A

    # calculate expected deaths by age and country
    E_deathsByAge = circular_vecmat(rev_ifr_daysSinceInfection[1:], E_casesByAge_local[:-1])
    E_deathsByAge = jnp.concatenate([1e-15 * E_casesByAge_local[:1], E_deathsByAge])

    E_deathsByAge *= jnp.exp(log_ifr_age_base + jnp.concatenate([
        jnp.zeros(4),
        jnp.repeat(log_ifr_age_rnde_mid1_local, 6),
        jnp.repeat(log_ifr_age_rnde_mid2_local, 4),
        jnp.repeat(log_ifr_age_rnde_old_local, 4)]))

    E_deathsByAge += 1e-15
    return E_deathsByAge


class NegBinomial2(dist.GammaPoisson):
    def __init__(self, mu, phi):
        super().__init__(phi, phi / mu)


def countries_log_dens(
    deaths_slice: np.int64,  # M x N2
    # NB: start and end are not required, it is useful for Stan map-reduce but we don't use it
    # start: int,
    # end: int,
    # parameters
    R0: np.float64,  # M
    e_cases_N0: np.float64,  # M
    beta: np.float64,  # COVARIATES_N - 1
    dip_rdeff: np.float64,  # M
    upswing_timeeff_reduced: np.float64,  # N2 x M
    timeeff_shift_age: np.float64,  # M x A
    log_relsusceptibility_age: np.float64,  # 2
    phi: float,
    impact_intv_children_effect: float,
    impact_intv_onlychildren_effect: float,
    # data
    N0: int,
    elementary_school_reopening_idx: np.int64,  # M
    N2: int,
    SCHOOL_STATUS: np.float64,  # N2 x M
    A: int,
    A_CHILD: int,
    AGE_CHILD: np.int64,  # A_CHILD
    COVARIATES_N: int,
    SI_CUT: int,
    # num_wkend_idx: np.int64,  # M
    # wkend_idx: np.int64,  # N2 x M
    wkend_mask: np.bool,  # M x N2
    upswing_timeeff_map: np.int64,  # N2 x M
    avg_cntct: np.float64,  # M
    covariates: np.float64,  # M x COVARIATES_N x N2 x A
    cntct_weekends_mean: np.float64,  # M x A x A
    cntct_weekdays_mean: np.float64,  # M x A x A
    cntct_school_closure_weekends: np.float64,  # M x A x A
    cntct_school_closure_weekdays: np.float64,  # M x A x A
    cntct_elementary_school_reopening_weekends: np.float64,  # M x A x A
    cntct_elementary_school_reopening_weekdays: np.float64,  # M x A x A
    rev_ifr_daysSinceInfection: np.float64,  # N2
    log_ifr_age_base: np.float64,  # A
    log_ifr_age_rnde_mid1: np.float64,  # M
    log_ifr_age_rnde_mid2: np.float64,  # 1M
    log_ifr_age_rnde_old: np.float64,  # M
    rev_serial_interval: np.float64,  # SI_CUT
    # epidemicStart: np.int64,  # M
    # N: np.int64,  # M
    epidemic_mask: np.bool,  # M x N2
    N_init_A: int,
    init_A: np.int64,  # N_init_A
    # A_AD: np.int64,  # M_AD
    dataByAgestart: np.int64,  # M_AD
    dataByAge_mask: np.bool,  # M_AD x N2
    dataByAge_AD_mask: np.bool,  # M_AD x N2 x A
    map_age: np.float64,  # M_AD x A x A
    deathsByAge: np.float64,  # N2 x A x M_AD
    map_country: np.int64,  # M x 2
    popByAge_abs: np.float64,  # M x A
    # ones_vector_A: np.float64,  # A
    smoothed_logcases_weeks_n: np.int64,  # M
    smoothed_logcases_week_map: np.int64,  # M x smoothed_logcases_weeks_n_max x 7
    smoothed_logcases_week_pars: np.float64,  # M x smoothed_logcases_weeks_n_max x 3
    school_case_time_idx: np.int64,  # M x 2
    school_case_data: np.float64,  # M x 4
    school_switch: np.int64,  # M x N2-N0
) -> float:
    lpmf = 0.

    impact_intv = jax.vmap(
        lambda dip_rdeff, upswing_timeeff_reduced, covariates, timeeff_shift_age, upswing_timeeff_map:
            country_impact(
                beta,
                dip_rdeff,
                upswing_timeeff_reduced,
                # N2,
                # A,
                # A_CHILD,
                AGE_CHILD,
                # COVARIATES_N,
                covariates,
                timeeff_shift_age,
                upswing_timeeff_map)
    )(dip_rdeff, upswing_timeeff_reduced.T, covariates, timeeff_shift_age, upswing_timeeff_map.T)

    E_casesByAge = jax.vmap(
        lambda R0, e_cases_N0, impact_intv, elementary_school_reopening_idx, SCHOOL_STATUS, wkend_mask,
        avg_cntct, cntct_weekends_mean, cntct_weekdays_mean, cntct_school_closure_weekends,
        cntct_school_closure_weekdays, cntct_elementary_school_reopening_weekends,
        cntct_elementary_school_reopening_weekdays, popByAge_abs, school_switch:
        country_EcasesByAge(
                R0,
                e_cases_N0,
                log_relsusceptibility_age,
                impact_intv_children_effect,
                impact_intv_onlychildren_effect,
                impact_intv,
                N0,
                elementary_school_reopening_idx,
                N2,
                SCHOOL_STATUS,
                A,
                A_CHILD,
                SI_CUT,
                wkend_mask,
                avg_cntct,
                cntct_weekends_mean,
                cntct_weekdays_mean,
                cntct_school_closure_weekends,
                cntct_school_closure_weekdays,
                cntct_elementary_school_reopening_weekends,
                cntct_elementary_school_reopening_weekdays,
                rev_serial_interval,
                popByAge_abs,
                N_init_A,
                init_A,
                school_switch)
    )(
        R0, e_cases_N0, impact_intv, elementary_school_reopening_idx, SCHOOL_STATUS.T, wkend_mask[:, N0:],
        avg_cntct, cntct_weekends_mean, cntct_weekdays_mean, cntct_school_closure_weekends,
        cntct_school_closure_weekdays, cntct_elementary_school_reopening_weekends,
        cntct_elementary_school_reopening_weekdays, popByAge_abs, school_switch
    )

    E_deathsByAge = jax.vmap(
        lambda E_casesByAge, log_ifr_age_rnde_mid1, log_ifr_age_rnde_mid2, log_ifr_age_rnde_old:
            country_EdeathsByAge(
                E_casesByAge,
                # N2,
                # A,
                rev_ifr_daysSinceInfection,
                log_ifr_age_base,
                log_ifr_age_rnde_mid1,
                log_ifr_age_rnde_mid2,
                log_ifr_age_rnde_old)
    )(E_casesByAge, log_ifr_age_rnde_mid1, log_ifr_age_rnde_mid2, log_ifr_age_rnde_old)

    E_cases = E_casesByAge.sum(-1)  # M x N2

    E_deaths = E_deathsByAge.sum(-1)  # M x N2

    # likelihood death data this location
    lpmf += NegBinomial2(E_deaths, phi).mask(epidemic_mask).log_prob(deaths_slice).sum()

    # filter out countries with deaths by age data
    E_deathsByAge = E_deathsByAge[map_country[:, 0] == 1]  # M_AD x N2 x A
    # first day of data is sumulated death
    E_deathsByAge_firstday = (dataByAge_mask[..., None] * E_deathsByAge).sum(-2)  # M_AD x A
    E_deathsByAge = jax.vmap(lambda x, i, v: ops.index_update(x, i, v))(
        E_deathsByAge, dataByAgestart, E_deathsByAge_firstday)
    # after daily death
    # NB: we mask to get valid mu
    masked_mu = jnp.where(dataByAge_AD_mask, E_deathsByAge @ map_age, 1.)
    lpmf += NegBinomial2(masked_mu, phi).mask(dataByAge_AD_mask).log_prob(jnp.moveaxis(deathsByAge, -1, 0)).sum()
    # likelihood case data this location
    M = E_cases.shape[0]
    E_casesByWeek = jnp.take_along_axis(
        E_cases, smoothed_logcases_week_map.reshape((M, -1)), -1).reshape((M, -1, 7))
    E_log_week_avg_cases = jnp.log(E_casesByWeek).mean(-1)
    studentt_cdf = dist.StudentT(
        smoothed_logcases_week_pars[..., 2],
        smoothed_logcases_week_pars[..., 0],
        smoothed_logcases_week_pars[..., 1]).cdf(E_log_week_avg_cases)
    lpmf += jnp.where(jnp.arange(E_log_week_avg_cases.shape[1]) < smoothed_logcases_weeks_n[:, None],
                      jnp.log(jnp.clip(studentt_cdf, a_min=jnp.finfo(studentt_cdf).tiny)),
                      0.).sum()

    # likelihood school case data this location
    school_case_mask = school_case_time_idx[:, 0] >= 0
    school_case_time_idx = school_case_time_idx[school_case_mask]
    E_casesByAge = E_casesByAge[school_case_mask]
    popByAge_abs = popByAge_abs[school_case_mask]
    school_case_data = school_case_data[school_case_mask]
    time_slice = jnp.arange(N2)
    school_case_time_mask = jnp.logical_and(school_case_time_idx[:, :1] <= time_slice,
                                            time_slice <= school_case_time_idx[:, 1:])
    school_case_weights = jnp.array([1., 1., 0.8])
    school_attack_rate = (school_case_time_mask * (E_casesByAge[..., 1:4] @ school_case_weights)).sum(-1)
    school_attack_rate /= popByAge_abs[:, 1:4] @ school_case_weights

    # prevent over/underflow
    school_attack_rate = jnp.minimum(school_attack_rate, school_case_data[:, 2] * 4)
    lpmf += jnp.log(dist.Normal(school_case_data[:, 0], school_case_data[:, 1])
                        .cdf(school_attack_rate)).sum()
    lpmf += jnp.log(dist.Normal(-school_case_data[:, 2], school_case_data[:, 3])
                        .cdf(-school_attack_rate)).sum()
    return lpmf
