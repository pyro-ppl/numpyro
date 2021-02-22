# Ported from https://github.com/ImperialCollegeLondon/covid19model/blob/master/covid19AgeModel/inst/stan-models/covid19AgeModel_v120_cmdstanv.stan

import jax.numpy as jnp
import jax.ops as ops


# checks if pos is in pos_var
def r_in(pos: int, pos_var: list(int)) -> bool:
    return pos in pos_var  # using scan?  it seems that this function is not necessary


# returns multiplier on the rows of the contact matrix over time for one country
def country_impact(
    beta: jnp.float,  # 2
    dip_rdeff_local: float,
    upswing_timeeff_reduced_local: jnp.float,  # None
    N2: int,  # num of days
    A: int,  # num of ages
    A_CHILD: int,  # num of children ages
    AGE_CHILD: jnp.int,  # A_CHILD
    COVARIATES_N: int,
    covariates_local: jnp.float,  # 3 x N2 x A
    upswing_age_rdeff_local: jnp.float,  # A
    upswing_timeeff_map_local: jnp.int,  # N2
) -> jnp.float:  # N2 x A
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
    log_relsusceptibility_age: jnp.float,  # A
    impact_intv_children_effect: float,
    impact_intv_onlychildren_effect: float,
    impact_intv: jnp.float,  # N2 x A
    # data
    N0: int,
    elementary_school_reopening_idx_local: int,
    N2: int,
    SCHOOL_STATUS_local: jnp.float,  # N2
    A: int,
    A_CHILD: int,
    SI_CUT: int,
    # TODO: convert this to a boolean vector of length N2 - N0:
    #   [t in wken_idx_local for t in [N0, N2)]
    wkend_idx_local: jnp.int,  # None
    avg_cntct_local: float,
    cntct_weekends_mean_local: jnp.float,  # A x A
    cntct_weekdays_mean_local: jnp.float,  # A x A
    cntct_school_closure_weekends_local: jnp.float,  # A x A
    cntct_school_closure_weekdays_local: jnp.float,  # A x A
    cntct_elementary_school_reopening_weekends_local: jnp.float,  # A x A
    cntct_elementary_school_reopening_weekdays_local: jnp.float,  # A x A
    rev_serial_interval: jnp.float,  # SI_CUT
    popByAge_abs_local: jnp.float,  # A
    N_init_A: int,
    init_A: jnp.int,  # N_init_A
) -> jnp.float:  # N2 x A

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
    # TODO: vectorize the loop over N0 -> N2 days (seems possible by masking the convolution)

    return E_casesByAge


def country_EdeathsByAge(
    # parameters
    E_casesByAge_local: jnp.float,  # N2 x A
    # data
    N2: int,
    A: int,
    rev_ifr_daysSinceInfection: jnp.float,  # N2
    log_ifr_age_base: jnp.float,  # A
    log_ifr_age_rnde_mid1_local: float,
    log_ifr_age_rnde_mid2_local: float,
    log_ifr_age_rnde_old_local: float,
) -> jnp.float:  # N2 x A

    E_deathsByAge = jnp.zeros((N2, A))

    # calculate expected deaths by age and country
    # NB: seems like we can mask tril and using matmul
    # Based on the code, A = 18: 4 for children, 6 for middle age 1, 4 for middle 2, 4 for old?

    E_deathsByAge += 1e-15
    return E_deathsByAge


def countries_log_dens(
    deaths_slice: jnp.int,  # 2D
    # NB: start and end are not required, it is useful for Stan map-reduce but we don't use it
    start: int,
    end: int,
    # parameters
    R0: jnp.float,  # 1D
    e_cases_N0: jnp.float,  # 1D
    beta: jnp.float,  # 1D
    dip_rdeff: jnp.float,  # 1D
    upswing_timeeff_reduced: jnp.float,  # 2D
    timeeff_shift_age: jnp.float,  # 2D
    log_relsusceptibility_age: jnp.float,  # 1D
    phi: float,
    impact_intv_children_effect: float,
    impact_intv_onlychildren_effect: float,
    # data
    N0: int,
    elementary_school_reopening_idx: jnp.int,  # 1D
    N2: int,
    SCHOOL_STATUS: jnp.float,  # 2D
    A: int,
    A_CHILD: int,
    AGE_CHILD: jnp.int,  # 1D
    COVARIATES_N: int,
    SI_CUT: int,
    num_wkend_idx: jnp.int,  # 1D
    wkend_idx: jnp.int,  # 2D
    upswing_timeeff_map: jnp.int,  # 2D
    avg_cntct: jnp.float,  # 1D
    covariates: jnp.float,  # 2D
    cntct_weekends_mean: jnp.float,  # 2D
    cntct_weekdays_mean: jnp.float,  # 2D
    cntct_school_closure_weekends: jnp.float,  # 2D
    cntct_school_closure_weekdays: jnp.float,  # 2D
    cntct_elementary_school_reopening_weekends: jnp.float,  # 2D
    cntct_elementary_school_reopening_weekdays: jnp.float,  # 2D
    rev_ifr_daysSinceInfection: jnp.float,  # 1D
    log_ifr_age_base: jnp.float,  # 1D
    log_ifr_age_rnde_mid1: jnp.float,  # 1D
    log_ifr_age_rnde_mid2: jnp.float,  # 1D
    log_ifr_age_rnde_old: jnp.float,  # 1D
    rev_serial_interval: jnp.float,  # 1D
    epidemicStart: jnp.int,  # 1D
    N: jnp.int,  # 1D
    N_init_A: int,
    init_A: jnp.int,  # 1D
    A_AD: jnp.int,  # 1D
    dataByAgestart: jnp.int,  # 1D
    map_age: jnp.float,  # 2D
    deathsByAge: jnp.float,  # 3D
    map_country: jnp.int,  # 2D
    popByAge_abs: jnp.float,  # 2D
    ones_vector_A: jnp.float,  # 1D
    smoothed_logcases_weeks_n: jnp.int,  # 1D
    smoothed_logcases_week_map: jnp.int,  # 3D
    smoothed_logcases_week_pars: jnp.float,  # 3D
    school_case_time_idx: jnp.int,  # 2D
    school_case_data: jnp.float,  # 2D
) -> float:
    lpmf = 0.
    M_slice = end - start + 1

    for m_slice in range(M_slice):  # vmap
        m = m_slice + start
        impact_intv = country_impact(
            beta,
            dip_rdeff[m],
            upswing_timeeff_reduced[:, m],
            N2,
            A,
            A_CHILD,
            AGE_CHILD,
            COVARIATES_N,
            covariates[m],
            timeeff_shift_age[m],
            upswing_timeeff_map[:, m],
        )

        E_casesByAge = country_EcasesByAge(
            R0[m],
            e_cases_N0[m],
            log_relsusceptibility_age,
            impact_intv_children_effect,
            impact_intv_onlychildren_effect,
            impact_intv,
            N0,
            elementary_school_reopening_idx[m],
            N2,
            SCHOOL_STATUS[:, m],
            A,
            A_CHILD,
            SI_CUT,
            wkend_idx[:num_wkend_idx[m], m],
            avg_cntct[m],
            cntct_weekends_mean[m],
            cntct_weekdays_mean[m],
            cntct_school_closure_weekends[m],
            cntct_school_closure_weekdays[m],
            cntct_elementary_school_reopening_weekends[m],
            cntct_elementary_school_reopening_weekdays[m],
            rev_serial_interval,
            popByAge_abs[m],
            N_init_A,
            init_A,
        )

        E_deathsByAge = country_EdeathsByAge(
            E_casesByAge,
            N2,
            A,
            rev_ifr_daysSinceInfection,
            log_ifr_age_base,
            log_ifr_age_rnde_mid1[m],
            log_ifr_age_rnde_mid2[m],
            log_ifr_age_rnde_old[m],
        )

        E_cases = E_casesByAge * ones_vector_A

        E_deaths = E_deathsByAge * ones_vector_A

        # likelihood death data this location
        lpmf += 0.

        # likelihood case data this location
        lpmf += 0.

        # likelihood school case data this location
        lpmf += 0.

    return lpmf


# transform data before feeding into the model/mcmc
def transformed_data(data):  # lines 438 -> 503
    # given a dictionary of data, return a dictionary of data + transformed data
    pass


def model(data):  # lines 523 -> end
    # TODO: add a bunch of priors
    pass
