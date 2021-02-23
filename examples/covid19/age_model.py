# Ported from https://github.com/ImperialCollegeLondon/covid19model/blob/master/
# covid19AgeModel/inst/stan-models/covid19AgeModel_v120_cmdstanv.stan

import numpy as np

import jax.numpy as jnp
import jax.ops as ops

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform


# checks if pos is in pos_var
def r_in(pos: int, pos_var: list) -> bool:
    return pos in pos_var  # it seems that this function is not necessary


# returns multiplier on the rows of the contact matrix over time for one country
def country_impact(
    beta: jnp.float32,  # 2
    dip_rdeff_local: float,
    upswing_timeeff_reduced_local: jnp.float32,  # 1D
    N2: int,  # num of days
    A: int,  # num of ages
    A_CHILD: int,  # num of children ages
    AGE_CHILD: jnp.int32,  # A_CHILD
    COVARIATES_N: int,
    covariates_local: jnp.float32,  # 3 x N2 x A
    upswing_age_rdeff_local: jnp.float32,  # A
    upswing_timeeff_map_local: jnp.int32,  # N2
) -> jnp.float32:  # N2 x A
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
    n = b.shape[0]
    B = jnp.pad(jnp.broadcast_to(b, (n, n)), (0, n - 1))
    B = B.reshape(-1)[:n * (2 * n - 2)].reshape((n, -1))[:, :n].T
    return B @ A


def country_EdeathsByAge(
    # parameters
    E_casesByAge_local: jnp.float32,  # N2 x A
    # data
    N2: int,
    A: int,
    rev_ifr_daysSinceInfection: jnp.float32,  # N2
    log_ifr_age_base: jnp.float32,  # A
    log_ifr_age_rnde_mid1_local: float,
    log_ifr_age_rnde_mid2_local: float,
    log_ifr_age_rnde_old_local: float,
) -> jnp.float32:  # N2 x A

    # calculate expected deaths by age and country
    E_deathsByAge = circular_vecmat(rev_ifr_daysSinceInfection[1:], E_casesByAge_local[:-1])
    E_deathsByAge = jnp.concatenate([1e-15 * E_casesByAge_local[:1], E_deathsByAge])

    # Based on the code, A = 18: 4 for children, 6 for middle age 1, 4 for middle 2, 4 for old?
    assert A == 18
    E_deathsByAge *= jnp.exp(log_ifr_age_base + jnp.concatenate([
        jnp.zeros(4),
        jnp.full((6,), log_ifr_age_rnde_mid1_local),
        jnp.full((4,), log_ifr_age_rnde_mid2_local),
        jnp.full((4,), log_ifr_age_rnde_old_local)]))

    E_deathsByAge += 1e-15
    return E_deathsByAge


class NegBinomial2(dist.GammaPoisson):
    def __init__(self, mu, phi):
        super().__init__(phi, phi / mu)


def countries_log_dens(
    deaths_slice: jnp.int32,  # 2D
    # NB: start and end are not required, it is useful for Stan map-reduce but we don't use it
    start: int,
    end: int,
    # parameters
    R0: jnp.float32,  # 1D
    e_cases_N0: jnp.float32,  # 1D
    beta: jnp.float32,  # 1D
    dip_rdeff: jnp.float32,  # 1D
    upswing_timeeff_reduced: jnp.float32,  # 2D
    timeeff_shift_age: jnp.float32,  # 2D
    log_relsusceptibility_age: jnp.float32,  # 1D
    phi: float,
    impact_intv_children_effect: float,
    impact_intv_onlychildren_effect: float,
    # data
    N0: int,
    elementary_school_reopening_idx: jnp.int32,  # 1D
    N2: int,
    SCHOOL_STATUS: jnp.float32,  # 2D
    A: int,
    A_CHILD: int,
    AGE_CHILD: jnp.int32,  # 1D
    COVARIATES_N: int,
    SI_CUT: int,
    num_wkend_idx: jnp.int32,  # 1D
    wkend_idx: jnp.int32,  # 2D
    upswing_timeeff_map: jnp.int32,  # 2D
    avg_cntct: jnp.float32,  # 1D
    covariates: jnp.float32,  # 2D
    cntct_weekends_mean: jnp.float32,  # 2D
    cntct_weekdays_mean: jnp.float32,  # 2D
    cntct_school_closure_weekends: jnp.float32,  # 2D
    cntct_school_closure_weekdays: jnp.float32,  # 2D
    cntct_elementary_school_reopening_weekends: jnp.float32,  # 2D
    cntct_elementary_school_reopening_weekdays: jnp.float32,  # 2D
    rev_ifr_daysSinceInfection: jnp.float32,  # 1D
    log_ifr_age_base: jnp.float32,  # 1D
    log_ifr_age_rnde_mid1: jnp.float32,  # 1D
    log_ifr_age_rnde_mid2: jnp.float32,  # 1D
    log_ifr_age_rnde_old: jnp.float32,  # 1D
    rev_serial_interval: jnp.float32,  # 1D
    epidemicStart: jnp.int32,  # 1D
    N: jnp.int32,  # 1D
    N_init_A: int,
    init_A: jnp.int32,  # 1D
    A_AD: jnp.int32,  # 1D
    dataByAgestart: jnp.int32,  # 1D
    map_age: jnp.float32,  # 2D
    deathsByAge: jnp.float32,  # 3D
    map_country: jnp.int32,  # 2D
    popByAge_abs: jnp.float32,  # 2D
    ones_vector_A: jnp.float32,  # 1D
    smoothed_logcases_weeks_n: jnp.int32,  # 1D
    smoothed_logcases_week_map: jnp.int32,  # 3D
    smoothed_logcases_week_pars: jnp.float32,  # 3D
    school_case_time_idx: jnp.int32,  # 2D
    school_case_data: jnp.float32,  # 2D
) -> float:
    lpmf = 0.
    M_slice = end - start + 1

    # TODO: finish the implementation and vmap
    for m_slice in range(M_slice):
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


data_def = {
    "M": int,  # number of countries
    "N0": int,  # number of initial days for which to estimate infections
    "N": jnp.int32,  # M - days of observed data for country m. each entry must be <= N2
    "N2": int,  # days of observed data + # of days to forecast
    "A": int,  # number of age bands
    "SI_CUT": int,  # number of days in serial interval to consider
    "COVARIATES_N": int,  # number of days in serial interval to consider
    "N_init_A": int,  # number of age bands with initial cases
    "WKEND_IDX_N": jnp.int32,  # M - number of weekend indices in each location
    "N_IMP": int,  # number of impact invervention time effects. <= M
    # data
    "pop": jnp.float32,  # M
    "popByAge": jnp.float32,  # A x M - proportion of age bracket in population in location
    "epidemicStart": jnp.int32,  # M
    "deaths": jnp.int32,  # N2 x M - reported deaths -- the rows with i >= N contain -1 and should be ignored
    # time index after which schools reopen for country m IF school status is set to 0. each entry must be <= N2
    "elementary_school_reopening_idx": jnp.int32,  # M
    "wkend_idx": jnp.int32,  # N2 x M - indices of 0:N2 that correspond to weekends in location m
    "upswing_timeeff_map": jnp.int32,  # N2 x M - map of impact intv time effects to time units in model for each state
    # mobility trends
    "covariates": jnp.float32,  # M x COVARIATES_N x N2 x A - predictors for fsq contacts by age
    # death data by age
    "M_AD": int,  # number of countries with deaths by age data
    "dataByAgestart": jnp.int32,  # M_AD - start of death by age data
    # the rows with i < dataByAgestart[M_AD] contain -1 and should be ignored
    # + the column with j > A2[M_AD] contain -1 and should be ignored
    "deathsByAge": jnp.int32,  # N2 x A x M_AD - reported deaths by age
    "A_AD": jnp.int32,  # M_AD - number of age groups reported >= 1
    # the column with j > A2[M_AD] contain -1 and should be ignored
    "map_age": jnp.float32,  # M_AD x A x A - map the age groups reported with 5 y age group
    # first column indicates if country has death by age date (1 if yes), 2 column map the country to M_AD
    "map_country":  jnp.float32,  # M x 2
    # case data by age
    "smoothed_logcases_weeks_n_max": int,
    "smoothed_logcases_weeks_n": jnp.int32,  # M - number of week indices per location
    # map of week indices to time indices
    "smoothed_logcases_week_map": jnp.int32,  # M x smoothed_logcases_weeks_n_max x 7
    # likelihood parameters for observed cases
    "smoothed_logcases_week_pars": jnp.float32,  # M x smoothed_logcases_weeks_n_max x 3
    # school case data
    "school_case_time_idx": jnp.int32,  # M x 2
    "school_case_data": jnp.float32,  # M x 4
    # school closure status
    "SCHOOL_STATUS": jnp.float32,  # N2 x M - school status, 1 if close, 0 if open
    # contact matrices
    "A_CHILD": int,  # number of age band for child
    "AGE_CHILD": int,  # A_CHILD - age bands with child
    # min cntct_weekdays_mean and contact intensities during outbreak estimated in Zhang et al
    "cntct_school_closure_weekdays": jnp.float32,  # M x A x A
    # min cntct_weekends_mean and contact intensities during outbreak estimated in Zhang et al
    "cntct_school_closure_weekends": jnp.float32,  # M x A x A
    "cntct_elementary_school_reopening_weekdays": jnp.float32,  # M x A x A - contact matrix for school reopening
    "cntct_elementary_school_reopening_weekends": jnp.float32,  # M x A x A - contact matrix for school reopening
    # priors
    "cntct_weekdays_mean": jnp.float32,  # M x A x A - mean of prior contact rates between age groups on weekdays
    "cntct_weekends_mean": jnp.float32,  # M x A x A - mean of prior contact rates between age groups on weekends
    "hyperpara_ifr_age_lnmu": jnp.float32,  # A - hyper-parameters for probability of death in age band a log normal mean
    "hyperpara_ifr_age_lnsd": jnp.float32,  # A - hyper-parameters for probability of death in age band a log normal sd
    "rev_ifr_daysSinceInfection": jnp.float32,  # N2 - probability of death s days after infection in reverse order
    # fixed pre-calculated serial interval using empirical data from Neil in reverse order
    "rev_serial_interval": jnp.float32,  # SI_CUT
    "init_A": jnp.int32,  # N_init_A - age band in which initial cases occur in the first N0 days
}


# transform data before feeding into the model/mcmc
def transform_data(data):  # lines 438 -> 503
    # given a dictionary of data, return a dictionary of data + transformed data
    # TODO: shift index arrays by 1
    cntct_weekdays_mean = data["cntct_weekdays_mean"].sum(-1)
    cntct_weekends_mean = data["cntct_weekends_mean"].sum(-1)
    popByAge_trans = data["popByAge"].T
    avg_cntct = (popByAge_trans * cntct_weekdays_mean).sum(-1) * (5. / 7.)  # M
    data["avg_cntct"] = avg_cntct + (popByAge_trans * cntct_weekends_mean).sum(-1) * (2. / 7.)
    # reported deaths -- the rows with i > N contain -1 and should be ignored
    data["trans_deaths"] = data["deaths"].T  # M x N2
    data["popByAge_abs"] = data["popByAge"].T * data["pop"][:, None]  # M x A


def model(data):  # lines 523 -> end
    # priors
    sd_dip_rnde = numpyro.sample("sd_dip_rnde", dist.Exponential(1.5))
    phi = numpyro.sample("phi", dist.Normal(0, 5))  # overdispersion parameter for likelihood model
    hyper_log_ifr_age_rnde_mid1 = numpyro.sample("hyper_log_ifr_age_rnde_mid1", dist.Exponential(.1))
    hyper_log_ifr_age_rnde_mid2 = numpyro.sample("hyper_log_ifr_age_rnde_mid2", dist.Exponential(.1))
    hyper_log_ifr_age_rnde_old = numpyro.sample("hyper_log_ifr_age_rnde_old", dist.Exponential(.1))
    log_relsusceptibility_age_reduced = numpyro.sample(
        "log_relsusceptibility_age_reduced",
        dist.Normal(jnp.array([-1.0702331, 0.3828269]), jnp.array([0.2169696, 0.1638433])))
    sd_upswing_timeeff_reduced = numpyro.sample("sd_upswing_timeeff_reduced", dist.LogNormal(-1.2, 0.2))
    hyper_timeeff_shift_mid1 = numpyro.sample("hyper_timeeff_shift_mid1", dist.Exponential(.1))
    impact_intv_children_effect = numpyro.sample("impact_intv_children_effect", dist.Uniform(0.1, 1.0))
    impact_intv_onlychildren_effect = numpyro.sample(
        "impact_intv_onlychildren_effect", dist.LogNormal(0, 0.35))

    with numpyro.plate("M", data["M"]):
        R0 = numpyro.sample("R0", dist.LogNormal(0.98, 0.2))
        # expected number of cases per day in the first N0 days, for each country
        e_cases_N0 = numpyro.sample("e_cases_N0", dist.LogNormal(4.85, 0.4))
        with numpyro.handlers.reparam(config=numpyro.infer.reparam.TransformReparam()):
            dip_rnde = numpyro.sample("dip_rnde", dist.TransformedDistribution(
                dist.Normal(0., 1.), AffineTransform(0., sd_dip_rnde)))
            log_ifr_age_rnde_mid1 = numpyro.sample(
                "log_ifr_age_rnde_mid1", dist.TransformedDistribution(
                    dist.Exponential(1.), AffineTransform(0., 1 / hyper_log_ifr_age_rnde_mid1)))
            log_ifr_age_rnde_mid2 = numpyro.sample(
                "log_ifr_age_rnde_mid2", dist.TransformedDistribution(
                    dist.Exponential(1.), AffineTransform(0., 1 / hyper_log_ifr_age_rnde_mid2)))
            log_ifr_age_rnde_old = numpyro.sample(
                "log_ifr_age_rnde_old", dist.TransformedDistribution(
                    dist.Exponential(1.), AffineTransform(0., 1 / hyper_log_ifr_age_rnde_old)))
            sd_upswing_timeeff_reduced = jnp.concatenate(
                [jnp.array([0.025]), jnp.repeat(sd_upswing_timeeff_reduced, data["N_IMP"] - 1)])
            upswing_timeeff_reduced = numpyro.sample(
                "upswing_timeeff_reduced_base", dist.TransformedDistribution(
                    dist.GaussianRandomWalk(1., data["N_IMP"]),
                    AffineTransform(0., sd_upswing_timeeff_reduced)))
            upswing_timeeff_reduced = upswing_timeeff_reduced.T  # M x N_IMP
            timeeff_shift_mid1 = numpyro.sample(
                "timeeff_shift_mid1", dist.TransformedDistribution(
                    dist.Exponential(1.), AffineTransform(0., hyper_timeeff_shift_mid1)))

    with numpyro.plate("COVARIATES_Nm1", data["COVARIATES_N"] - 1):
        # regression coefficients for time varying multipliers on contacts
        beta = numpyro.sample("beta", dist.Normal(0., 1.))

    with numpyro.plate("A", data["A"]):
        # probability of death for age band a
        log_ifr_age_base = numpyro.sample(
            "log_ifr_age_base", dist.Normal(data["hyperpara_ifr_age_lnmu"], data["hyperpara_ifr_age_lnsd"]))

    # transformed parameters
    log_relsusceptibility_age = jnp.concatenate(
        [
            jnp.repeat(log_relsusceptibility_age_reduced[0], 3),
            jnp.zeros(10),
            jnp.repeat(log_relsusceptibility_age_reduced[1], 5)
        ],
        axis=-1)  # A

    timeeff_shift_age = jnp.concatenate(
        [jnp.zeros(4), jnp.broadcast_to(timeeff_shift_mid1[:, None], (data["M"], 6)), jnp.zeros(8)],
        axis=-1)  # M x A

    countries_log_factor = countries_log_dens(
        data["trans_deaths"],
        0,
        R0,
        e_cases_N0,
        beta,
        dip_rnde,
        upswing_timeeff_reduced,
        timeeff_shift_age,
        log_relsusceptibility_age,
        phi,
        impact_intv_children_effect,
        impact_intv_onlychildren_effect,
        data["N0"],
        data["elementary_school_reopening_idx"],
        data["N2"],
        data["SCHOOL_STATUS"],
        data["A"],
        data["A_CHILD"],
        data["AGE_CHILD"],
        data["COVARIATES_N"],
        data["SI_CUT"],
        data["WKEND_IDX_N"],
        data["wkend_idx"],
        data["upswing_timeeff_map"],
        data["avg_cntct"],
        data["covariates"],
        data["cntct_weekends_mean"],
        data["cntct_weekdays_mean"],
        data["cntct_school_closure_weekends"],
        data["cntct_school_closure_weekdays"],
        data["cntct_elementary_school_reopening_weekends"],
        data["cntct_elementary_school_reopening_weekdays"],
        data["rev_ifr_daysSinceInfection"],
        log_ifr_age_base,
        log_ifr_age_rnde_mid1,
        log_ifr_age_rnde_mid2,
        log_ifr_age_rnde_old,
        data["rev_serial_interval"],
        data["epidemicStart"],
        data["N"],
        data["N_init_A"],
        data["init_A"],
        data["A_AD"],
        data["dataByAgestart"],
        data["map_age"],
        data["deathsByAge"],
        data["map_country"],
        data["popByAge_abs"],
        data["ones_vector_A"],
        data["smoothed_logcases_weeks_n"],
        data["smoothed_logcases_week_map"],
        data["smoothed_logcases_week_pars"],
        data["school_case_time_idx"],
        data["school_case_data"],
    )
    numpyro.factor("contries_log_factor", countries_log_factor)


# num_samples=1500
# num_warmup=1000
# target_accept_prob=0.95
# max_tree_depth=15
# algo=NUTS
def get_data():
    import numpy as np
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri

    from rpy2.robjects.conversion import localconverter

    file = "covid19AgeModel_v120_cmdstanv-40states_Oct29-140172_stanin.RData"
    ro.r['load'](file)
    r_df = ro.r['stan_data']
    r_df = dict(zip(r_df.names, list(r_df)))
    data = {}
    with localconverter(ro.default_converter + pandas2ri.converter):
        for k, v in r_df.items():
            if isinstance(v, ro.vectors.ListVector):
                df = np.array([ro.conversion.rpy2py(x) for x in v])
            else:
                df = ro.conversion.rpy2py(v)
            data[k] = df
    return data


data = get_data()
for k, v in data.items():
    assert isinstance(v, np.ndarray)
