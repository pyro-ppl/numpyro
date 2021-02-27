# Ported from https://github.com/ImperialCollegeLondon/covid19model/blob/master/
# covid19AgeModel/inst/stan-models/covid19AgeModel_v120_cmdstanv.stan

import numpy as np

import jax
import jax.numpy as jnp
import jax.ops as ops

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer.reparam import TransformReparam


# returns multiplier on the rows of the contact matrix over time for one country
def country_impact(
    beta: np.float64,  # 2
    N2: int,  # num of days
    A: int,  # num of ages
    A_CHILD: int,  # num of children ages
    AGE_CHILD: np.int64,  # A_CHILD
    COVARIATES_N: int,
    dip_rdeff_local: float,
    upswing_timeeff_reduced_local: np.float64,  # 1D
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
    rev_serial_interval: np.float64,  # SI_CUT
    N_init_A: int,
    init_A: np.int64,  # N_init_A
    R0_local: float,
    e_cases_N0_local: float,
    elementary_school_reopening_idx_local: int,
    SCHOOL_STATUS_local: np.float64,  # N2
    # wkend_idx_local: np.int64,  # 1D
    wkend_mask_local: np.bool,  # 1D
    avg_cntct_local: float,
    cntct_weekends_mean_local: np.float64,  # A x A
    cntct_weekdays_mean_local: np.float64,  # A x A
    cntct_school_closure_weekends_local: np.float64,  # A x A
    cntct_school_closure_weekdays_local: np.float64,  # A x A
    cntct_elementary_school_reopening_weekends_local: np.float64,  # A x A
    cntct_elementary_school_reopening_weekdays_local: np.float64,  # A x A
    popByAge_abs_local: np.float64,  # A
) -> np.float64:  # N2 x A

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
    # TODO: implement using scan
    E_casesByAge += 1000.  # NB: added this for temporary testing

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
    # data
    N2: int,
    A: int,
    rev_ifr_daysSinceInfection: np.float64,  # N2
    log_ifr_age_base: np.float64,  # A
    # parameters
    E_casesByAge_local: np.float64,  # N2 x A
    log_ifr_age_rnde_mid1_local: float,
    log_ifr_age_rnde_mid2_local: float,
    log_ifr_age_rnde_old_local: float,
) -> np.float64:  # N2 x A

    # calculate expected deaths by age and country
    E_deathsByAge = circular_vecmat(rev_ifr_daysSinceInfection[1:], E_casesByAge_local[:-1])
    E_deathsByAge = jnp.concatenate([1e-15 * E_casesByAge_local[:1], E_deathsByAge])

    # Based on the code, A = 18: 4 for children, 6 for middle age 1, 4 for middle 2, 4 for old?
    assert A == 18
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
    deaths_slice: np.int64,  # 2D
    # NB: start and end are not required, it is useful for Stan map-reduce but we don't use it
    # start: int,
    # end: int,
    # parameters
    R0: np.float64,  # 1D
    e_cases_N0: np.float64,  # 1D
    beta: np.float64,  # 1D
    dip_rdeff: np.float64,  # 1D
    upswing_timeeff_reduced: np.float64,  # 2D
    timeeff_shift_age: np.float64,  # 2D
    log_relsusceptibility_age: np.float64,  # 1D
    phi: float,
    impact_intv_children_effect: float,
    impact_intv_onlychildren_effect: float,
    # data
    N0: int,
    elementary_school_reopening_idx: np.int64,  # 1D
    N2: int,
    SCHOOL_STATUS: np.float64,  # 2D
    A: int,
    A_CHILD: int,
    AGE_CHILD: np.int64,  # 1D
    COVARIATES_N: int,
    SI_CUT: int,
    # num_wkend_idx: np.int64,  # 1D
    # wkend_idx: np.int64,  # 2D
    wkend_mask: np.bool,  # 2D
    upswing_timeeff_map: np.int64,  # 2D
    avg_cntct: np.float64,  # 1D
    covariates: np.float64,  # 2D
    cntct_weekends_mean: np.float64,  # 2D
    cntct_weekdays_mean: np.float64,  # 2D
    cntct_school_closure_weekends: np.float64,  # 2D
    cntct_school_closure_weekdays: np.float64,  # 2D
    cntct_elementary_school_reopening_weekends: np.float64,  # 2D
    cntct_elementary_school_reopening_weekdays: np.float64,  # 2D
    rev_ifr_daysSinceInfection: np.float64,  # 1D
    log_ifr_age_base: np.float64,  # 1D
    log_ifr_age_rnde_mid1: np.float64,  # 1D
    log_ifr_age_rnde_mid2: np.float64,  # 1D
    log_ifr_age_rnde_old: np.float64,  # 1D
    rev_serial_interval: np.float64,  # 1D
    # epidemicStart: np.int64,  # 1D
    # N: np.int64,  # 1D
    epidemic_mask: np.bool,  # 2D
    N_init_A: int,
    init_A: np.int64,  # 1D
    # A_AD: np.int64,  # 1D
    dataByAgestart: np.int64,  # 1D
    dataByAge_mask: np.bool,  # 2D: M_AD x N2
    dataByAge_AD_mask: np.bool,  # 3D: M_AD x N2 x A
    map_age: np.float64,  # 2D
    deathsByAge: np.float64,  # 3D
    map_country: np.int64,  # 2D
    popByAge_abs: np.float64,  # 2D
    # ones_vector_A: np.float64,  # 1D
    smoothed_logcases_weeks_n: np.int64,  # 1D
    smoothed_logcases_week_map: np.int64,  # 3D
    smoothed_logcases_week_pars: np.float64,  # 3D
    # school_case_time_idx: np.int64,  # 2D
    school_case_time_mask: np.bool,  # M x N2
    school_case_data: np.float64,  # M x 4
) -> float:
    lpmf = 0.

    impact_intv = jax.vmap(jax.partial(
        country_impact,
        beta,
        N2,
        A,
        A_CHILD,
        AGE_CHILD,
        COVARIATES_N,
    ))(
        dip_rdeff,
        upswing_timeeff_reduced.T,
        covariates,
        timeeff_shift_age,
        upswing_timeeff_map.T,
    )

    E_casesByAge = jax.vmap(jax.partial(
        country_EcasesByAge,
        log_relsusceptibility_age,
        impact_intv_children_effect,
        impact_intv_onlychildren_effect,
        impact_intv,
        N0,
        N2,
        A,
        A_CHILD,
        SI_CUT,
        rev_serial_interval,
        N_init_A,
        init_A,
    ))(
        R0,
        e_cases_N0,
        elementary_school_reopening_idx,
        SCHOOL_STATUS.T,
        # wkend_idx[:num_wkend_idx[m], m],
        wkend_mask,
        avg_cntct,
        cntct_weekends_mean,
        cntct_weekdays_mean,
        cntct_school_closure_weekends,
        cntct_school_closure_weekdays,
        cntct_elementary_school_reopening_weekends,
        cntct_elementary_school_reopening_weekdays,
        popByAge_abs,
    )

    E_deathsByAge = jax.vmap(jax.partial(
        country_EdeathsByAge,
        N2,
        A,
        rev_ifr_daysSinceInfection,
        log_ifr_age_base,
    ))(
        E_casesByAge,
        log_ifr_age_rnde_mid1,
        log_ifr_age_rnde_mid2,
        log_ifr_age_rnde_old,
    )

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
    E_casesByWeek = jnp.take_along_axis(
        E_cases, smoothed_logcases_week_map.reshape((data["M"], -1)), -1).reshape((data["M"], -1, 7))
    E_log_week_avg_cases = jnp.log(E_casesByWeek).mean(-1)
    lpmf += jnp.where(jnp.arange(E_log_week_avg_cases.shape[1]) < smoothed_logcases_weeks_n[:, None],
                      jnp.log(dist.StudentT(smoothed_logcases_week_pars[..., 2],
                                            smoothed_logcases_week_pars[..., 0],
                                            smoothed_logcases_week_pars[..., 1])
                                  .cdf(E_log_week_avg_cases)),
                      0.).sum()

    # likelihood school case data this location
    school_case_weights = jnp.array([1., 1., 0.8])
    school_attack_rate = (school_case_time_mask * (E_casesByAge[:, :, 1:4] @ school_case_weights)).sum(-1)
    school_attack_rate /= popByAge_abs[:, 1:4] @ school_case_weights

    # prevent over/underflow
    school_attack_rate = jnp.minimum(school_attack_rate, school_case_data[:, 2] * 4)
    lpmf += jnp.where(school_case_time_mask.all(-1),
                      jnp.log(dist.Normal(school_case_data[:, [0, 2]], school_case_data[:, [1, 3]])
                                  .cdf(school_attack_rate[:, None])).sum(-1),
                      0.).sum()
    return lpmf


data_def = {
    "M": int,  # number of countries
    "N0": int,  # number of initial days for which to estimate infections
    "N": np.int64,  # M - days of observed data for country m. each entry must be <= N2
    "N2": int,  # days of observed data + # of days to forecast
    "A": int,  # number of age bands
    "SI_CUT": int,  # number of days in serial interval to consider
    "COVARIATES_N": int,  # number of days in serial interval to consider
    "N_init_A": int,  # number of age bands with initial cases
    "WKEND_IDX_N": np.int64,  # M - number of weekend indices in each location
    "N_IMP": int,  # number of impact invervention time effects. <= M
    # data
    "pop": np.float64,  # M
    "popByAge": np.float64,  # A x M - proportion of age bracket in population in location
    "epidemicStart": np.int64,  # M
    "deaths": np.int64,  # N2 x M - reported deaths -- the rows with i >= N contain -1 and should be ignored
    # time index after which schools reopen for country m IF school status is set to 0. each entry must be <= N2
    "elementary_school_reopening_idx": np.int64,  # M
    "wkend_idx": np.int64,  # N2 x M - indices of 0:N2 that correspond to weekends in location m
    "upswing_timeeff_map": np.int64,  # N2 x M - map of impact intv time effects to time units in model for each state
    # mobility trends
    "covariates": np.float64,  # M x COVARIATES_N x N2 x A - predictors for fsq contacts by age
    # death data by age
    "M_AD": int,  # number of countries with deaths by age data
    "dataByAgestart": np.int64,  # M_AD - start of death by age data
    # the rows with i < dataByAgestart[M_AD] contain -1 and should be ignored
    # + the column with j > A2[M_AD] contain -1 and should be ignored
    "deathsByAge": np.int64,  # N2 x A x M_AD - reported deaths by age
    "A_AD": np.int64,  # M_AD - number of age groups reported >= 1
    # the column with j > A2[M_AD] contain -1 and should be ignored
    "map_age": np.float64,  # M_AD x A x A - map the age groups reported with 5 y age group
    # first column indicates if country has death by age date (1 if yes), 2 column map the country to M_AD
    "map_country": np.int64,  # M x 2
    # case data by age
    "smoothed_logcases_weeks_n_max": int,
    "smoothed_logcases_weeks_n": np.int64,  # M - number of week indices per location
    # map of week indices to time indices
    "smoothed_logcases_week_map": np.int64,  # M x smoothed_logcases_weeks_n_max x 7
    # likelihood parameters for observed cases
    "smoothed_logcases_week_pars": np.float64,  # M x smoothed_logcases_weeks_n_max x 3
    # school case data
    "school_case_time_idx": np.int64,  # M x 2
    "school_case_data": np.float64,  # M x 4
    # school closure status
    "SCHOOL_STATUS": np.float64,  # N2 x M - school status, 1 if close, 0 if open
    # contact matrices
    "A_CHILD": int,  # number of age band for child
    "AGE_CHILD": np.int64,  # A_CHILD - age bands with child
    # min cntct_weekdays_mean and contact intensities during outbreak estimated in Zhang et al
    "cntct_school_closure_weekdays": np.float64,  # M x A x A
    # min cntct_weekends_mean and contact intensities during outbreak estimated in Zhang et al
    "cntct_school_closure_weekends": np.float64,  # M x A x A
    "cntct_elementary_school_reopening_weekdays": np.float64,  # M x A x A - contact matrix for school reopening
    "cntct_elementary_school_reopening_weekends": np.float64,  # M x A x A - contact matrix for school reopening
    # priors
    "cntct_weekdays_mean": np.float64,  # M x A x A - mean of prior contact rates between age groups on weekdays
    "cntct_weekends_mean": np.float64,  # M x A x A - mean of prior contact rates between age groups on weekends
    "hyperpara_ifr_age_lnmu": np.float64,  # A - hyper-parameters for probability of death in age band a log normal mean
    "hyperpara_ifr_age_lnsd": np.float64,  # A - hyper-parameters for probability of death in age band a log normal sd
    "rev_ifr_daysSinceInfection": np.float64,  # N2 - probability of death s days after infection in reverse order
    # fixed pre-calculated serial interval using empirical data from Neil in reverse order
    "rev_serial_interval": np.float64,  # SI_CUT
    "init_A": np.int64,  # N_init_A - age band in which initial cases occur in the first N0 days
}


# transform data before feeding into the model/mcmc
def transform_data(data):  # lines 438 -> 503
    data = data.copy()
    # given a dictionary of data, return a dictionary of data + transformed data
    cntct_weekdays_mean = data["cntct_weekdays_mean"].sum(-1)
    cntct_weekends_mean = data["cntct_weekends_mean"].sum(-1)
    popByAge_trans = data["popByAge"].T
    avg_cntct = (popByAge_trans * cntct_weekdays_mean).sum(-1) * (5. / 7.)  # M
    data["avg_cntct"] = avg_cntct + (popByAge_trans * cntct_weekends_mean).sum(-1) * (2. / 7.)
    # reported deaths -- the rows with i > N contain -1 and should be ignored
    data["trans_deaths"] = data["deaths"].T  # M x N2
    data["popByAge_abs"] = data["popByAge"].T * data["pop"][:, None]  # M x A

    # Extra transform code for NumPyro

    # shift index arrays by 1 because python starts index at 0
    data["init_A"] = data["init_A"] - 1
    data["epidemicStart"] = data["epidemicStart"] - 1
    data["dataByAgestart"] = data["dataByAgestart"] - 1
    data["wkend_idx"] = data["wkend_idx"] - 1
    data["school_case_time_idx"] = data["school_case_time_idx"] - 1
    data["AGE_CHILD"] = data["AGE_CHILD"] - 1
    data["upswing_timeeff_map"] = data["upswing_timeeff_map"] - 1

    # create epidemic_mask for indices from epidemicStart to dataByAgeStart or N
    epidemic_mask = np.full((data["M"], data["N2"]), False)
    for m, (start, byAge_start, end) in enumerate(zip(
            data["epidemicStart"], data["dataByAgestart"][data["map_country"][:, 1] - 1], data["N"])):
        if data["map_country"][m, 0] == 1:
            epidemic_mask[m, start:byAge_start] = True
        else:
            epidemic_mask[m, start:end] = True
    data["epidemic_mask"] = epidemic_mask

    # correct index_country_slice
    country_idx = data["map_country"][:, 1][data["map_country"][:, 0] == 1] - 1
    data["dataByAgestart"] = data["dataByAgestart"][country_idx]
    data["deathsByAge"] = data["deathsByAge"][:, :, country_idx]
    data["map_age"] = data["map_age"][country_idx]
    data["A_AD"] = data["A_AD"][country_idx]

    # create dataByAge_mask
    assert data["M_AD"] == sum(data["map_country"][:, 0])
    assert (data["map_country"][:, 1][data["map_country"][:, 0] == 1] - 1 == np.arange(data["M_AD"])).all()
    dataByAge_mask = np.full((data["M_AD"], data["N2"]), False)
    dataByAge_AD_mask = np.full((data["M_AD"], data["N2"], data["A"]), False)
    for m, (byAge_start, end, A_AD_local) in enumerate(zip(
            data["dataByAgestart"], data["N"][data["map_country"][:, 0] == 1], data["A_AD"])):
        dataByAge_mask[m, byAge_start:end] = True
        dataByAge_AD_mask[m, byAge_start:end, :A_AD_local] = True
    data["dataByAge_mask"] = dataByAge_mask
    data["dataByAge_AD_mask"] = dataByAge_AD_mask

    # convert wkend_idx (N2 x M) to wkend_mask (M x N2)
    wkend_mask = np.full((data["M"], data["N2"]), False)
    for m, num_wkend_idx in enumerate(data["WKEND_IDX_N"]):
        wkend_mask[m, data["wkend_idx"][:num_wkend_idx, m]] = True
    data["wkend_mask"] = wkend_mask

    # convert school_case_time_idx (M x 2) to school_case_time_mask (M x N2)
    start = data["school_case_time_idx"][:, None, 0]
    end = data["school_case_time_idx"][:, None, 1]
    time_slice = np.arange(data["N2"])
    school_case_time_mask = np.logical_and(start <= time_slice, time_slice <= end)
    data["school_case_time_mask"] = school_case_time_mask

    # replace -1. values by 1. to avoid NaN in cdf
    data["smoothed_logcases_week_pars"] = np.where(
        (data["smoothed_logcases_week_map"] > 0).all(-1, keepdims=True),
        data["smoothed_logcases_week_pars"],
        1.
    )
    data["school_case_data"] = np.where(
        data["school_case_time_mask"].all(-1), data["school_case_data"], 1.)
    return data


def transform_parameters(M, log_relsusceptibility_age_reduced, timeeff_shift_mid1):
    # transformed parameters
    log_relsusceptibility_age = jnp.concatenate(
        [
            jnp.repeat(log_relsusceptibility_age_reduced[0], 3),
            jnp.zeros(10),
            jnp.repeat(log_relsusceptibility_age_reduced[1], 5)
        ],
        axis=-1)  # A

    timeeff_shift_age = jnp.concatenate(
        [jnp.zeros((M, 4)), jnp.broadcast_to(timeeff_shift_mid1[:, None], (M, 6)), jnp.zeros((M, 8))],
        axis=-1)  # M x A
    return log_relsusceptibility_age, timeeff_shift_age


def model(data):  # lines 523 -> end
    # priors
    sd_dip_rnde = numpyro.sample("sd_dip_rnde", dist.Exponential(1.5))
    phi = numpyro.sample("phi", dist.HalfNormal(5))  # overdispersion parameter for likelihood model
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
        with numpyro.plate("N_IMP", data["N_IMP"]):
            upswing_timeeff_reduced = numpyro.sample(
                "upswing_timeeff_reduced",
                dist.ImproperUniform(dist.constraints.positive, (), ()))
        reparam_config = {k: TransformReparam() for k in [
            "dip_rnde", "log_ifr_age_rnde_mid1", "log_ifr_age_rnde_mid2", "log_ifr_age_rnde_old",
            "upswing_timeeff_reduced_base", "timeeff_shift_mid1"
        ]}
        reparam_config = {}
        with numpyro.handlers.reparam(config=reparam_config):
            dip_rnde = numpyro.sample("dip_rnde", dist.TransformedDistribution(
                dist.Normal(0., 1.), AffineTransform(0., sd_dip_rnde)))
            log_ifr_age_rnde_mid1 = numpyro.sample(
                "log_ifr_age_rnde_mid1", dist.TransformedDistribution(
                    dist.Exponential(1.),
                    AffineTransform(0., 1 / hyper_log_ifr_age_rnde_mid1, domain=dist.constraints.positive)
                ))
            log_ifr_age_rnde_mid2 = numpyro.sample(
                "log_ifr_age_rnde_mid2",
                dist.TransformedDistribution(
                    dist.Exponential(1.),
                    AffineTransform(0., 1 / hyper_log_ifr_age_rnde_mid2, domain=dist.constraints.positive)
                ))
            log_ifr_age_rnde_old = numpyro.sample(
                "log_ifr_age_rnde_old",
                dist.TransformedDistribution(
                    dist.Exponential(1.),
                    AffineTransform(0., 1 / hyper_log_ifr_age_rnde_old, domain=dist.constraints.positive)
                ))
            timeeff_shift_mid1 = numpyro.sample(
                "timeeff_shift_mid1",
                dist.TransformedDistribution(
                    dist.Exponential(1.),
                    AffineTransform(0., 1 / hyper_timeeff_shift_mid1, domain=dist.constraints.positive)
                ))

    numpyro.factor("upswing_timeeff_reduced_init_log_factor",
                   dist.HalfNormal(0.025).log_prob(upswing_timeeff_reduced[0]))
    numpyro.factor("upswing_timeeff_reduced_log_factor",
                   dist.TruncatedDistribution(dist.Normal(
                       upswing_timeeff_reduced[:-1], sd_upswing_timeeff_reduced), low=0.)
                   .log_prob(upswing_timeeff_reduced[1:]))

    with numpyro.plate("COVARIATES_Nm1", data["COVARIATES_N"] - 1):
        # regression coefficients for time varying multipliers on contacts
        beta = numpyro.sample("beta", dist.Normal(0., 1.))

    with numpyro.plate("A", data["A"]):
        # probability of death for age band a
        log_ifr_age_base = numpyro.sample(
            "log_ifr_age_base",
            dist.TruncatedDistribution(
                dist.Normal(data["hyperpara_ifr_age_lnmu"], data["hyperpara_ifr_age_lnsd"]),
                high=0.))

    log_relsusceptibility_age, timeeff_shift_age = transform_parameters(
        data["M"], log_relsusceptibility_age_reduced, timeeff_shift_mid1)
    numpyro.deterministic("log_relsusceptibility_age", log_relsusceptibility_age)
    numpyro.deterministic("timeeff_shift_age", timeeff_shift_age)

    countries_log_factor = countries_log_dens(
        data["trans_deaths"],
        # 0,
        # data["M"],
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
        # data["WKEND_IDX_N"],
        # data["wkend_idx"],
        data["wkend_mask"],
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
        # data["epidemicStart"],
        # data["N"],
        data["epidemic_mask"],
        data["N_init_A"],
        data["init_A"],
        # data["A_AD"],
        data["dataByAgestart"],
        data["dataByAge_mask"],
        data["dataByAge_AD_mask"],
        data["map_age"],
        data["deathsByAge"],
        data["map_country"],
        data["popByAge_abs"],
        # data["ones_vector_A"],
        data["smoothed_logcases_weeks_n"],
        data["smoothed_logcases_week_map"],
        data["smoothed_logcases_week_pars"],
        # data["school_case_time_idx"],
        data["school_case_time_mask"],
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
        for k, rv in r_df.items():
            if k not in data_def:
                continue
            if isinstance(rv, ro.vectors.ListVector):
                v = np.array([ro.conversion.rpy2py(x) for x in rv])
            else:
                v = ro.conversion.rpy2py(rv)
            dtype = data_def[k]
            if dtype in (int, float):
                data[k] = dtype(v.reshape(()))
            else:
                data[k] = v.astype(dtype)
    return data


data = get_data()
for k, v in data.items():
    assert isinstance(v, np.ndarray) or isinstance(v, (int, float))
