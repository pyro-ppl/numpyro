import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

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
    data["elementary_school_reopening_idx"] = data["elementary_school_reopening_idx"] - 1
    data["smoothed_logcases_week_map"] = data["smoothed_logcases_week_map"] - 1

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

    # replace -1. values by 1. to avoid NaN in cdf
    data["smoothed_logcases_week_pars"] = np.where(
        data["smoothed_logcases_week_pars"] > 0,
        data["smoothed_logcases_week_pars"],
        1.
    )
    return data


def get_data():
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


def test_get_data():
    data = get_data()
    for k, v in data.items():
        assert isinstance(v, np.ndarray) or isinstance(v, (int, float))
