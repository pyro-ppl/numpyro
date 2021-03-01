# Ported from https://github.com/ImperialCollegeLondon/covid19model/blob/master/
# covid19AgeModel/inst/stan-models/covid19AgeModel_v120_cmdstanv.stan

import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer.reparam import TransformReparam  # noqa: F401

from functions import countries_log_dens


class UnnormalizedPositiveGRW(dist.GaussianRandomWalk):
    support = dist.constraints.independent(dist.constraints.positive, 1)


def model(data):
    # priors
    sd_dip_rnde = numpyro.sample("sd_dip_rnde", dist.Exponential(1.5))
    # alternatively, we can use HalfNormal for phi
    # but doing like this to get the same normalization constant as Stan
    phi = numpyro.sample("phi", dist.ImproperUniform(dist.constraints.positive, (), ()))
    numpyro.factor("phi_log_factor", dist.Normal(0., 5.).log_prob(phi))
    hyper_log_ifr_age_rnde_mid1 = numpyro.sample("hyper_log_ifr_age_rnde_mid1", dist.Exponential(.1))
    hyper_log_ifr_age_rnde_mid2 = numpyro.sample("hyper_log_ifr_age_rnde_mid2", dist.Exponential(.1))
    hyper_log_ifr_age_rnde_old = numpyro.sample("hyper_log_ifr_age_rnde_old", dist.Exponential(.1))
    log_relsusceptibility_age_reduced = numpyro.sample(
        "log_relsusceptibility_age_reduced",
        dist.Normal(jnp.array([-1.0702331, 0.3828269]), jnp.array([0.2169696, 0.1638433])))
    sd_upswing_timeeff_reduced = numpyro.sample("sd_upswing_timeeff_reduced", dist.LogNormal(-1.2, 0.2))
    hyper_timeeff_shift_mid1 = numpyro.sample("hyper_timeeff_shift_mid1", dist.Exponential(.1))
    impact_intv_children_effect = numpyro.sample("impact_intv_children_effect", dist.Uniform(0.1, 1.0).mask(False))
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
            # "dip_rnde",
            # "log_ifr_age_rnde_mid1",
            # "log_ifr_age_rnde_mid2",
            # "log_ifr_age_rnde_old",
            # "upswing_timeeff_reduced",
            # "timeeff_shift_mid1",
        ]}
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

    with numpyro.plate("COVARIATES_Nm1", data["COVARIATES_N"] - 1):
        # regression coefficients for time varying multipliers on contacts
        beta = numpyro.sample("beta", dist.Normal(0., 1.))

    with numpyro.plate("A", data["A"]):
        # probability of death for age band a
        # alternatively, we can use dist.TruncatedDistribution(dist.Normal(loc, scale), high=0.)
        log_ifr_age_base = numpyro.sample(
            "log_ifr_age_base",
            dist.ImproperUniform(dist.constraints.less_than(0), (), ()))

    numpyro.factor("log_ifr_age_base_log_factor",
                   dist.Normal(data["hyperpara_ifr_age_lnmu"], data["hyperpara_ifr_age_lnsd"])
                       .log_prob(log_ifr_age_base).sum())
    numpyro.factor("upswing_timeeff_reduced_init_log_factor",
                   # alternatively, we can use dist.HalfNormal(0.025)
                   dist.Normal(0., 0.025).log_prob(upswing_timeeff_reduced[0]))
    # FIXME: can we reparam upswing_timeeff_reduced?
    numpyro.factor("upswing_timeeff_reduced_log_factor",
                   dist.Normal(upswing_timeeff_reduced[:-1], sd_upswing_timeeff_reduced)
                       .log_prob(upswing_timeeff_reduced[1:]).sum())

    # transformed parameters
    log_relsusceptibility_age = numpyro.deterministic(
        "log_relsusceptibility_age",
        jnp.concatenate([jnp.repeat(log_relsusceptibility_age_reduced[0], 3),
                         jnp.zeros(10),
                         jnp.repeat(log_relsusceptibility_age_reduced[1], 5)],
                        axis=-1))  # A
    timeeff_shift_age = numpyro.deterministic(
        "timeeff_shift_age",
        jnp.concatenate([jnp.zeros((data["M"], 4)),
                         jnp.broadcast_to(timeeff_shift_mid1[:, None], (data["M"], 6)),
                         jnp.zeros((data["M"], 8))],
                        axis=-1))  # M x A

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
        data["school_case_time_idx"],
        data["school_case_data"],
        data["school_switch"]
    )
    numpyro.factor("countries_log_factor", countries_log_factor)
