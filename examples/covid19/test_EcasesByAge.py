import jax

import numpy as np

import jax.numpy as jnp

import numpyro

from functions import country_EcasesByAge


def EcasesByAge_vmap(
    # parameters
    R0: float,
    e_cases_N0: float,
    log_relsusceptibility_age: jnp.float32,  # A
    impact_intv_children_effect: float,
    impact_intv_onlychildren_effect: float,
    impact_intv: jnp.float32,  # N2 x A
    # data
    N0: int,
    elementary_school_reopening_idx: int,  # M
    N2: int,
    SCHOOL_STATUS: jnp.float32,  # M N2
    A: int,
    A_CHILD: int,
    SI_CUT: int,
    wkend_mask,  # boolean array  M N0-N0
    avg_cntct: float,  # M
    cntct_weekends_mean: jnp.float32,  # M x A x A
    cntct_weekdays_mean: jnp.float32,  # M x A x A
    cntct_school_closure_weekends: jnp.float32,  # M x A x A
    cntct_school_closure_weekdays: jnp.float32,  # M x A x A
    cntct_elementary_school_reopening_weekends: jnp.float32,  # M x A x A
    cntct_elementary_school_reopening_weekdays: jnp.float32,  # M x A x A
    rev_serial_interval: jnp.float32,  # SI_CUT
    popByAge_abs: jnp.float32,  # M A
    N_init_A: int,
    init_A: jnp.int32,  # N_init_A
    school_switch):

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
        R0, e_cases_N0, impact_intv, elementary_school_reopening_idx, SCHOOL_STATUS, wkend_mask,
        avg_cntct, cntct_weekends_mean, cntct_weekdays_mean, cntct_school_closure_weekends,
        cntct_school_closure_weekdays, cntct_elementary_school_reopening_weekends,
        cntct_elementary_school_reopening_weekdays, popByAge_abs, school_switch
    )

    return E_casesByAge


if __name__ == '__main__':
    numpyro.enable_x64()

    NUM_TESTS = 4

    M = 3
    A = 6
    A_CHILD = 2
    N_init_A = 2
    init_A = [2, 4]

    for test in range(NUM_TESTS):
        N0 = 4 + 4 * test // 2
        N2 = 12 + 2 * test // 2
        SI_CUT = 9 + test

        R0 = np.random.rand(M)
        e_cases_N0 = np.random.rand(M)
        log_relsusceptibility_age = np.random.randn(A)
        impact_intv_children_effect = np.random.rand(1).item()
        impact_intv_onlychildren_effect = np.random.rand(1).item()
        impact_intv = np.random.rand(M, N2, A)
        elementary_school_reopening_idx = np.random.randint(N0 + 2, N0 + 6, M)
        SCHOOL_STATUS = np.array(np.random.randint(0, 2, (M, N2)), dtype=np.float32)
        avg_cntct = np.random.rand(M)
        wkend_mask = np.array(np.random.randint(0, 2, (M, N2 - N0)), dtype=np.bool)
        cntct_weekends_mean = np.random.rand(M, A, A)
        cntct_weekdays_mean = np.random.rand(M, A, A)
        cntct_school_closure_weekends = np.random.rand(M, A, A)
        cntct_school_closure_weekdays = np.random.rand(M, A, A)
        cntct_elementary_school_reopening_weekends = np.random.rand(M, A, A)
        cntct_elementary_school_reopening_weekdays = np.random.rand(M, A, A)
        rev_serial_interval = np.random.rand(SI_CUT)
        popByAge_abs = 12.3 * np.random.rand(M, A)

        school_switch = 2 * SCHOOL_STATUS[:, N0:].astype(np.int32) + \
            (np.arange(N0, N2) >= elementary_school_reopening_idx[:, None]).astype(np.int32)

        value_vmap = EcasesByAge_vmap(
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

        print("value_vmap",value_vmap.shape)
        assert value_vmap.shape == (M, N2, A)
