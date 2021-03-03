import jax
import jax.numpy as jnp
import jax.ops as ops
from jax.lax import scan, cond, switch

import numpy as np

import jax.numpy as jnp

import numpyro
from numpyro.util import identity

from functions2 import country_EcasesByAge


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


def EcasesByAge(
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

    cntct_mean = jnp.stack([cntct_weekends_mean, cntct_weekdays_mean])
    cntct_school_closure = jnp.stack([cntct_school_closure_weekends, cntct_school_closure_weekdays])
    cntct_elementary_school_reopening = jnp.stack([cntct_elementary_school_reopening_weekends,
                                                   cntct_elementary_school_reopening_weekdays])
    print("cntct_elementary_school_reopening",cntct_elementary_school_reopening.shape)

    # probability of infection given contact in location m
    rho0 = R0 / avg_cntct
    rev_serial_interval = rev_serial_interval * rho0[:, None]  # M SI_CUT

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

        prop_susceptibleByAge = 1.0 - E_casesByAge_sum / popByAge_abs  # M A
        prop_susceptibleByAge = jnp.maximum(0.0, prop_susceptibleByAge)

        # this convolution is effectively zero padded on the left
        tmp_row_vector_A_no_impact_intv = (rev_serial_interval[:, None] @ E_casesByAge_SI_CUT)[:, 0]  # M S @ M S A => M A
        assert tmp_row_vector_A_no_impact_intv.shape == (M, A)
        tmp_row_vector_A = impact_intv_t * tmp_row_vector_A_no_impact_intv  # M A
        assert tmp_row_vector_A.shape == (M, A)

        # choose weekend/weekday contact matrices
        cntct_mean_t = jnp.take_along_axis(cntct_mean, weekend_t[None, :, None, None], 0)[0]  # M A A
        cntct_school_closure_t = jnp.take_along_axis(cntct_school_closure, weekend_t[None, :, None, None], 0)[0]
        cntct_elementary_school_reopening_t = jnp.take_along_axis(cntct_elementary_school_reopening, weekend_t[None, :, None, None], 0)[0]
        assert cntct_mean_t.shape == (M, A, A)
        assert cntct_school_closure_t.shape == (M, A, A)
        assert cntct_elementary_school_reopening_t.shape == (M, A, A)

        # school open
        col1_left_so = tmp_row_vector_A_no_impact_intv[:, :, None]
        col1_right_so = cntct_mean_t[:, :, :A_CHILD]
        col2_topleft_so = tmp_row_vector_A_no_impact_intv[:, :A_CHILD, None]
        col2_topright_so = cntct_mean_t[:, :A_CHILD, A_CHILD:]
        col2_bottomleft_so = tmp_row_vector_A[:, A_CHILD:, None]
        col2_bottomright_so = cntct_mean_t[:, A_CHILD:, A_CHILD:]
        impact_intv_adult_so = impact_intv_t[:, A_CHILD:]

        # school reopen
        col1_left_sr = (tmp_row_vector_A * impact_intv_children_onlychildren_padded)[:, :, None]
        col1_right_sr = cntct_elementary_school_reopening_t[:, :, :A_CHILD] * impact_intv_children_effect
        col2_topleft_sr = tmp_row_vector_A[:, :A_CHILD, None] * impact_intv_children_effect
        col2_topright_sr = cntct_elementary_school_reopening_t[:, :A_CHILD, A_CHILD:] * impact_intv_t[:, None, A_CHILD:]
        col2_bottomleft_sr = tmp_row_vector_A[:, A_CHILD:, None]
        col2_bottomright_sr = cntct_elementary_school_reopening_t[:, A_CHILD:, A_CHILD:] * impact_intv_t[:, None, A_CHILD:]
        impact_intv_adult_sr = jnp.ones((M, A - A_CHILD))

        # school closed
        col1_left_sc = tmp_row_vector_A_no_impact_intv[:, :, None]
        col1_right_sc = cntct_school_closure_t[:, :, :A_CHILD]
        col2_topleft_sc = tmp_row_vector_A_no_impact_intv[:, :A_CHILD, None]
        col2_topright_sc = cntct_school_closure_t[:, :A_CHILD, A_CHILD:]
        col2_bottomleft_sc = tmp_row_vector_A[:, A_CHILD:, None]
        col2_bottomright_sc = cntct_school_closure_t[:, A_CHILD:, A_CHILD:]
        impact_intv_adult_sc = impact_intv_t[:, A_CHILD:]

        # combine ingredients in preparation for selecting on school_switch_t
        col1_left = jnp.stack([col1_left_so, col1_left_sr, col1_left_sc])
        col1_right = jnp.stack([col1_right_so, col1_right_sr, col1_right_sc])
        col2_topleft = jnp.stack([col2_topleft_so, col2_topleft_sr, col2_topleft_sc])
        col2_topright = jnp.stack([col2_topright_so, col2_topright_sr, col2_topright_sc])
        col2_bottomleft = jnp.stack([col2_bottomleft_so, col2_bottomleft_sr, col2_bottomleft_sc])
        col2_bottomright = jnp.stack([col2_bottomright_so, col2_bottomright_sr, col2_bottomright_sc])
        impact_intv_adult = jnp.stack([impact_intv_adult_so, impact_intv_adult_sr, impact_intv_adult_sc])

        # select on school_switch_t
        col1_left = jnp.take_along_axis(col1_left, school_switch_t[None, :, None, None], 0)[0]  # M A 1
        col1_right = jnp.take_along_axis(col1_right, school_switch_t[None, :, None, None], 0)[0]  # M A AC
        col2_topleft = jnp.take_along_axis(col2_topleft, school_switch_t[None, :, None, None], 0)[0]  # M AC 1
        col2_topright = jnp.take_along_axis(col2_topright, school_switch_t[None, :, None, None], 0)[0]  # M AC A-AC
        col2_bottomleft = jnp.take_along_axis(col2_bottomleft, school_switch_t[None, :, None, None], 0)[0]  # M A-AC 1
        col2_bottomright = jnp.take_along_axis(col2_bottomright, school_switch_t[None, :, None, None], 0)[0]  # M A-AC A-AC
        impact_intv_adult = jnp.take_along_axis(impact_intv_adult, school_switch_t[None, :, None], 0)[0]  # M A-AC

        assert col1_left.shape == (M, A, 1)
        assert col1_right.shape == (M, A, A_CHILD)
        assert col2_topleft.shape == (M, A_CHILD, 1)
        assert col2_topright.shape == (M, A_CHILD, A - A_CHILD)
        assert col2_bottomleft.shape == (M, A - A_CHILD, 1)
        assert col2_bottomright.shape == (M, A - A_CHILD, A - A_CHILD)
        assert impact_intv_adult.shape == (M, A - A_CHILD)

        col1 = (jnp.moveaxis(col1_left, -1, -2) @ col1_right)[:, 0]  # M AC
        col2 = (jnp.moveaxis(col2_topleft, -1, -2) @ col2_topright)[:, 0] +\
            (jnp.moveaxis(col2_bottomleft, -1, -2) @ col2_bottomright)[:, 0] * impact_intv_adult  # M A-AC

        assert col1.shape == (M, A_CHILD)
        assert col2.shape == (M, A - A_CHILD)

        E_casesByAge_t = jnp.concatenate([col1, col2], axis=-1)
        E_casesByAge_t *= prop_susceptibleByAge * relsusceptibility_age

        # add term to cumulative sum
        E_casesByAge_sum = E_casesByAge_sum + E_casesByAge_t
        # basically "roll left and append most recent time slice at right"
        E_casesByAge_SI_CUT = jnp.concatenate([E_casesByAge_SI_CUT[:, 1:], E_casesByAge_t[:, None, :]], axis=-2)

        return (E_casesByAge_sum, E_casesByAge_SI_CUT), E_casesByAge_t

    # init expected cases by age and location in first N0 days
    E_casesByAge_init = ops.index_update(jnp.zeros((M, N0, A)), ops.index[:, :N0, init_A], e_cases_N0[:, None, None] / N_init_A,
                                         indices_are_sorted=True, unique_indices=True)

    # initialize carry variables
    E_casesByAge_sum = E_casesByAge_init.sum(-2)  # M A
    # pad with zeros on left  =>  (M SI_CUT A)
    E_casesByAge_SI_CUT = jnp.concatenate([jnp.zeros((M, SI_CUT - N0, A)), E_casesByAge_init], axis=-2)

    init = (E_casesByAge_sum, E_casesByAge_SI_CUT)
    xs = (jnp.moveaxis(impact_intv[:, N0:], 0, 1), (wkend_mask).astype(jnp.int32).T, school_switch.T)

    # execute for loop using JAX primitive scan
    E_casesByAge_N0_N2 = scan(scan_body, init, xs, length=N2 - N0)[1]
    E_casesByAge_N0_N2 = jnp.moveaxis(E_casesByAge_N0_N2, 0, 1)
    E_casesByAge = jnp.concatenate([E_casesByAge_init, E_casesByAge_N0_N2], axis=-2)

    return E_casesByAge


if __name__ == '__main__':
    numpyro.enable_x64()

    NUM_TESTS = 1

    M = 3
    A = 6
    A_CHILD = 2
    N_init_A = 2
    init_A = [2, 4]

    for test in range(NUM_TESTS):
        N0 = 4 + 4 * test // 2
        N2 = 12 + 2 * test // 2
        SI_CUT = 9 + test

        print("N0 N2 M", N0, N2, M)

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

        value_custom = EcasesByAge(
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

        print("value_custom",value_vmap.shape)
        assert value_custom.shape == (M, N2, A)

        delta = value_custom - value_vmap
        max_delta = np.max(np.fabs(delta))

        print("[Test {}] Max delta: {:.2e}".format(test, max_delta))
        assert max_delta < 1.0e-12

