# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import pytest

import jax
from jax import random
from jax.lax import stop_gradient
import jax.numpy as jnp

import numpyro as pyro
from numpyro import handlers, infer
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer.elbo import get_importance_log_probs
from numpyro.ops.indexing import Vindex

# put all funsor-related imports here, so test collection works without funsor
try:
    import funsor
    import numpyro.contrib.funsor
    from numpyro.contrib.funsor import config_enumerate

    funsor.set_backend("jax")
except ImportError:
    pytestmark = pytest.mark.skip(reason="funsor is not installed")

logger = logging.getLogger(__name__)

def assert_equal(a, b, prec=0):
    return jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b, atol=prec), a, b
    )


def model_0(data, params):
    with pyro.plate("data", len(data)):
        z = pyro.sample("z", dist.Categorical(jnp.array([0.3, 0.7])))
        pyro.sample("x", dist.Normal(z, 1), obs=data)


def guide_0(data, params):
    probs = pyro.param("probs", params["probs"], constraint=constraints.simplex)
    with pyro.plate("data", len(data)):
        pyro.sample("z", dist.Categorical(probs))


params_0 = {"probs": jnp.array([[0.4, 0.6], [0.5, 0.5]])}


#  def model_1(data):
#      a = pyro.sample("a", dist.Categorical(torch.tensor([0.3, 0.7])))
#      with pyro.plate("data", len(data)):
#          probs_b = torch.tensor([[0.1, 0.9], [0.2, 0.8]])
#          b = pyro.sample("b", dist.Categorical(probs_b[a.long()]))
#          pyro.sample("c", dist.Normal(b.to(data.dtype), 1), obs=data)
#
#
#  def guide_1(data):
#      probs_a = pyro.param(
#          "probs_a",
#          lambda: torch.tensor([0.5, 0.5]),
#      )
#      a = pyro.sample("a", dist.Categorical(probs_a))
#      with pyro.plate("data", len(data)) as idx:
#          probs_b = pyro.param(
#              "probs_b",
#              lambda: torch.tensor(
#                  [[[0.5, 0.5], [0.6, 0.4]], [[0.4, 0.6], [0.35, 0.65]]]
#              ),
#          )
#          pyro.sample("b", dist.Categorical(Vindex(probs_b)[a.long(), idx]))
#
#
#  def model_2(data):
#      prob_b = torch.tensor([[0.3, 0.7], [0.4, 0.6]])
#      prob_c = torch.tensor([[0.5, 0.5], [0.6, 0.4]])
#      prob_d = torch.tensor([[0.2, 0.8], [0.3, 0.7]])
#      prob_e = torch.tensor([[0.5, 0.5], [0.1, 0.9]])
#      a = pyro.sample("a", dist.Categorical(torch.tensor([0.3, 0.7])))
#      with pyro.plate("data", len(data)):
#          b = pyro.sample("b", dist.Categorical(prob_b[a.long()]))
#          c = pyro.sample("c", dist.Categorical(prob_c[b.long()]))
#          pyro.sample("d", dist.Categorical(prob_d[b.long()]))
#          pyro.sample("e", dist.Categorical(prob_e[c.long()]), obs=data)
#
#
#  def guide_2(data):
#      prob_a = pyro.param("prob_a", lambda: torch.tensor([0.5, 0.5]))
#      prob_b = pyro.param("prob_b", lambda: torch.tensor([[0.4, 0.6], [0.3, 0.7]]))
#      prob_c = pyro.param(
#          "prob_c",
#          lambda: torch.tensor([[[0.3, 0.7], [0.8, 0.2]], [[0.2, 0.8], [0.5, 0.5]]]),
#      )
#      prob_d = pyro.param(
#          "prob_d",
#          lambda: torch.tensor([[[0.2, 0.8], [0.9, 0.1]], [[0.1, 0.9], [0.4, 0.6]]]),
#      )
#      a = pyro.sample("a", dist.Categorical(prob_a))
#      with pyro.plate("data", len(data)) as idx:
#          b = pyro.sample("b", dist.Categorical(prob_b[a.long()]))
#          pyro.sample("c", dist.Categorical(Vindex(prob_c)[b.long(), idx]))
#          pyro.sample("d", dist.Categorical(Vindex(prob_d)[b.long(), idx]))


@pytest.mark.parametrize(
    "model,guide,params,data",
    [
        (model_0, guide_0, params_0, jnp.array([-0.5, 2.0])),
        #  (model_1, guide_1, torch.tensor([-0.5, 2.0])),
        #  (model_2, guide_2, torch.tensor([0.0, 1.0])),
    ],
)
def test_gradient(model, guide, params, data):

    # Expected grads based on exact integration
    elbo = infer.TraceEnum_ELBO(
        max_plate_nesting=1,  # set this to ensure rng agrees across runs
    )
    expected_loss_fn = lambda params: elbo.loss(
        random.PRNGKey(0), {}, model, config_enumerate(guide), data, params
    )
    expected_loss, expected_grads = jax.value_and_grad(expected_loss_fn)(params)

    # Actual grads averaged over num_particles
    elbo = infer.Trace_ELBO(
        num_particles=50000,
    )
    actual_loss_fn = lambda params: elbo.loss(
        random.PRNGKey(0), {}, model, guide, data, params
    )
    actual_loss, actual_grads = jax.value_and_grad(actual_loss_fn)(params)

    for name in sorted(params):
        logger.info("expected {} = {}".format(name, expected_grads[name]))
        logger.info("actual   {} = {}".format(name, actual_grads[name]))

    assert_equal(actual_grads, expected_grads, prec=0.02)


#  def test_particle_gradient_0():
#      # model
#      # +---------+
#      # | z --> x |
#      # +---------+
#      #
#      # guide
#      # +---+
#      # | z |
#      # +---+
#      data = jnp.array([-0.5, 2.0])
#      params = {"rate": jnp.array([3.5, 1.5])}
#
#      def model(params):
#          with pyro.plate("data", len(data)):
#              z = pyro.sample("z", dist.Poisson(3))
#              pyro.sample("x", dist.Normal(z, 1), obs=data)
#
#      def guide(params):
#          rate = pyro.param("rate", params["rate"])
#          with pyro.plate("data", len(data)):
#              pyro.sample("z", dist.Poisson(rate))
#
#      elbo = infer.Trace_ELBO(num_particles=1)
#      # Trace_ELBO gradients
#      actual_loss_fn = lambda params: elbo.loss(
#          random.PRNGKey(0), {}, model, guide, params
#      )
#      actual_loss, actual_grads = jax.value_and_grad(actual_loss_fn)(params)
#
#      # Hand derived gradients
#      # elbo = MonteCarlo(
#      #   [q(z_i) * log_pz_i].sum(i)
#      #   + [q(z_i) * log_px_i].sum(i)
#      #   - [q(z_i) * log_qx_i].sum(i)
#      # )
#      # log factors
#      def expected_loss_fn(params):
#          model_seed, guide_seed = random.split(random.PRNGKey(0))
#          seeded_model = handlers.seed(model, model_seed)
#          seeded_guide = handlers.seed(guide, guide_seed)
#          model_probs, guide_probs = get_importance_log_probs(
#              seeded_model, seeded_guide, (params,), {}, {}
#          )
#          logpx = model_probs["x"]
#          logpz = model_probs["z"]
#          logqz = guide_probs["z"]
#          # dice factor
#          df_z = jnp.exp(logqz - stop_gradient(logqz))
#          # dice elbo
#          dice_elbo = jnp.sum(df_z * (logpz + logpx - logqz))
#          return -dice_elbo
#
#      # backward run
#      expected_loss, expected_grads = jax.value_and_grad(expected_loss_fn)(params)
#
#      for name in sorted(params):
#          logger.info("expected {} = {}".format(name, expected_grads[name]))
#          logger.info("actual   {} = {}".format(name, actual_grads[name]))
#
#      assert_equal(actual_grads, expected_grads, prec=1e-4)


#  @pyroapi.pyro_backend("contrib.funsor")
#  def test_particle_gradient_1():
#      # model
#      #    +-----------+
#      # a -|-> b --> c |
#      #    +-----------+
#      #
#      # guide
#      #    +-----+
#      # a -|-> b |
#      #    +-----+
#      data = torch.tensor([-0.5, 2.0])
#
#      def model():
#          a = pyro.sample("a", dist.Bernoulli(0.3))
#          with pyro.plate("data", len(data)):
#              rate = torch.tensor([2.0, 3.0])
#              b = pyro.sample("b", dist.Poisson(rate[a.long()]))
#              pyro.sample("c", dist.Normal(b, 1), obs=data)
#
#      def guide():
#          # set this to ensure rng agrees across runs
#          # this should be ok since we are comparing a single particle gradients
#          pyro.set_rng_seed(0)
#          prob = pyro.param(
#              "prob",
#              lambda: torch.tensor(0.5),
#          )
#          a = pyro.sample("a", dist.Bernoulli(prob))
#          with pyro.plate("data", len(data)):
#              rate = pyro.param("rate", lambda: torch.tensor([[3.5, 1.5], [0.5, 2.5]]))
#              pyro.sample("b", dist.Poisson(rate[a.long()]))
#
#      elbo = infer.Trace_ELBO(
#          max_plate_nesting=1,  # set this to ensure rng agrees across runs
#          num_particles=1,
#          strict_enumeration_warning=False,
#      )
#
#      # Trace_ELBO gradients
#      pyro.clear_param_store()
#      elbo.loss_and_grads(model, guide)
#      params = dict(pyro.get_param_store().named_parameters())
#      actual_grads = {name: param.grad.detach().cpu() for name, param in params.items()}
#
#      # Hand derived gradients
#      # elbo = MonteCarlo(
#      #   q(a) * log_pa
#      #   + q(a) * [q(b_i|a) * log_pb_i].sum(i)
#      #   + q(a) * [q(b_i|a) * log_pc_i].sum(i)
#      #   - q(a) * log_qa
#      #   - q(a) * [q(b_i|a) * log_qb_i].sum(i)
#      # )
#      pyro.clear_param_store()
#      guide_tr = handlers.trace(guide).get_trace()
#      model_tr = handlers.trace(handlers.replay(model, guide_tr)).get_trace()
#      guide_tr.compute_log_prob()
#      model_tr.compute_log_prob()
#      # log factors
#      logpa = model_tr.nodes["a"]["log_prob"]
#      logpb = model_tr.nodes["b"]["log_prob"]
#      logpc = model_tr.nodes["c"]["log_prob"]
#      logqa = guide_tr.nodes["a"]["log_prob"]
#      logqb = guide_tr.nodes["b"]["log_prob"]
#      # dice factors
#      df_a = (logqa - logqa.detach()).exp()
#      df_b = (logqb - logqb.detach()).exp()
#      # dice elbo
#      dice_elbo = (
#          df_a * logpa
#          + df_a * (df_b * logpb).sum()
#          + df_a * (df_b * logpc).sum()
#          - df_a * logqa
#          - df_a * (df_b * logqb).sum()
#      )
#      # backward run
#      loss = -dice_elbo
#      loss.backward()
#      params = dict(pyro.get_param_store().named_parameters())
#      expected_grads = {name: param.grad.detach().cpu() for name, param in params.items()}
#
#      for name in sorted(params):
#          logger.info("expected {} = {}".format(name, expected_grads[name]))
#          logger.info("actual   {} = {}".format(name, actual_grads[name]))
#
#      assert_equal(actual_grads, expected_grads, prec=1e-4)
#
#
#  @pyroapi.pyro_backend("contrib.funsor")
#  def test_particle_gradient_2():
#      # model
#      #    +-----------------+
#      # a -|-> b --> c --> e |
#      #    |    \--> d       |
#      #    +-----------------+
#      #
#      # guide
#      #    +-----------+
#      # a -|-> b --> c |
#      #    |    \--> d |
#      #    +-----------+
#      data = torch.tensor([0.0, 1.0])
#
#      def model():
#          prob_b = torch.tensor([0.3, 0.4])
#          prob_c = torch.tensor([0.5, 0.6])
#          prob_d = torch.tensor([0.2, 0.3])
#          prob_e = torch.tensor([0.5, 0.1])
#          a = pyro.sample("a", dist.Bernoulli(0.3))
#          with pyro.plate("data", len(data)):
#              b = pyro.sample("b", dist.Bernoulli(prob_b[a.long()]))
#              c = pyro.sample("c", dist.Bernoulli(prob_c[b.long()]))
#              pyro.sample("d", dist.Bernoulli(prob_d[b.long()]))
#              pyro.sample("e", dist.Bernoulli(prob_e[c.long()]), obs=data)
#
#      def guide():
#          # set this to ensure rng agrees across runs
#          # this should be ok since we are comparing a single particle gradients
#          pyro.set_rng_seed(0)
#          prob_a = pyro.param("prob_a", lambda: torch.tensor(0.5))
#          prob_b = pyro.param("prob_b", lambda: torch.tensor([0.4, 0.3]))
#          prob_c = pyro.param("prob_c", lambda: torch.tensor([[0.3, 0.8], [0.2, 0.5]]))
#          prob_d = pyro.param("prob_d", lambda: torch.tensor([[0.2, 0.9], [0.1, 0.4]]))
#          a = pyro.sample("a", dist.Bernoulli(prob_a))
#          with pyro.plate("data", len(data)) as idx:
#              b = pyro.sample("b", dist.Bernoulli(prob_b[a.long()]))
#              pyro.sample("c", dist.Bernoulli(Vindex(prob_c)[b.long(), idx]))
#              pyro.sample("d", dist.Bernoulli(Vindex(prob_d)[b.long(), idx]))
#
#      elbo = infer.Trace_ELBO(
#          max_plate_nesting=1,  # set this to ensure rng agrees across runs
#          num_particles=1,
#          strict_enumeration_warning=False,
#      )
#
#      # Trace_ELBO gradients
#      pyro.clear_param_store()
#      elbo.loss_and_grads(model, guide)
#      params = dict(pyro.get_param_store().named_parameters())
#      actual_grads = {name: param.grad.detach().cpu() for name, param in params.items()}
#
#      # Hand derived gradients
#      # elbo = MonteCarlo(
#      #   q(a) * log_pa
#      #   + q(a) * [q(b_i|a) * log_pb_i].sum(i)
#      #   + q(a) * [q(b_i|a) * q(c_i|b_i) * log_pc_i].sum(i)
#      #   + q(a) * [q(b_i|a) * q(c_i|b_i) * log_pe_i].sum(i)
#      #   + q(a) * [q(b_i|a) * q(d_i|b_i) * log_pd_i].sum(i)
#      #   - q(a) * log_qa
#      #   - q(a) * [q(b_i|a) * log_qb_i].sum(i)
#      #   - q(a) * [q(b_i|a) * q(c_i|b_i) * log_qc_i].sum(i)
#      #   - q(a) * [q(b_i|a) * q(d_i|b_i) * log_qd_i].sum(i)
#      # )
#      pyro.clear_param_store()
#      guide_tr = handlers.trace(guide).get_trace()
#      model_tr = handlers.trace(handlers.replay(model, guide_tr)).get_trace()
#      guide_tr.compute_log_prob()
#      model_tr.compute_log_prob()
#      # log factors
#      logpa = model_tr.nodes["a"]["log_prob"]
#      logpb = model_tr.nodes["b"]["log_prob"]
#      logpc = model_tr.nodes["c"]["log_prob"]
#      logpd = model_tr.nodes["d"]["log_prob"]
#      logpe = model_tr.nodes["e"]["log_prob"]
#
#      logqa = guide_tr.nodes["a"]["log_prob"]
#      logqb = guide_tr.nodes["b"]["log_prob"]
#      logqc = guide_tr.nodes["c"]["log_prob"]
#      logqd = guide_tr.nodes["d"]["log_prob"]
#      # dice factors
#      df_a = (logqa - logqa.detach()).exp()
#      df_b = (logqb - logqb.detach()).exp()
#      df_c = (logqc - logqc.detach()).exp()
#      df_d = (logqd - logqd.detach()).exp()
#      # dice elbo
#      dice_elbo = (
#          df_a * logpa
#          + df_a * (df_b * logpb).sum()
#          + df_a * (df_b * df_c * logpc).sum()
#          + df_a * (df_b * df_c * logpe).sum()
#          + df_a * (df_b * df_d * logpd).sum()
#          - df_a * logqa
#          - df_a * (df_b * logqb).sum()
#          - df_a * (df_b * df_c * logqc).sum()
#          - df_a * (df_b * df_d * logqd).sum()
#      )
#      # backward run
#      loss = -dice_elbo
#      loss.backward()
#      params = dict(pyro.get_param_store().named_parameters())
#      expected_grads = {name: param.grad.detach().cpu() for name, param in params.items()}
#
#      for name in sorted(params):
#          logger.info("expected {} = {}".format(name, expected_grads[name]))
#          logger.info("actual   {} = {}".format(name, actual_grads[name]))
#
#      assert_equal(actual_grads, expected_grads, prec=1e-4)
#
#
#  @pyroapi.pyro_backend("contrib.funsor")
#  def test_particle_gradient_3():
#      # model
#      #    +-----------------+
#      # a -|-> b --> c --> d |
#      #    +-----------------+
#      #
#      # guide (b is enumerated)
#      #    +-----------+
#      # a -|-> b --> c |
#      #    +-----------+
#      data = torch.tensor([0.0, 1.0])
#
#      def model():
#          prob_b = torch.tensor([[0.3, 0.7], [0.4, 0.6]])
#          prob_c = torch.tensor([0.5, 0.6])
#          prob_d = torch.tensor([0.5, 0.1])
#          a = pyro.sample("a", dist.Bernoulli(0.3))
#          with pyro.plate("data", len(data)):
#              b = pyro.sample("b", dist.Categorical(prob_b[a.long()]))
#              c = pyro.sample("c", dist.Bernoulli(prob_c[b.long()]))
#              pyro.sample("d", dist.Bernoulli(prob_d[c.long()]), obs=data)
#
#      def guide():
#          # set this to ensure rng agrees across runs
#          # this should be ok since we are comparing a single particle gradients
#          pyro.set_rng_seed(0)
#          prob_a = pyro.param("prob_a", lambda: torch.tensor(0.5))
#          prob_b = pyro.param("prob_b", lambda: torch.tensor([[0.4, 0.6], [0.3, 0.7]]))
#          prob_c = pyro.param("prob_c", lambda: torch.tensor([[0.3, 0.8], [0.2, 0.5]]))
#          a = pyro.sample("a", dist.Bernoulli(prob_a))
#          with pyro.plate("data", len(data)) as idx:
#              b = pyro.sample(
#                  "b", dist.Categorical(prob_b[a.long()]), infer={"enumerate": "parallel"}
#              )
#              pyro.sample("c", dist.Bernoulli(Vindex(prob_c)[b.long(), idx]))
#
#      elbo = infer.Trace_ELBO(
#          max_plate_nesting=1,  # set this to ensure rng agrees across runs
#          num_particles=1,
#          strict_enumeration_warning=False,
#      )
#
#      # Trace_ELBO gradients
#      pyro.clear_param_store()
#      elbo.loss_and_grads(model, guide)
#      params = dict(pyro.get_param_store().named_parameters())
#      actual_grads = {name: param.grad.detach().cpu() for name, param in params.items()}
#
#      # Hand derived gradients (b is exactly integrated)
#      # elbo = MonteCarlo(
#      #   q(a) * log_pa
#      #   + q(a) * [q(b_i|a) * log_pb_i].sum(i, b)
#      #   + q(a) * [q(b_i|a) * q(c_i|b_i) * log_pc_i].sum(i, b)
#      #   + q(a) * [q(b_i|a) * q(c_i|b_i) * log_pd_i].sum(i, b)
#      #   - q(a) * log_qa
#      #   - q(a) * [q(b_i|a) * log_qb_i].sum(i)
#      #   - q(a) * [q(b_i|a) * q(c_i|b_i) * log_qc_i].sum(i, b)
#      # )
#      pyro.clear_param_store()
#      with handlers.enum(first_available_dim=(-2)), handlers.provenance():
#          guide_tr = handlers.trace(guide).get_trace()
#          model_tr = handlers.trace(handlers.replay(model, guide_tr)).get_trace()
#      guide_tr.compute_log_prob()
#      model_tr.compute_log_prob()
#      # log factors
#      logpa = model_tr.nodes["a"]["log_prob"]
#      logpb = model_tr.nodes["b"]["log_prob"]
#      logpc = model_tr.nodes["c"]["log_prob"]
#      logpd = model_tr.nodes["d"]["log_prob"]
#
#      logqa = guide_tr.nodes["a"]["log_prob"]
#      logqb = guide_tr.nodes["b"]["log_prob"]
#      logqc = guide_tr.nodes["c"]["log_prob"]
#      # dice factors
#      df_a = (logqa - logqa.detach()).exp()
#      qb = logqb.exp()
#      df_c = (logqc - logqc.detach()).exp()
#      # dice elbo
#      dice_elbo = (
#          df_a * logpa
#          + df_a * (qb * logpb).sum()
#          + df_a * (qb * df_c * logpc).sum()
#          + df_a * (qb * df_c * logpd).sum()
#          - df_a * logqa
#          - df_a * (qb * logqb).sum()
#          - df_a * (qb * df_c * logqc).sum()
#      )
#      # backward run
#      loss = -dice_elbo
#      loss.backward()
#      params = dict(pyro.get_param_store().named_parameters())
#      expected_grads = {name: param.grad.detach().cpu() for name, param in params.items()}
#
#      for name in sorted(params):
#          logger.info("expected {} = {}".format(name, expected_grads[name]))
#          logger.info("actual   {} = {}".format(name, actual_grads[name]))
#
#      assert_equal(actual_grads, expected_grads, prec=1e-4)
#
#
#  @pyroapi.pyro_backend("contrib.funsor")
#  def test_particle_gradient_4():
#      # model
#      #    +-----------------+
#      # a -|-> b --> c --> d |
#      #    +-----------------+
#      #
#      # guide (c is enumerated)
#      #    +-----------+
#      # a -|-> b --> c |
#      #    +-----------+
#      data = torch.tensor([0.0, 1.0])
#
#      def model():
#          prob_b = torch.tensor([[0.3, 0.7], [0.4, 0.6]])
#          prob_c = torch.tensor([[0.5, 0.5], [0.6, 0.4]])
#          prob_d = torch.tensor([0.5, 0.1])
#          a = pyro.sample("a", dist.Bernoulli(0.3))
#          with pyro.plate("data", len(data)):
#              b = pyro.sample("b", dist.Categorical(prob_b[a.long()]))
#              c = pyro.sample("c", dist.Categorical(prob_c[b.long()]))
#              pyro.sample("d", dist.Bernoulli(prob_d[c.long()]), obs=data)
#
#      def guide():
#          # set this to ensure rng agrees across runs
#          # this should be ok since we are comparing a single particle gradients
#          pyro.set_rng_seed(0)
#          prob_a = pyro.param("prob_a", lambda: torch.tensor(0.5))
#          prob_b = pyro.param("prob_b", lambda: torch.tensor([[0.4, 0.6], [0.3, 0.7]]))
#          prob_c = pyro.param(
#              "prob_c",
#              lambda: torch.tensor([[[0.3, 0.7], [0.8, 0.2]], [[0.2, 0.8], [0.5, 0.5]]]),
#          )
#          a = pyro.sample("a", dist.Bernoulli(prob_a))
#          with pyro.plate("data", len(data)) as idx:
#              b = pyro.sample("b", dist.Categorical(prob_b[a.long()]))
#              pyro.sample(
#                  "c",
#                  dist.Categorical(Vindex(prob_c)[b.long(), idx]),
#                  infer={"enumerate": "parallel"},
#              )
#
#      elbo = infer.Trace_ELBO(
#          max_plate_nesting=1,  # set this to ensure rng agrees across runs
#          num_particles=1,
#          strict_enumeration_warning=False,
#      )
#
#      # Trace_ELBO gradients
#      pyro.clear_param_store()
#      elbo.loss_and_grads(model, guide)
#      params = dict(pyro.get_param_store().named_parameters())
#      actual_grads = {name: param.grad.detach().cpu() for name, param in params.items()}
#
#      # Hand derived gradients (c is exactly integrated)
#      # elbo = MonteCarlo(
#      #   q(a) * log_pa
#      #   + q(a) * [q(b_i|a) * log_pb_i].sum(i)
#      #   + q(a) * [q(b_i|a) * q(c_i|b_i) * log_pc_i].sum(i, c)
#      #   + q(a) * [q(b_i|a) * q(c_i|b_i) * log_pd_i].sum(i, c)
#      #   - q(a) * log_qa
#      #   - q(a) * [q(b_i|a) * log_qb_i].sum(i)
#      #   - q(a) * [q(b_i|a) * q(c_i|b_i) * log_qc_i].sum(i, c)
#      # )
#      pyro.clear_param_store()
#      with handlers.enum(first_available_dim=(-2)), handlers.provenance():
#          guide_tr = handlers.trace(guide).get_trace()
#          model_tr = handlers.trace(handlers.replay(model, guide_tr)).get_trace()
#      guide_tr.compute_log_prob()
#      model_tr.compute_log_prob()
#      # log factors
#      logpa = model_tr.nodes["a"]["log_prob"]
#      logpb = model_tr.nodes["b"]["log_prob"]
#      logpc = model_tr.nodes["c"]["log_prob"]
#      logpd = model_tr.nodes["d"]["log_prob"]
#
#      logqa = guide_tr.nodes["a"]["log_prob"]
#      logqb = guide_tr.nodes["b"]["log_prob"]
#      logqc = guide_tr.nodes["c"]["log_prob"]
#      # dice factors
#      df_a = (logqa - logqa.detach()).exp()
#      df_b = (logqb - logqb.detach()).exp()
#      qc = logqc.exp()
#      # dice elbo
#      dice_elbo = (
#          df_a * logpa
#          + df_a * (df_b * logpb).sum()
#          + df_a * (df_b * qc * logpc).sum()
#          + df_a * (df_b * qc * logpd).sum()
#          - df_a * logqa
#          - df_a * (df_b * logqb).sum()
#          - df_a * (df_b * qc * logqc).sum()
#      )
#      # backward run
#      loss = -dice_elbo
#      loss.backward()
#      params = dict(pyro.get_param_store().named_parameters())
#      expected_grads = {name: param.grad.detach().cpu() for name, param in params.items()}
#
#      for name in sorted(params):
#          logger.info("expected {} = {}".format(name, expected_grads[name]))
#          logger.info("actual   {} = {}".format(name, actual_grads[name]))
#
#      assert_equal(actual_grads, expected_grads, prec=1e-4)
#
#
#  @pyroapi.pyro_backend("contrib.funsor")
#  def test_particle_gradient_5():
#      # model
#      #    +-----------------+
#      # a -|-> b --> c --> e |
#      #    |    \--> d       |
#      #    +-----------------+
#      #
#      # guide (b is enumerated)
#      #    +-----------+
#      # a -|-> b --> c |
#      #    |    \--> d |
#      #    +-----------+
#      data = torch.tensor([0.0, 1.0])
#
#      def model():
#          prob_b = torch.tensor([[0.3, 0.7], [0.4, 0.6]])
#          prob_c = torch.tensor([0.5, 0.6])
#          prob_d = torch.tensor([0.2, 0.3])
#          prob_e = torch.tensor([0.5, 0.1])
#          a = pyro.sample("a", dist.Bernoulli(0.3))
#          with pyro.plate("data", len(data)):
#              b = pyro.sample("b", dist.Categorical(prob_b[a.long()]))
#              c = pyro.sample("c", dist.Bernoulli(prob_c[b.long()]))
#              pyro.sample("d", dist.Bernoulli(prob_d[b.long()]))
#              pyro.sample("e", dist.Bernoulli(prob_e[c.long()]), obs=data)
#
#      def guide():
#          # set this to ensure rng agrees across runs
#          # this should be ok since we are comparing a single particle gradients
#          pyro.set_rng_seed(0)
#          prob_a = pyro.param("prob_a", lambda: torch.tensor(0.5))
#          prob_b = pyro.param("prob_b", lambda: torch.tensor([[0.4, 0.6], [0.3, 0.7]]))
#          prob_c = pyro.param("prob_c", lambda: torch.tensor([[0.3, 0.8], [0.2, 0.5]]))
#          prob_d = pyro.param("prob_d", lambda: torch.tensor([[0.2, 0.9], [0.1, 0.4]]))
#          a = pyro.sample("a", dist.Bernoulli(prob_a))
#          with pyro.plate("data", len(data)) as idx:
#              b = pyro.sample(
#                  "b", dist.Categorical(prob_b[a.long()]), infer={"enumerate": "parallel"}
#              )
#              pyro.sample("c", dist.Bernoulli(Vindex(prob_c)[b.long(), idx]))
#              pyro.sample("d", dist.Bernoulli(Vindex(prob_d)[b.long(), idx]))
#
#      elbo = infer.Trace_ELBO(
#          max_plate_nesting=1,  # set this to ensure rng agrees across runs
#          num_particles=1,
#          strict_enumeration_warning=False,
#      )
#
#      # Trace_ELBO gradients
#      pyro.clear_param_store()
#      elbo.loss_and_grads(model, guide)
#      params = dict(pyro.get_param_store().named_parameters())
#      actual_grads = {name: param.grad.detach().cpu() for name, param in params.items()}
#
#      # Hand derived gradients (b exactly integrated out)
#      # elbo = MonteCarlo(
#      #   q(a) * log_pa
#      #   + q(a) * [q(b_i|a) * log_pb_i].sum(i, b)
#      #   + q(a) * [q(b_i|a) * q(c_i|b_i) * log_pc_i].sum(i, b)
#      #   + q(a) * [q(b_i|a) * q(c_i|b_i) * log_pe_i].sum(i, b)
#      #   + q(a) * [q(b_i|a) * q(d_i|b_i) * log_pd_i].sum(i, b)
#      #   - q(a) * log_qa
#      #   - q(a) * [q(b_i|a) * log_qb_i].sum(i, b)
#      #   - q(a) * [q(b_i|a) * q(c_i|b_i) * log_qc_i].sum(i, b)
#      #   - q(a) * [q(b_i|a) * q(d_i|b_i) * log_qd_i].sum(i, b)
#      # )
#      pyro.clear_param_store()
#      with handlers.enum(first_available_dim=(-2)), handlers.provenance():
#          guide_tr = handlers.trace(guide).get_trace()
#          model_tr = handlers.trace(handlers.replay(model, guide_tr)).get_trace()
#      guide_tr.compute_log_prob()
#      model_tr.compute_log_prob()
#      # log factors
#      logpa = model_tr.nodes["a"]["log_prob"]
#      logpb = model_tr.nodes["b"]["log_prob"]
#      logpc = model_tr.nodes["c"]["log_prob"]
#      logpd = model_tr.nodes["d"]["log_prob"]
#      logpe = model_tr.nodes["e"]["log_prob"]
#
#      logqa = guide_tr.nodes["a"]["log_prob"]
#      logqb = guide_tr.nodes["b"]["log_prob"]
#      logqc = guide_tr.nodes["c"]["log_prob"]
#      logqd = guide_tr.nodes["d"]["log_prob"]
#      # dice factors
#      df_a = (logqa - logqa.detach()).exp()
#      qb = logqb.exp()
#      df_c = (logqc - logqc.detach()).exp()
#      df_d = (logqd - logqd.detach()).exp()
#      # dice elbo
#      dice_elbo = (
#          df_a * logpa
#          + df_a * (qb * logpb).sum()
#          + df_a * (qb * df_c * logpc).sum()
#          + df_a * (qb * df_c * logpe).sum()
#          + df_a * (qb * df_d * logpd).sum()
#          - df_a * logqa
#          - df_a * (qb * logqb).sum()
#          - df_a * (qb * df_c * logqc).sum()
#          - df_a * (qb * df_d * logqd).sum()
#      )
#      # backward run
#      loss = -dice_elbo
#      loss.backward()
#      params = dict(pyro.get_param_store().named_parameters())
#      expected_grads = {name: param.grad.detach().cpu() for name, param in params.items()}
#
#      for name in sorted(params):
#          logger.info("expected {} = {}".format(name, expected_grads[name]))
#          logger.info("actual   {} = {}".format(name, actual_grads[name]))
#
#      assert_equal(actual_grads, expected_grads, prec=1e-4)
