from re import L
import jax 
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist 
from numpyro import sample
from numpyro.contrib.control_flow import scan
from numpyro.infer import Predictive, NUTS, MCMC


class NUTModelTrainer(object):
    """ 
    basically a wrapper class that trains a model 
    using NUT and MCMC. We can come back and add different 
    types of trainers afterwards. 
    """
    def __init__(self, rng_seed = 1000):
        self.rng_key = jax.random.PRNGKey(seed=rng_seed)
        self.trained = False

    def model(self, data=None):
        raise NotImplementedError
    
    def fit(self, data, num_steps=1000, num_warmup_steps=1000):
        self.sampler = NUTS(self.model)
        self.mcmc = MCMC(self.sampler, 
                    num_warmup=num_warmup_steps, 
                    num_samples=num_steps)
        self.mcmc.run(rng_key=self.rng_key, data=data) 
        self.trained = True

    def predict(self, data, num_samples=100): 
        posterior = Predictive(self.model, num_samples=num_samples)
        posterior_sample = posterior(self.rng_key)
        return posterior_sample



class StateSpaceModel(NUTModelTrainer):
    """ 
    with more structure to help run state space models. 
    """
    def __init__(self):
        self.params = {} 
        super(StateSpaceModel, self).__init__()

    def param_initialization(self):
        raise NotImplementedError

    def generate_initial_values(self, data):
        raise NotImplementedError
    
    def transition_function(self):
        raise NotImplementedError

    def model(self, data=None):
        self.param_initialization()
        init, data_obs = self.generate_initial_values(data)
        time_steps = jnp.arange(data_obs.shape[0])

        with numpyro.handlers.condition(data={"y": data_obs}):
            scan(self.transition_function, init, time_steps)


# two ways to define a model: 
# method 1: directly write the model function. 
# method 2: write initialization, transition components, 
#           and let StateSpaceModel take care of the rest

# method 1: 
#
# good: if user is familiar with numpyro already, this is pretty simple 
#       in terms of code flow. easy to extend. 
#
# bad: you have to handle some quirkyness like `condition`, `scan` etc. 
class AR1Model(NUTModelTrainer):
    def model(self, data=None): 
        """ a simple AR1 model """
        a1 = sample("a1", dist.Normal())
        sigma = sample("sigma", dist.Exponential())

        def transition(carry, _):
            y_prev = carry[0]
            m_t = a1 * y_prev
            y_t = sample("y", dist.Normal(m_t, sigma))
            return (y_t, ), None

        timesteps = jnp.arange(data.shape[0]-1)
        init = (0, )
        with numpyro.handlers.condition(data={"y": data[1:]}):
            scan(transition, init, timesteps)


# method 2: 
# 
# good: we manage to hide away a lot of quirky functions. easier 
#       for new users to understand
#
# bad: more convoluted. 
class AR1Model2(StateSpaceModel):
    def param_initialization(self):
        self.params['a1'] = sample("a1", dist.Normal())
        self.params['sigma'] = sample("sigma", dist.Exponential())

    def generate_initial_values(self, data):
        init = (data[0], )
        data_obs = data[1:]
        return (init, data_obs)

    def transition_function(self, carry, _):
        y_prev = carry[0]
        m_t = self.params['a1'] * y_prev
        y_t = sample("y", dist.Normal(m_t, self.params['sigma']))
        return (y_t, ), None


if __name__=='__main__':
    rng_key = jax.random.PRNGKey(100)
    n = 1000 
    tmp = jax.random.normal(rng_key, (10000, ))
    data = tmp[:(-1)] + tmp[1:]

    ar1_model = AR1Model()
    ar1_model.fit(data)

    ar1_model_2 = AR1Model2()
    ar1_model_2.fit(data)