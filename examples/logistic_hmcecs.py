""" Logistic regression model as implemetned in https://arxiv.org/pdf/1708.00955.pdf with Higgs Dataset """
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, Predictive
import sys, os
from jax.config import config
import datetime
sys.path.append('/home/lys/Dropbox/PhD/numpyro/numpyro/contrib/')
sys.path.append('/home/lys/Dropbox/PhD/numpyro/numpyro/examples/')

from hmcecs import HMC
#from numpyro.contrib.hmcecs import HMC

from sklearn.datasets import load_breast_cancer
from datasets import _load_higgs
numpyro.set_platform("cpu")

# TODO: import Higgs data! ---> http://archive.ics.uci.edu/ml/machine-learning-databases/00280/
# https://towardsdatascience.com/identifying-higgs-bosons-from-background-noise-pyspark-d7983234207e

def model(feats, obs):
    """  Logistic regression model

    """
    n, m = feats.shape
    #precision = numpyro.sample('precision', dist.continuous.Uniform(1, 4))
    precision = 0.5
    theta = numpyro.sample('theta', dist.continuous.Normal(jnp.zeros(m), precision * jnp.ones(m)))

    numpyro.sample('obs', dist.Bernoulli(logits=jnp.matmul(feats, theta)), obs=obs)


def infer_nuts(rng_key, feats, obs, samples=5, warmup=0, ):
    kernel = NUTS(model=model)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples)
    mcmc.run(rng_key, feats, obs)
    # mcmc.print_summary()
    return mcmc.get_samples()


def infer_hmcecs(rng_key, feats, obs, m=50,g=20,samples=10, warmup=5, ):
    hmcecs_key, map_key = jax.random.split(rng_key)
    n, _ = feats.shape



    print("Running NUTS for map estimation")
    z_map = {key: value.mean(0) for key, value in infer_nuts(map_key, feats, obs).items()}

    #Observations = (569,1)
    #Features = (569,31)
    print("Running MCMC subsampling")
    kernel = HMC(model=model,z_ref=z_map,m=m,g=g,subsample_method="perturb")
    mcmc = MCMC(kernel,num_warmup=warmup,num_samples=samples)
    mcmc.run(rng_key,feats,obs)
    return mcmc.get_samples()



def breast_cancer_data():
    dataset = load_breast_cancer()
    feats = dataset.data
    feats = (feats - feats.mean(0)) / feats.std(0)
    feats = jnp.hstack((feats, jnp.ones((feats.shape[0], 1))))

    return feats[:500], dataset.target[:500]


def higgs_data():
    observations,features = _load_higgs()
    return observations[:1000],features[:1000]


def Plot(samples):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import time

    for sample in [0,7,15,25]:
        plt.figure(sample)

        #samples = pd.DataFrame.from_records(samples,index="theta")
        sns.kdeplot(data=samples["theta"][sample])
        plt.xlabel(r"$\theta")
        plt.ylabel("Density")
        plt.title(r"$\theta$ {} Density plot".format(sample))
        plt.savefig("{}/KDE_plot_theta_{}.png".format("PLOTS_{}".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss")),sample))


def Folders(folder_name):
    """ Folder for all the generated images It will updated everytime!!! Save the previous folder before running again. Creates folder in current directory"""
    import os
    import shutil
    basepath = os.getcwd()
    if not basepath:
        newpath = folder_name
    else:
        newpath = basepath + "/%s" % folder_name

    if not os.path.exists(newpath):
        try:
            original_umask = os.umask(0)
            os.makedirs(newpath, 0o777)
        finally:
            os.umask(original_umask)
    else:
        shutil.rmtree(newpath)  # removes all the subdirectories!
        os.makedirs(newpath,0o777)

if __name__ == '__main__':
    rng_key = jax.random.PRNGKey(37)
    rng_key, feat_key, obs_key = jax.random.split(rng_key, 3)


    #feats, obs = breast_cancer_data()
    feats,obs = higgs_data()

    now = datetime.datetime.now()
    Folders("PLOTS_{}".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss")))
    config.update('jax_disable_jit', True)
    est_posterior = infer_hmcecs(rng_key, feats=feats, obs=obs, m =50,g=20)

    exit()
    predictions = Predictive(model, posterior_samples=est_posterior)(rng_key, feats, None)['obs']

    # for i, y in enumerate(obs):
    #     print(i, y[0], jnp.sum(predictions[i]) > 50)