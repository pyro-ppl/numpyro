""" Logistic regression model as implemetned in https://arxiv.org/pdf/1708.00955.pdf with Higgs Dataset """
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, Predictive
import sys, os
from jax.config import config
import datetime,time

sys.path.append('/home/lys/Dropbox/PhD/numpyro/numpyro/contrib/')
sys.path.append('/home/lys/Dropbox/PhD/numpyro/numpyro/examples/')

from hmcecs import HMC
#from numpyro.contrib.hmcecs import HMC
#numpyro.set_host_device_count(2)
from sklearn.datasets import load_breast_cancer
from datasets import _load_higgs
import jax.numpy as np_jax
numpyro.set_platform("cpu")

# TODO: import Higgs data! ---> http://archive.ics.uci.edu/ml/machine-learning-databases/00280/
# https://towardsdatascience.com/identifying-higgs-bosons-from-background-noise-pyspark-d7983234207e

def model(feats, obs):
    """  Logistic regression model

    """
    n, m = feats.shape
    precision = numpyro.sample('precision', dist.continuous.Uniform(1, 4))
    #precision = 0.5
    theta = numpyro.sample('theta', dist.continuous.Normal(jnp.zeros(m), precision * jnp.ones(m)))

    numpyro.sample('obs', dist.Bernoulli(logits=jnp.matmul(feats, theta)), obs=obs)


def infer_nuts(rng_key, feats, obs, samples=10, warmup=5, ):
    kernel = NUTS(model=model)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples)
    mcmc.run(rng_key, feats, obs)
    # mcmc.print_summary()
    return mcmc.get_samples()


def infer_hmcecs(rng_key, feats, obs, m=None,g=None,samples=10, warmup=5,algo="NUTS",subsample_method=None ):
    hmcecs_key, map_key = jax.random.split(rng_key)
    n, _ = feats.shape
    print("Using {} samples".format(str(samples+warmup)))


    print("Running NUTS for map estimation")
    if subsample_method=="perturb":
        z_map = {key: value.mean(0) for key, value in infer_nuts(map_key, feats, obs).items()}
    else:
        z_map = None
    print("Running MCMC subsampling")
    start = time.time()
    kernel = HMC(model=model,z_ref=z_map,m=m,g=g,algo=algo,subsample_method=subsample_method)

    mcmc = MCMC(kernel,num_warmup=warmup,num_samples=samples,num_chains=1)
    mcmc.run(rng_key,feats,obs)
    stop = time.time()
    file_hyperparams = open("PLOTS_{}/Hyperparameters_{}.txt".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss"),now.strftime("%Y_%m_%d_%Hh%Mmin%Ss")), "a")
    file_hyperparams.write('MCMC/NUTS elapsed time {}: {} \n'.format(subsample_method,time.time() - start))
    file_hyperparams.write('Effective size {}: {}\n'.format(subsample_method,samples))
    file_hyperparams.write('Warm up size {}: {}\n'.format(subsample_method,warmup))
    file_hyperparams.write('Subsample size (m): {}\n'.format(m))
    file_hyperparams.write('Block size (g): {}\n'.format(g))

    file_hyperparams.close()

    save_obj(mcmc.get_samples(),"{}/MCMC_Dict_Samples_{}.pkl".format("PLOTS_{}".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss")),subsample_method))

    return mcmc.get_samples()



def breast_cancer_data():
    dataset = load_breast_cancer()
    feats = dataset.data
    feats = (feats - feats.mean(0)) / feats.std(0)
    feats = jnp.hstack((feats, jnp.ones((feats.shape[0], 1))))

    return feats[:50], dataset.target[:50]


def higgs_data():
    observations,features = _load_higgs()
    return features[:10],observations[:10]
def save_obj(obj, name):
    import _pickle as cPickle
    import bz2
    with bz2.BZ2File(name, "wb") as f:
        cPickle.dump(obj, f)

def Plot(samples_ECS,samples_NUTS):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import time

    for sample in [0,7,15,25]:
        plt.figure(sample)

        #samples = pd.DataFrame.from_records(samples,index="theta")
        sns.kdeplot(data=samples_ECS["theta"][sample],color="r",label="ECS")
        sns.kdeplot(data=samples_NUTS["theta"][sample],color="b",label="NUTS")

        plt.xlabel(r"$\theta")
        plt.ylabel("Density")
        plt.legend()
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
    m = int(np_jax.sqrt(obs.shape[0])*2)
    g= int(m//3)
    est_posterior_ECS = infer_hmcecs(rng_key, feats=feats, obs=obs, m =m,g=g,algo="NUTS",subsample_method="perturb")
    est_posterior_NUTS = infer_hmcecs(rng_key, feats=feats, obs=obs, m =m,g=g,algo="NUTS")

    Plot(est_posterior_ECS,est_posterior_NUTS)
    exit()
    predictions = Predictive(model, posterior_samples=est_posterior_ECS)(rng_key, feats, None)['obs']

