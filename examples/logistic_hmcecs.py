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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from numpyro.diagnostics import summary
from jax.tree_util import tree_flatten,tree_map

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


def infer_nuts(rng_key, feats, obs, samples, warmup ):
    kernel = NUTS(model=model)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples)
    mcmc.run(rng_key, feats, obs)
    #mcmc.print_summary()
    samples = mcmc.get_samples()
    samples = tree_map(lambda x: x[None, ...], samples)
    r_hat_average = np_jax.sum(summary(samples)["theta"]["r_hat"])/len(summary(samples)["theta"]["r_hat"])

    return mcmc.get_samples(), r_hat_average


def infer_hmcecs(rng_key, feats, obs, m=None,g=None,n_samples=None, warmup=None,algo="NUTS",subsample_method=None ):
    hmcecs_key, map_key = jax.random.split(rng_key)
    n, _ = feats.shape
    print("Using {} samples".format(str(n_samples+warmup)))


    print("Running NUTS for map estimation")
    if subsample_method=="perturb":
        samples,r_hat_average = infer_nuts(map_key, feats, obs,samples=15,warmup=5)
        z_map = {key: value.mean(0) for key, value in samples.items()}
    else:
        z_map = None
    print("Running MCMC subsampling")
    start = time.time()
    kernel = HMC(model=model,z_ref=z_map,m=m,g=g,algo=algo,subsample_method=subsample_method)

    mcmc = MCMC(kernel,num_warmup=warmup,num_samples=n_samples,num_chains=1)
    mcmc.run(rng_key,feats,obs)
    stop = time.time()
    file_hyperparams = open("PLOTS_{}/Hyperparameters_{}.txt".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss"),now.strftime("%Y_%m_%d_%Hh%Mmin%Ss")), "a")
    file_hyperparams.write('MCMC/NUTS elapsed time {}: {} \n'.format(subsample_method,time.time() - start))
    file_hyperparams.write('Effective size {}: {}\n'.format(subsample_method,n_samples))
    file_hyperparams.write('Warm up size {}: {}\n'.format(subsample_method,warmup))
    file_hyperparams.write('Subsample size (m): {}\n'.format(m))
    file_hyperparams.write('Block size (g): {}\n'.format(g))
    file_hyperparams.write('Data size (n): {}\n'.format(feats.shape[0]))
    file_hyperparams.close()

    save_obj(mcmc.get_samples(),"{}/MCMC_Dict_Samples_{}.pkl".format("PLOTS_{}".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss")),subsample_method))

    return mcmc.get_samples()



def breast_cancer_data():
    dataset = load_breast_cancer()
    feats = dataset.data
    feats = (feats - feats.mean(0)) / feats.std(0)
    feats = jnp.hstack((feats, jnp.ones((feats.shape[0], 1))))

    return feats[:100], dataset.target[:100]


def higgs_data():
    observations,features = _load_higgs()
    return features,observations
def save_obj(obj, name):
    import _pickle as cPickle
    import bz2
    with bz2.BZ2File(name, "wb") as f:
        cPickle.dump(obj, f)

def determine_best_sample_size(rng_key,feats,obs):
    """Determine amount of effective sample size for z_map initialization"""
    effective_sample_list=[5,10,20,30,50]
    r_hat_average_list=[]
    for effective_sample in effective_sample_list:
        samples, r_hat_average = infer_nuts(rng_key,feats,obs,effective_sample,warmup=6)
        r_hat_average_list.append(r_hat_average)

    plt.plot(effective_sample_list,r_hat_average_list)
    plt.xlabel(r"Effective sample size")
    plt.ylabel(r"$\hat{r}$")
    plt.title("Determine best effective sample size for z_map")
    plt.savefig("{}/Best_effective_size_z_map.png".format("PLOTS_{}".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss"))))


def Plot(samples_ECS,samples_NUTS):


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


    #determine_best_sample_size(rng_key,feats[:100],obs[:100])

    m = int(np_jax.sqrt(obs.shape[0])*2)
    g= int(m//3)
    est_posterior_ECS = infer_hmcecs(rng_key, feats=feats[:100], obs=obs[:100],n_samples=100,warmup=50, m =m,g=g,algo="NUTS",subsample_method="perturb")
    est_posterior_NUTS = infer_hmcecs(rng_key, feats=feats[:100], obs=obs[:100], n_samples=100,warmup=50,m =m,g=g,algo="NUTS")

    Plot(est_posterior_ECS,est_posterior_NUTS)
    exit()
    predictions = Predictive(model, posterior_samples=est_posterior_ECS)(rng_key, feats, None)['obs']

