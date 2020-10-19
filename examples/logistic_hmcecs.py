""" Logistic regression model as implemetned in https://arxiv.org/pdf/1708.00955.pdf with Higgs Dataset """
#!/usr/bin/env python
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC, Predictive
import sys, os
from jax.config import config
import datetime,time
import argparse
import numpy as np
from numpyro.distributions.kl import kl_divergence
from matplotlib.pyplot import cm

sys.path.append('/home/lys/Dropbox/PhD/numpyro/numpyro/contrib/')
sys.path.append('/home/lys/Dropbox/PhD/numpyro/numpyro/examples/')

from hmcecs import HMC
#from numpyro.contrib.hmcecs import HMC

from sklearn.datasets import load_breast_cancer
#from datasets import _load_higgs
from numpyro.examples.datasets import _load_higgs
from logistic_hmcecs_svi import svi_map
import jax.numpy as np_jax
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from numpyro.diagnostics import summary
from jax.tree_util import tree_flatten,tree_map

numpyro.set_platform("cpu")

def breast_cancer_data():
    dataset = load_breast_cancer()
    feats = dataset.data
    feats = (feats - feats.mean(0)) / feats.std(0)
    feats = jnp.hstack((feats, jnp.ones((feats.shape[0], 1))))

    return feats, dataset.target


def higgs_data():
    observations,features = _load_higgs()
    return features,observations
def save_obj(obj, name):
    import _pickle as cPickle
    import bz2
    with bz2.BZ2File(name, "wb") as f:
        cPickle.dump(obj, f)
def load_obj(name):
    import _pickle as cPickle
    import bz2
    data = bz2.BZ2File(name, "rb")
    data = cPickle.load(data)

    return data

def model(feats, obs):
    """  Logistic regression model

    """
    n, m = feats.shape
    theta = numpyro.sample('theta', dist.continuous.Normal(jnp.zeros(m), 2 * jnp.ones(m)))

    numpyro.sample('obs', dist.Bernoulli(logits=jnp.matmul(feats, theta)), obs=obs)

def infer_nuts(rng_key, feats, obs, samples, warmup ):
    kernel = NUTS(model=model,target_accept_prob=0.8)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples)
    mcmc.run(rng_key, feats, obs)
    #mcmc.print_summary()
    samples = mcmc.get_samples()
    samples = tree_map(lambda x: x[None, ...], samples)
    r_hat_average = np_jax.sum(summary(samples)["theta"]["r_hat"])/len(summary(samples)["theta"]["r_hat"])

    return mcmc.get_samples(), r_hat_average




def infer_hmc(rng_key, feats, obs, samples, warmup ):
    kernel = HMC(model=model,target_accept_prob=0.8)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples)
    mcmc.run(rng_key, feats, obs)
    #mcmc.print_summary()
    samples = mcmc.get_samples()
    samples = tree_map(lambda x: x[None, ...], samples)
    r_hat_average = np_jax.sum(summary(samples)["theta"]["r_hat"])/len(summary(samples)["theta"]["r_hat"])

    return mcmc.get_samples(), r_hat_average





def infer_hmcecs(rng_key, feats, obs, m=None,g=None,n_samples=None, warmup=None,algo="NUTS",subsample_method=None,map_method=None,proxy="taylor",num_epochs=None ):
    hmcecs_key, map_key = jax.random.split(rng_key)
    n, _ = feats.shape
    file_hyperparams = open("PLOTS_{}/Hyperparameters_{}.txt".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms")), "a")
    if subsample_method=="perturb" and proxy== "taylor":
        map_samples = 100
        map_warmup = 50
        factor_NUTS = 1000
        if map_method == "NUTS":
            print("Running NUTS for map estimation {} + {} samples".format(map_samples,map_warmup))
            file_hyperparams.write('MAP samples : {} \n'.format(map_samples))
            file_hyperparams.write('MAP warmup : {} \n'.format(map_warmup))
            samples,r_hat_average = infer_nuts(map_key, feats[:factor_NUTS], obs[:factor_NUTS],samples=map_samples,warmup=map_warmup)
            z_ref = {key: value.mean(0) for key, value in samples.items()}
        if map_method == "HMC":
            print("Running HMC for map estimation")
            file_hyperparams.write('MAP samples : {} \n'.format(map_samples))
            file_hyperparams.write('MAP warmup : {} \n'.format(map_warmup))
            samples, r_hat_average = infer_hmc(map_key, feats[:factor_NUTS], obs[:factor_NUTS], samples=map_samples, warmup=map_warmup)
            z_ref = {key: value.mean(0) for key, value in samples.items()}

        if map_method == "SVI":
            print("Running SVI for map estimation")
            file_hyperparams.write('SVI epochs : {} \n'.format(num_epochs))
            z_ref = svi_map(model, map_key, feats=feats, obs=obs,num_epochs=num_epochs,batch_size = m)
            z_ref = {k[5:]: v for k, v in z_ref.items()} #highlight: [5:] is to skip the "auto" part
        svi = None
        save_obj(z_ref,"{}/MAP_Dict_Samples_MAP_{}.pkl".format("PLOTS_{}".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms")), map_method))
        print("Running MCMC subsampling with Taylor proxy")
    elif subsample_method =="perturb" and proxy=="svi":
        factor_SVI = obs.shape[0]
        batch_size = int(factor_SVI//10)
        print("Running SVI for map estimation")
        file_hyperparams.write('SVI epochs : {} \n'.format(num_epochs))
        map_key, post_key = jax.random.split(map_key)
        z_ref, svi, svi_state = svi_map(model, map_key, feats=feats[:factor_SVI], obs=obs[:factor_SVI],
                                        num_epochs=num_epochs, batch_size=batch_size)
        z_ref = svi.guide.sample_posterior(post_key, svi.get_params(svi_state), (100,))
        z_ref = {name: value.mean(0) for name, value in z_ref.items()} #highlight: AutoDiagonalNormal does not have auto_ in front of the parmeters

        save_obj(z_ref,"{}/MAP_Dict_Samples_Proxy_{}.pkl".format("PLOTS_{}".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms")),
                                                           proxy))
        print("Running MCMC subsampling with SVI proxy")

    else:
        z_ref = None
        svi = None

    start = time.time()
    kernel = HMC(model=model,z_ref=z_ref,m=m,g=g,algo=algo,subsample_method=subsample_method,proxy=proxy,svi_fn=svi,target_accept_prob=0.8)

    mcmc = MCMC(kernel,num_warmup=warmup,num_samples=n_samples,num_chains=1)
    mcmc.run(rng_key,feats,obs)
    stop = time.time()
    file_hyperparams.write('MCMC/NUTS elapsed time {}: {} \n'.format(subsample_method,time.time() - start))
    file_hyperparams.write('Effective size {}: {}\n'.format(subsample_method,n_samples))
    file_hyperparams.write('Warm up size {}: {}\n'.format(subsample_method,warmup))
    file_hyperparams.write('Subsample size (m): {}\n'.format(m))
    file_hyperparams.write('Block size (g): {}\n'.format(g))
    file_hyperparams.write('Data size (n): {}\n'.format(feats.shape[0]))
    file_hyperparams.write('...........................................\n')
    file_hyperparams.close()

    save_obj(mcmc.get_samples(),"{}/MCMC_Dict_Samples_{}_m_{}.pkl".format("PLOTS_{}".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms")),subsample_method,m))

    return mcmc.get_samples()



def Determine_best_sample_size(rng_key,feats,obs):
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
    plt.savefig("{}/Best_effective_size_z_map.png".format("PLOTS_{}".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"))))
def Plot(samples_ECS,samples_NUTS,ecs_algo,algo,proxy,m,kl=None):
    for sample in [0,7,15,25]:
        plt.figure(sample + m +3)
        #samples = pd.DataFrame.from_records(samples,index="theta")
        sns.kdeplot(data=samples_ECS["theta"][sample],color="r",label="ECS-{}-{} proxy".format(ecs_algo,proxy))
        sns.kdeplot(data=samples_NUTS["theta"][sample],color="b",label="{}".format(algo))
        #if kl != None:
        #        sns.kdeplot(data=kl, color="g", label="KL; m: {}".format(m))
        plt.xlabel(r"$\theta")
        plt.ylabel("Density")
        plt.legend()
        plt.title(r"$\theta$ {} m: {} Density plot".format(sample,str(m)))
        plt.savefig("{}/KDE_plot_theta_{}_m_{}.png".format("PLOTS_{}".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms")),sample,str(m)))
        plt.clf()
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
def Plot_KL(map_method,ecs_algo,algo,n_samples,n_warmup,epochs):
    factor_ECS=obs.shape[0]
    m = [int(np_jax.sqrt(obs[:factor_ECS].shape[0])),2*int(np_jax.sqrt(obs[:factor_ECS].shape[0])),4*int(np_jax.sqrt(obs[:factor_ECS].shape[0])),8*int(np_jax.sqrt(obs[:factor_ECS].shape[0]))]
    g = 5
    factor_NUTS = 1000
    colors = cm.rainbow(np.linspace(0, 1, len(m)))
    est_posterior_NUTS = infer_hmcecs(rng_key, feats=feats[:factor_NUTS], obs=obs[:factor_NUTS],
                                      n_samples=n_samples, warmup=n_warmup, m="all", g=g, algo=algo)
    for m_val, color in zip(m,colors):
        est_posterior_ECS = infer_hmcecs(rng_key, feats=feats[:factor_ECS], obs=obs[:factor_ECS],
                                         n_samples=n_samples,
                                         warmup=n_warmup,
                                         m=m_val, g=g,
                                         algo=ecs_algo,
                                         subsample_method="perturb",
                                         map_method=map_method,
                                         num_epochs=epochs)

        p = dist.Normal(est_posterior_ECS["theta"])
        q = dist.Normal(est_posterior_NUTS["theta"])
        kl = kl_divergence(p, q)
        Plot(est_posterior_ECS, est_posterior_NUTS, ecs_algo, algo,m_val,kl=kl )
def Tests(map_method,ecs_algo,algo,n_samples,n_warmup,epochs,proxy):
    m = int(np_jax.sqrt(obs.shape[0])*2)
    g= 5
    est_posterior_ECS = infer_hmcecs(rng_key, feats=feats, obs=obs,
                                     n_samples=n_samples,
                                     warmup=n_warmup,
                                     m =m,g=g,
                                     algo=ecs_algo,
                                     subsample_method="perturb",
                                     proxy=proxy,
                                     map_method = map_method,
                                     num_epochs=epochs)
    est_posterior_NUTS = infer_hmcecs(rng_key, feats=feats, obs=obs, n_samples=n_samples,warmup=n_warmup,m =m,g=g,algo=algo)

    Plot(est_posterior_ECS,est_posterior_NUTS,ecs_algo,algo,proxy,m)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-num_samples', nargs='?', default=50, type=int)
    parser.add_argument('-num_warmup', nargs='?', default=5, type=int)
    parser.add_argument('-ecs_algo', nargs='?', default="NUTS", type=str)
    parser.add_argument('-ecs_proxy', nargs='?', default="svi", type=str)
    parser.add_argument('-algo', nargs='?', default="HMC", type=str)
    parser.add_argument('-map_init', nargs='?', default="NUTS", type=str)
    parser.add_argument("-epochs",default=2,type=int)
    args = parser.parse_args()


    rng_key = jax.random.PRNGKey(37)

    rng_key, feat_key, obs_key = jax.random.split(rng_key, 3)
    if args.ecs_proxy == "svi":
        args.map_init = "SVI"

    now = datetime.datetime.now()
    Folders("PLOTS_{}".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms")))
    file_hyperparams = open("PLOTS_{}/Hyperparameters_{}.txt".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms"),
                                                                     now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms")), "a")
    file_hyperparams.write('ECS algo : {} \n'.format(args.ecs_algo))
    file_hyperparams.write('algo : {} \n'.format(args.algo))
    file_hyperparams.write('MAP init : {} \n'.format(args.map_init))

    higgs = False
    if higgs:
        feats,obs = higgs_data()
        file_hyperparams.write('Dataset : HIGGS \n')

    else:
        feats, obs = breast_cancer_data()
        file_hyperparams.write('Dataset : BREAST CANCER DATA \n')

    file_hyperparams.close()
    config.update('jax_disable_jit', True)

    #Determine_best_sample_size(rng_key,feats[:100],obs[:100])
    Tests(args.map_init,args.ecs_algo,args.algo,args.num_samples,args.num_warmup,args.epochs,args.ecs_proxy)
    #Plot_KL(args.map_init,args.ecs_algo,args.algo,args.num_samples,args.num_warmup,args.epochs)


    exit()
    samples_ECS_3316 = load_obj("/home/lys/Dropbox/PhD/numpyro/examples/PLOTS_2020_10_09_11h41min24s333577ms_DONOTREMOVE/MCMC_Dict_Samples_perturb_m_3316.pkl")
    samples_ECS_6632 = load_obj("/home/lys/Dropbox/PhD/numpyro/examples/PLOTS_2020_10_09_11h41min24s333577ms_DONOTREMOVE/MCMC_Dict_Samples_perturb_m_6632.pkl")
    samples_ECS_132264 = load_obj("/home/lys/Dropbox/PhD/numpyro/examples/PLOTS_2020_10_09_11h41min24s333577ms_DONOTREMOVE/MCMC_Dict_Samples_perturb_m_13264.pkl")

    samples_HMC = load_obj("/home/lys/Dropbox/PhD/numpyro/examples/PLOTS_2020_10_09_11h41min24s333577ms_DONOTREMOVE/MCMC_Dict_Samples_None_m_all.pkl")

    p = dist.Normal(samples_ECS_3316["theta"])
    q = dist.Normal(samples_HMC["theta"])
    kl = kl_divergence(p, q)
    print(np_jax.average(kl))
    Plot(samples_ECS_3316, samples_HMC, args.ecs_algo, args.algo, 3316,args.ecs_proxy ,kl=kl)

    # samples = pd.DataFrame.from_records(samples,index="theta")
    # sns.kdeplot(data=kl, color=color, label="m : ".format(m_val))
    # plt.figure(m_val)
    # plt.xlabel(r"$\theta")
    # plt.ylabel("Density")
    # plt.legend()
    # plt.title(r"$\theta$ KL-divergence")
    # plt.savefig("{}/KL_divergence_m_{}.png".format("PLOTS_{}".format(now.strftime("%Y_%m_%d_%Hh%Mmin%Ss%fms")),
    #                                            str(m_val)))


