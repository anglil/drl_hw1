from drl_hw1.utils.gym_env import GymEnv
from drl_hw1.policies.gaussian_mlp import MLP
from drl_hw1.baselines.linear_baseline import LinearBaseline
from drl_hw1.baselines.MLP_baseline import MLPBaseline
from drl_hw1.algos.batch_reinforce import BatchREINFORCE
from drl_hw1.utils.train_agent import train_agent
import drl_hw1.envs

import matplotlib.pyplot as plt
import time as timer
import pickle
import sys
import os
import csv
import numpy as np

# ---- utility ----
def read_eval_score(f_name0):
  stoc_pol_mean = []
  eval_score = []
  if f_name0.endswith("csv"):
    with open(f_name0) as f:
      reader = csv.DictReader(f)
      for row in reader:
        stoc_pol_mean.append(row['stoc_pol_mean'])
        eval_score.append(row['eval_score'])
  elif f_name0.endswith("pickle"):
    #col = pickle.load(open(f_name0, 'rb')).keys()
    stoc_pol_mean = pickle.load(open(f_name0, 'rb'))['stoc_pol_mean']
    eval_score = pickle.load(open(f_name0, 'rb'))['eval_score']
  else:
    raise Exception("file format not supported.")
  return np.array(stoc_pol_mean), np.array(eval_score)


# ---- parameters ----
envs0 = ["drl_hw1_point_mass-v0","drl_hw1_swimmer-v0","drl_hw1_half_cheetah-v0","drl_hw1_ant-v0"]
[print(str(i)+": "+envs0[i]) for i in range(len(envs0))]
env_name = envs0[int(input("choose an environemnt: "))]

budget = 500
layer1_size=32
layer2_size=32
gamma=0.995
gae_lambda=0.97
train_niter=50
train_save_freq=1
eval_rollout=1

to_visualize=False
vis_nrollout=2

be_adaptive = False
use_natural_grad = True
MLP_baseline = True

fig_stoc_pol_mean = plt.figure(1)
fig_eval_score = plt.figure(2)

lrs = [0.5]#[0.1,0.5]
eval_nrollouts = [10]#[10,20]
kl_thresholds = [0.5]#[0.5,1]

SEEDs = [100,500,1000]

for kl_threshold in kl_thresholds:
  for lr in lrs:
    for eval_nrollout in eval_nrollouts:
      eval_scores = []
      stoc_pol_means = []
      for SEED in SEEDs:
        
        #train_niter=int(budget/eval_nrollout)
        #train_save_freq=max(int(train_niter/10),1)
        
        exp_name = "_".join([
          env_name,
          str(lr),
          str(eval_nrollout),
          str(be_adaptive),
          str(kl_threshold),
          str(use_natural_grad),
          str(MLP_baseline),
          str(SEED),
          str(layer1_size),
          str(layer2_size),
          str(train_niter),
          str(train_save_freq),
          str(gamma),
          str(gae_lambda)
          ])
        # --------
  
        f_name = os.path.join(exp_name, 'logs', 'log.pickle')
        if not os.path.exists(f_name):
          # load environment
          e = GymEnv(env_name)
          # create policy
          policy = MLP(e.spec, hidden_sizes=(layer1_size,layer2_size), seed=SEED)
          # create baseline policy
          if not MLP_baseline:
            baseline = LinearBaseline(e.spec)
          else:
            baseline = MLPBaseline(e.spec)
          # create agent, where policy is a member of a BatchReINFORCE instance
          agent = BatchREINFORCE(
              e, 
              policy, 
              baseline, 
              learn_rate=lr, 
              seed=SEED, 
              be_adaptive=False,
              kl_threshold=kl_threshold,
              use_natural_grad=True,
              save_logs=True)
  
          ts = timer.time()
          # train/optimize parameters, where "gae" means "generalized advantage estimation"
          train_agent(
              job_name=exp_name,
              agent=agent,
              seed=SEED,
              niter=train_niter,
              gamma=gamma,
              gae_lambda=gae_lambda,
              num_cpu='max',
              sample_mode='trajectories',
              num_traj=eval_nrollout,
              save_freq=train_save_freq,
              evaluation_rollouts=eval_rollout)
          print("time taken = %f" % (timer.time()-ts))
        else:
          print("log exists for lr={},rollout={},seed={}".format(lr,eval_nrollout,SEED)+" at "+f_name)
          print("================")
        
        if to_visualize:
          # mode.evaluation: play the mean of the stochastic policy
          # mode.exploration: play the stochastic policy
          e.visualize_policy(
              policy, 
              num_episodes=vis_nrollout, 
              horizon=e.horizon, 
              mode="evaluation")
  
        os.chdir("/home/anglil/Desktop/drl_hw1/examples") # train_agent changes directory...
        assert(os.path.exists(f_name))
        stoc_pol_mean, eval_score = read_eval_score(f_name)
        assert(len(stoc_pol_mean) == train_niter)
        stoc_pol_means.append(stoc_pol_mean)
        eval_scores.append(eval_score)
  
      plt.figure(1)
      plt.plot(np.mean(stoc_pol_means, axis=0), label="lr={},rollout={},kl={}".format(lr,eval_nrollout,kl_threshold))
      
      plt.figure(2)
      plt.plot(np.mean(eval_scores, axis=0), label="lr={},rollout={},kl={}".format(lr,eval_nrollout,kl_threshold))

plt.figure(1)
plt.title("stoc_pol_mean")
plt.xlabel("iterations")
plt.ylabel("reward")
plt.legend()

plt.figure(2)
plt.title("eval_score")
plt.xlabel("iterations")
plt.ylabel("reward")
plt.legend()

fig_stoc_pol_mean.savefig(env_name+"-stoc_pol_mean4.png")
fig_eval_score.savefig(env_name+"-eval_score4.png")




