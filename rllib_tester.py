import argparse
import os
import random

import ray
from ray import tune
from ray.rllib.examples.env.multi_agent import MultiAgentCartPole
from ray.rllib.examples.models.shared_weights_model import (SharedWeightsModel1, SharedWeightsModel2)
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.test_utils import check_learning_achieved


def gen_policy(i):
    config = {
        "model": {
            "custom_model": ["model1", "model2"][i % 2],
        },
        "gamma": random.choice([0.95, 0.99]),
    }
    return PolicySpec(config=config)


tf1, tf, tfv = try_import_tf()


def experiment(exp_config):
    algo = exp_config.get("algo")
    timesteps_total = exp_config.get("timesteps_total")
    n_workers = exp_config.get("n_workers")
    num_agents = exp_config.get("num_agents")
    n_cpus_per_worker = exp_config.get("n_cpus_per_worker")
    ray.init()
    # Register the models to use.

    mod1 = SharedWeightsModel1
    mod2 = SharedWeightsModel2
    ModelCatalog.register_custom_model("model1", mod1)
    ModelCatalog.register_custom_model("model2", mod2)


    policies = {"policy_{}".format(i): gen_policy(i) for i in range(num_agents)}
    policy_ids = list(policies.keys())

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        pol_id = random.choice(policy_ids)
        return pol_id

    config = {
        "env": MultiAgentCartPole,
        "env_config": {
            "num_agents": num_agents,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 0,
        "num_workers": n_workers,
        "num_cpus_per_worker": 0.5,
        #"num_sgd_iter": 10,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "framework": "tf",
    }
    stop = {
        "timesteps_total": timesteps_total,
    }

    results = tune.run(algo, name=f"test_{algo}_{n_workers}_workers_{n_cpus_per_worker}_cpus", stop=stop, config=config,
                       local_dir="logs", verbose=3)
    ray.shutdown()


timesteps_total = 100000
num_agents = 10

experiment_configs = [
    {"algo": "PPO",
     "timesteps_total": timesteps_total,
     "n_workers": 1,
     "num_agents": num_agents,
     "n_cpus_per_worker": 1},
    {"algo": "A3C",
     "timesteps_total": timesteps_total,
     "n_workers": 1,
     "num_agents": num_agents,
     "n_cpus_per_worker": 1},
    {"algo": "PPO",
     "timesteps_total": timesteps_total,
     "n_workers": 1,
     "num_agents": num_agents,
     "n_cpus_per_worker": 5},
    {"algo": "A3C",
     "timesteps_total": timesteps_total,
     "n_workers": 1,
     "num_agents": num_agents,
     "n_cpus_per_worker": 5},
    {"algo": "PPO",
     "timesteps_total": timesteps_total,
     "n_workers": 7,
     "num_agents": num_agents,
     "n_cpus_per_worker": 1},
    {"algo": "A3C",
     "timesteps_total": timesteps_total,
     "n_workers": 7,
     "num_agents": num_agents,
     "n_cpus_per_worker": 1},
    {"algo": "PPO",
     "timesteps_total": timesteps_total,
     "n_workers": 6,
     "num_agents": num_agents,
     "n_cpus_per_worker": 1},
    {"algo": "A3C",
     "timesteps_total": timesteps_total,
     "n_workers": 6,
     "num_agents": num_agents,
     "n_cpus_per_worker": 1},
    {"algo": "PPO",
     "timesteps_total": timesteps_total,
     "n_workers": 14,
     "num_agents": num_agents,
     "n_cpus_per_worker": 0.5},
    {"algo": "A3C",
     "timesteps_total": timesteps_total,
     "n_workers": 14,
     "num_agents": num_agents,
     "n_cpus_per_worker": 0.5},
]

for experiment_config in experiment_configs:
    experiment(experiment_config)
