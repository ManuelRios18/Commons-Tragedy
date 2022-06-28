import ray
import copy
import json
import platform
from ray import tune
from adapters.env_creator import EnvCreator
from ray.rllib.policy.policy import PolicySpec
from evaluator.metrics_callback import MetricsCallback
from ray.rllib.agents.registry import get_trainer_class
from examples.tutorial.harvest.configs.environment import harvest_uniandes as game


agent_algorithm = "A3C"
num_cpus = 1
save_dir = "logs"

n_gpus = 0 if platform.system() == "Darwin" else 1

trainer_config = copy.deepcopy(get_trainer_class(agent_algorithm).get_default_config())
trainer_config["env_config"] = game.get_config()

env_creator = EnvCreator()
test_env = env_creator.create_env(trainer_config["env_config"])
obs_space = test_env.single_player_observation_space()
act_space = test_env.single_player_action_space()


model = {"conv_filters": [[16, [8, 8], 8], [128, [11, 11], 1]],
         "conv_activation": "relu",
         "post_fcnet_hiddens": [256],
         "post_fcnet_activation": "relu",
         "use_lstm": True,
         "lstm_use_prev_action": True,
         "lstm_use_prev_reward": False,
         "lstm_cell_size": 256
         }

config = {"env": "meltingpot",
          "callbacks": MetricsCallback,
          "env_config": trainer_config["env_config"],
          "num_gpus": n_gpus,
          "num_workers": 1,
          "horizon": trainer_config["env_config"].lab2d_settings["maxEpisodeLengthFrames"],
          "batch_mode": "complete_episodes",
          "rollout_fragment_length": 1,
          "train_batch_size": trainer_config["env_config"].lab2d_settings["maxEpisodeLengthFrames"],
          #"sgd_minibatch_size": 128,
          "no_done_at_end": False,
          "model": model,
          "evaluation_interval": 50,
          "evaluation_duration": 1,
          "evaluation_duration_unit": "episodes",
          #"evaluation_num_workers": 1,
          "multiagent":
              {
                "policies": {
                "av": PolicySpec(
                policy_class=None,
                observation_space=obs_space,
                action_space=act_space,
                config={}),
                },
                "policy_mapping_fn": lambda agent_id, **kwargs: "av"
                }
              }

stop = {"timesteps_total": 16000000}


tune.register_env("meltingpot", env_creator.create_env)
ray.init(num_cpus=num_cpus + 1)

with open(save_dir + "/config.json", "w") as f:
    target_keys = ["num_gpus", "num_workers", "rollout_fragment_length", "train_batch_size", "horizon",
                   "no_done_at_end", "model"]
    config_to_store = dict()
    for tk in target_keys:
        config_to_store[tk] = config[tk]
    json.dump(config_to_store, f)

results = tune.run(agent_algorithm, config=config, stop=stop, local_dir=save_dir,
         checkpoint_freq=200,
         keep_checkpoints_num=None, # 1 to debug
         checkpoint_score_attr='training_iteration')
