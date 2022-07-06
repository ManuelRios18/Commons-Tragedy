import ray
import copy
import platform
from ray import tune
import tensorflow as tf
import substrates as substrates_handler
from adapters.env_creator import EnvCreator
from ray.rllib.policy.policy import PolicySpec
from evaluator.metrics_callback import MetricsCallback
from ray.rllib.agents.registry import get_trainer_class


class Trainer:

    def __init__(self, model_name, substrate_name, agent_algorithm, n_steps, checkpoint_freq, keep_checkpoints_num,
                 num_workers, experiment_name=None, max_gpus=None):
        self.model_name = model_name
        self.substrate_name = substrate_name
        self.agent_algorithm = agent_algorithm
        self.n_steps = n_steps
        self.checkpoint_freq = checkpoint_freq
        self.keep_checkpoints_num = keep_checkpoints_num
        self.num_workers = num_workers
        self.experiment_name = experiment_name
        self.max_gpus = max_gpus

        self.n_gpus = self.get_n_gpus()
        self.trainer_config = copy.deepcopy(get_trainer_class(self.agent_algorithm).get_default_config())
        self.game = substrates_handler.get_game(substrate_name)
        self.trainer_config["env_config"] = self.game.get_config()

        self.env_creator = EnvCreator()
        self.test_env = self.env_creator.create_env(self.trainer_config["env_config"])
        self.obs_space = self.test_env.single_player_observation_space()
        self.act_space = self.test_env.single_player_action_space()
        self.model = self.get_model()
        self.config = self.get_config()

        tune.register_env("meltingpot", self.env_creator.create_env)

    def get_n_gpus(self):
        if platform.system() == "Darwin":
            n_gpus = 0
        else:
            n_detected_gpus = len(tf.config.list_physical_devices("GPU"))
            n_gpus = n_detected_gpus
            if self.max_gpus is not None:
                n_gpus = min(self.max_gpus, n_detected_gpus)
        return n_gpus

    def get_model(self):
        if self.model_name == "large-model":
            model = {"conv_filters": [[16, [8, 8], 8], [128, [11, 11], 1]],
                     "conv_activation": "relu",
                     "post_fcnet_hiddens": [256],
                     "post_fcnet_activation": "relu",
                     "use_lstm": True,
                     "lstm_use_prev_action": True,
                     "lstm_use_prev_reward": False,
                     "lstm_cell_size": 256
                     }
        else:
            raise NotImplementedError(f"The model {self.model_name} is not implemented !")

        return model

    def get_config(self):
        config = {
            "env": "meltingpot",
            "callbacks": MetricsCallback,
            "env_config": self.trainer_config["env_config"],
            "num_gpus": self.n_gpus,
            "num_workers": self.num_workers,
            "horizon": self.trainer_config["env_config"].lab2d_settings["maxEpisodeLengthFrames"],
            "batch_mode": "complete_episodes",
            "rollout_fragment_length": 1,
            "train_batch_size": self.trainer_config["env_config"].lab2d_settings["maxEpisodeLengthFrames"],
            "no_done_at_end": False,
            "model": self.model,
            "evaluation_interval": 50,
            "evaluation_duration": 1,
            "evaluation_duration_unit": "episodes",
            "multiagent":
                {
                    "policies": {
                        "av": PolicySpec(
                            policy_class=None,
                            observation_space=self.obs_space,
                            action_space=self.act_space,
                            config={}),
                    },
                    "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: "av",
                }
        }

        return config

    def start_training(self):
        ray.init()
        results = tune.run(self.agent_algorithm,
                           stop={"timesteps_total": self.n_steps},
                           config=self.config,
                           local_dir="logs",
                           checkpoint_freq=self.checkpoint_freq,
                           keep_checkpoints_num=self.keep_checkpoints_num,
                           checkpoint_score_attr='training_iteration',
                           metric="episode_reward_mean",
                           mode="max",
                           checkpoint_at_end=True)
        ray.shutdown()
        return results
