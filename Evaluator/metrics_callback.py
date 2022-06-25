from typing import Dict
import numpy as np
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks


class MetricsCallback(DefaultCallbacks):
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        pass
        #print(base_env._nicole)
        # print("episode {} started".format(episode.episode_id))
        # episode.user_data["pole_angles"] = []
        # episode.hist_data["pole_angles"] = []

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        #print("[MetricsCallback]", episode.last_info_for("player_0"))
        pass
        # print("Aqui toy", episode.prev_reward_for(1))
        # pole_angle = abs(episode.last_observation_for()[2])
        # raw_angle = abs(episode.last_raw_obs_for()[2])
        # assert pole_angle == raw_angle
        # episode.user_data["pole_angles"].append(pole_angle)

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        pass
        # pole_angle = np.mean(episode.user_data["pole_angles"])
        # print("episode {} ended with length {} and pole angles {}".format(
        #     episode.episode_id, episode.length, pole_angle))
        # episode.custom_metrics["pole_angle"] = pole_angle
        # episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]

    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        # print("returned sample batch of size {}".format(samples.count))
        pass

    def on_train_result(self, trainer, result: dict, **kwargs):
        pass
        # print("trainer.train() result: {} -> {} episodes".format(
        #     trainer, result["episodes_this_iter"]))
        # # you can mutate the result dict to add new fields to return
        # result["callback_ok"] = True

    def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass
        # print("postprocessed {} steps".format(postprocessed_batch.count))
        # if "num_batches" not in episode.custom_metrics:
        #     episode.custom_metrics["num_batches"] = 0
        # episode.custom_metrics["num_batches"] += 1
        pass
