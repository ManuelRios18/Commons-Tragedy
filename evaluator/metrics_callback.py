import numpy as np
from typing import Dict
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
import evaluator.metrics_utils as metrics_utils
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker


class MetricsCallback(DefaultCallbacks):
    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        episode.hist_data["WHO_ZAPPED_WHO"] = list()

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        world_data = episode.last_info_for("player_0")
        episode.hist_data["WHO_ZAPPED_WHO"].append(world_data["WORLD.WHO_ZAPPED_WHO"])
        pass

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):

        # Get Zap data
        T = len(episode.hist_data['WHO_ZAPPED_WHO'])
        N = episode.hist_data["WHO_ZAPPED_WHO"][-1].shape[0]
        n_zaps = np.sum(np.sum(np.vstack(episode.hist_data["WHO_ZAPPED_WHO"])))
        peacefulness = (N*T - n_zaps)/T
        episode.custom_metrics["peacefulness"] = peacefulness
        episode.custom_metrics["n_zaps"] = n_zaps

        # Get Apple Data
        world_data = episode.last_info_for("player_0")
        total_consumption = np.sum(world_data["WORLD.CONSUMPTION_BY_PLAYER"])
        gini_score = metrics_utils.gini(world_data["WORLD.CONSUMPTION_BY_PLAYER"])
        episode.custom_metrics["Efficiency"] = total_consumption/N
        episode.custom_metrics["Equality"] = gini_score

    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        pass

    def on_train_result(self, trainer, result: dict, **kwargs):
        pass

    def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass
