import json
import pandas as pd


def load_config():
    with open("config/config.json") as json_file:
        config_file = json.load(json_file)
    return config_file


def parse_metrics(progress_file_path):
    data = pd.read_csv(progress_file_path)
    metrics = {"elapsed_time": data.time_total_s.iloc[-1],
               "episodes": (data.timesteps_total.to_numpy()/1000).astype(int),
               "peacefulness": data["custom_metrics/peacefulness_mean"].to_numpy(),
               "efficiency": data["custom_metrics/Efficiency_mean"].to_numpy(),
               "equality": data["custom_metrics/Equality_mean"].to_numpy(),
               "reward": data.episode_reward_mean.to_numpy()}
    return metrics

