import json
import pandas as pd
import tensorflow as tf


def load_config():
    with open("config/config.json") as json_file:
        config_file = json.load(json_file)
    return config_file


def parse_metrics(progress_file_path):
    progress = pd.read_csv(progress_file_path)
    metrics = {"elapsed_time": progress.time_total_s.iloc[-1],
               "episodes": (progress.timesteps_total.to_numpy()/1000).astype(int),
               "peacefulness": progress["custom_metrics/peacefulness_mean"].to_numpy(),
               "efficiency": progress["custom_metrics/Efficiency_mean"].to_numpy(),
               "equality": progress["custom_metrics/Equality_mean"].to_numpy(),
               "reward": progress.episode_reward_mean.to_numpy()}
    return metrics


def parse_dreamer_logs(path, metric_name):
    counter = 0
    hist = list()
    episodes = list()
    for e in tf.compat.v1.train.summary_iterator(path):
        for v in e.summary.value:
            if v.tag == metric_name:
                hist.append(float(tf.make_ndarray(v.tensor)))
                episodes.append(e.step/1000)
                counter += 1
    return {"episodes": episodes, "metric": hist}


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed