import json
import numpy as np
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


def parse_recovered_image(image, image_source):
    image = image.squeeze()
    if image_source == "dream":
        image = (image - image.min()) / (image.max() - image.min()+0.001)
        image *= 255
    return image


def parse_ma_dreams(agents_dreams):
    agents_result = {}
    for key, agent_data in agents_dreams.items():
        player_id = key.replace("_openl_RGB", "")
        #agent_data = agent_data[:, :, :, ::-1].numpy()
        agent_data = agent_data.numpy()
        t, h, w, c = agent_data.shape
        n_memories = int(w / 64)
        # extract ground truth and dreams
        gt = agent_data[:, :64, :, :] * 255
        dream = agent_data[:, 64:128, :, :]

        dream = (dream - dream.min()) / (dream.max() - dream.min())
        dream *= 255

        # change data format
        gt = gt.astype(np.uint8)
        dream = dream.astype(np.uint8)
        agent_result = {}
        for data_type, recovered in {"gt": gt, "dream": dream}.items():
            memories = np.split(recovered, n_memories, axis=2)
            agent_result[data_type] = {}
            for mem_id, memory in enumerate(memories):
                agent_result[data_type][mem_id] = {i: d.squeeze() for i, d in enumerate(np.split(memory, t, axis=0))}
        agents_result[player_id] = agent_result
    return agents_result

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