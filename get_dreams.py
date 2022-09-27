import os
import pickle
import numpy as np
import time
import dreamerv2.api as dv2
import imageio
import utils
import pathlib
import ruamel.yaml as yaml
import dreamerv2.mob as mob
from dmlab2d.ui_renderer import pygame
import dreamerv2.agent as dreamer_agent
import substrates as substrates_handler
import dreamerv2.common as dreamer_common
from adapters.env_creator import EnvCreator
import cv2

total_dreams = 1000
substrate_name = "commons_harvest_uniandes"
substrate_config = {"prob_type": "meltingpot",
                    "map_name": "two_agents_small"}

default_config = yaml.safe_load(pathlib.Path('config/dreamer_configs.yaml').read_text())['defaults']
atari_config = {
    "logdir": "logs/dreamerv2_ma_2p",
    "task": "atari_pong",
    "encoder": {"mlp_keys": "$^", "cnn_keys": "RGB", "act": "elu", "norm": "none", "cnn_depth": 48, "cnn_kernels": [4, 4, 4, 4], "mlp_layers": [400, 400, 400, 400]},
    "decoder":  {"mlp_keys": '$^', "cnn_keys": 'RGB', "act": "elu", "norm": "none", "cnn_depth": 48, "cnn_kernels": [5, 5, 6, 6], "mlp_layers": [400, 400, 400, 400]},
    "time_limit": 27000,
    "action_repeat": 1,
    "steps": 5e7,
    "eval_every": 2.5e5,
    "log_every": 1e4,
    "prefill": 50000,
    "train_every": 16,
    "clip_rewards": "tanh",
    "rssm": {"ensemble": 1, "hidden": 600, "deter": 600, "stoch": 32, "discrete": 32, "act": "elu", "norm": 'none', "std_act": "sigmoid2", "min_std": 0.1},
    "model_opt": {"opt": "adam", "lr": 2e-4, "eps": 1e-5, "clip": 100, "wd": 1e-6},
    "actor_opt": {"opt": "adam", "lr": 4e-5, "eps": 1e-5, "clip": 100, "wd": 1e-6},
    "critic_opt": {"opt": "adam", "lr": 1e-4, "eps": 1e-5, "clip": 100, "wd": 1e-6},
    "actor_ent": 1e-3,
    "discount": 0.999,
    "loss_scales": {"kl": 0.1, "reward": 1.0, "discount": 0.5, "proprio": 1.0},
    "log_keys_video": ["RGB"]
    }
default_config.update(atari_config)
config = dreamer_common.Config(default_config)
logdir = pathlib.Path(config.logdir).expanduser()
replay = dreamer_common.Replay(logdir / 'train_episodes', **config.replay)

env_creator = EnvCreator()
game = substrates_handler.get_game(substrate_name)
env = env_creator.create_env(game.get_config(substrate_config))


env = dreamer_common.GymWrapperMultiAgent(env)
env = dreamer_common.ResizeImageMultiAgent(env)
if hasattr(env.act_space['action'], 'n'):
    env = dreamer_common.OneHotActionMultiAgent(env)
else:
    env = dreamer_common.NormalizeAction(env)
env = dreamer_common.TimeLimit(env, config.time_limit)

observation = env.reset()
step = dreamer_common.Counter()

n_agents = env._num_players
agents_prefix = "player_"
agents_mob = mob.Mob(config, logdir, n_agents, agents_prefix, load_train_ds=False)
agents_mob.initialize_recording()
agents_mob.create_agents(env.obs_space, env.act_space, step)

replays = agents_mob.record_replays
record_datasets = {player_id: iter(replays[player_id].dataset(**config.dataset))for player_id in replays.keys()}
train_agents = dreamer_common.CarryOverStateMultiAgent(agents_mob.train_mob, n_agents, agents_prefix)
train_agents(record_datasets)

dreams_root = os.path.join(str(logdir), "dreams")
if not os.path.isdir(dreams_root):
    os.mkdir(dreams_root)
    for agent_id in range(n_agents):
        dir_name = os.path.join(dreams_root, f"{agents_prefix}{agent_id}")
        os.mkdir(dir_name)
        os.mkdir(os.path.join(dir_name, "data"))
        os.mkdir(os.path.join(dir_name, "videos"))

agents_mob.load_agents()


dreams_counter = {f"{agents_prefix}{agent_id}": 0 for agent_id in range(n_agents)}

while dreams_counter["player_0"] < total_dreams:
    agents_dreams = agents_mob.report(record_datasets)
    parsed_dreams = utils.parse_ma_dreams(agents_dreams)

    for agent_id in range(n_agents):
        agent_name = f"{agents_prefix}{agent_id}"
        n_dreams = len(parsed_dreams[agent_name]["dream"])
        for dream_id in range(n_dreams):
            dream_number = dreams_counter[agent_name]
            dream = parsed_dreams[agent_name]["dream"][dream_id]
            gt = parsed_dreams[agent_name]["gt"][dream_id]
            dream_video_path = os.path.join(dreams_root, agent_name, "videos", f"dream_{dream_number}.mp4")
            imageio.mimsave(dream_video_path, list(dream.values()))
            gt_video_path = os.path.join(dreams_root, agent_name, "videos", f"gt{dream_number}.mp4")
            imageio.mimsave(gt_video_path, list(gt.values()))
            utils.save_pickle(os.path.join(dreams_root, agent_name, "data", f"dream_{dream_number}.pickle"), dream)
            utils.save_pickle(os.path.join(dreams_root, agent_name, "data", f"gt_{dream_number}.pickle"), gt)
            dreams_counter[agent_name] += 1

    print(f"Total stored dreams: {dreams_counter['player_0']}")
