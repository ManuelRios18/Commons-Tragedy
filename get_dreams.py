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


substrate_name = "commons_harvest_uniandes"
substrate_config = {"prob_type": "meltingpot",
                    "map_name": "two_agents_small"}

default_config = yaml.safe_load(pathlib.Path('config/dreamer_configs.yaml').read_text())['defaults']
atari_config = {
    "logdir": "logs/dreamerv2_ma_2p_test2",
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
agents_mob.create_agents(env.obs_space, env.act_space, step)
eval_datasets = agents_mob.get_datasets(mode="eval")
train_agents = dreamer_common.CarryOverStateMultiAgent(agents_mob.train_mob, n_agents, agents_prefix)
train_agents(eval_datasets)

agents_mob.load_agents()
agents_dreams = agents_mob.report(eval_datasets)
parsed_dreams = utils.parse_ma_dreams(agents_dreams)

# parsed_dreams["player_0"]["dream"][0][2]
imageio.mimsave('dream_0.mp4',  list(parsed_dreams["player_0"]["dream"][0].values()))
imageio.mimsave('dream_1.mp4',  list(parsed_dreams["player_1"]["dream"][0].values()))
