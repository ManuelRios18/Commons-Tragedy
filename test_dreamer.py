import numpy as np
import time
import dreamerv2.api as dv2
import pathlib
import ruamel.yaml as yaml
from dmlab2d.ui_renderer import pygame
import dreamerv2.agent as dreamer_agent
import substrates as substrates_handler
import dreamerv2.common as dreamer_common
from adapters.env_creator import EnvCreator


substrate_name = "commons_harvest_uniandes"
substrate_config = {"prob_type": "meltingpot",
                    "map_name": "single_agent_small"}
weights_path = "logs/dreamerv2/variables.pkl"

default_config = yaml.safe_load(pathlib.Path('config/dreamer_configs.yaml').read_text())['defaults']
atari_config = {
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


env = dreamer_common.GymWrapper(env)
env = dreamer_common.ResizeImage(env)
if hasattr(env.act_space['action'], 'n'):
    env = dreamer_common.OneHotAction(env)
else:
    env = dreamer_common.NormalizeAction(env)
env = dreamer_common.TimeLimit(env, config.time_limit)

observation = env.reset()
step = dreamer_common.Counter()
agnt = dreamer_agent.Agent(config, env.obs_space, env.act_space, step)
dataset = iter(replay.dataset(**config.dataset))
train_agent = dreamer_common.CarryOverState(agnt.train)
train_agent(next(dataset))
agnt.load(weights_path)

inner_env = env._env._env._env._env
agent_mode = "eval"
scale = 2
pygame.init()
pygame.display.set_caption('DM Lab2d')
shape = inner_env.world_view.shape
game_display = pygame.display.set_mode((int(shape[1] * scale), int(shape[0] * scale)))

state = None
score = 0
for sim_step in range(1000):
    image_to_render = inner_env.world_view
    image_to_render = np.transpose(image_to_render, (1, 0, 2))
    surface = pygame.surfarray.make_surface(image_to_render)
    rect = surface.get_rect()
    surf = pygame.transform.scale(surface, (int(rect[2] * scale), int(rect[3] * scale)))
    game_display.blit(surf, dest=(0, 0))
    pygame.display.update()
    score += observation["reward"]
    print("current score", score)
    observation["RGB"] = np.expand_dims(observation["RGB"], 0)
    observation["reward"] = np.array([observation["reward"]])
    observation["is_first"] = np.array([observation["is_first"]])
    observation["is_last"] = np.array([observation["is_last"]])
    observation["is_terminal"] = np.array([observation["is_terminal"]])

    actions, state = agnt.policy(observation, state, agent_mode)
    actions["action"] = np.array(actions["action"])[0]

    observation = env.step(actions)
    time.sleep(0.1)

print("agent score: ", score)