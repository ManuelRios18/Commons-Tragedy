import copy
import json
import time
import dm_env
import numpy as np
from dmlab2d.ui_renderer import pygame
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env
from examples.tutorial.harvest.configs.environment import harvest_uniandes as game
from adapters.env_creator import EnvCreator
from adapters.ray_model_policy import RayModelPolicy


agent_algorithm = "A3C"
substrate_name = "commons_harvest_open"
load_dir = "logs/"
checkpoint_path = "/A3C/A3C_meltingpot_87b20_00000_0_2022-06-26_19-17-29/checkpoint_000015/checkpoint-15"

env_creator = EnvCreator()

trainer_config = copy.deepcopy(get_trainer_class(agent_algorithm).get_default_config())
trainer_config["env_config"] = game.get_config()
register_env("meltingpot", env_creator.create_env)

test_env = env_creator.create_env(trainer_config["env_config"])
obs_space = test_env.single_player_observation_space()
act_space = test_env.single_player_action_space()

trainer_config["multiagent"] = {
  "policies": {
      "av": PolicySpec(
              policy_class=None,  # use default policy
              observation_space=obs_space,
              action_space=act_space,
              config={}),
  },
  "policy_mapping_fn": lambda agent_id, **kwargs: "av"
}

with open(load_dir + '/config.json', 'r') as f:
    trainer_config.update(json.load(f))

trainer = get_trainer_class(agent_algorithm)(
  env="meltingpot", config=trainer_config)

trainer.restore(load_dir + checkpoint_path)
env = test_env._env
bots = [RayModelPolicy(trainer, "av")] * trainer_config["env_config"]["num_players"]

timestep = env.reset()
states = [bot.initial_state() for bot in bots]
actions = [0] * len(bots)

scale = 2
pygame.init()
pygame.display.set_caption('DM Lab2d')
shape = env.observation_spec()[0]['WORLD.RGB'].shape
game_display = pygame.display.set_mode((int(shape[1] * scale), int(shape[0] * scale)))
reward_hist = {bot_num:0 for bot_num in range(len(bots))}
reward_hist["total"] = 0
zap_hist = list()
zap_mem = 0
red_hist = list()
blue_hist = list()
green_hist = list()
total_reward_hist = list()
for sim_step in range(1000):
    obs = timestep.observation[0]['WORLD.RGB']
    obs = np.transpose(obs, (1, 0, 2))
    surface = pygame.surfarray.make_surface(obs)
    rect = surface.get_rect()
    surf = pygame.transform.scale(surface, (int(rect[2] * scale), int(rect[3] * scale)))

    game_display.blit(surf, dest=(0, 0))
    pygame.display.update()

    for i, bot in enumerate(bots):
      timestep_bot = dm_env.TimeStep(
          step_type=timestep.step_type,
          reward=timestep.reward[i],
          discount=timestep.discount,
          observation=timestep.observation[i])

      actions[i], states[i] = bot.step(timestep_bot, states[i])

    timestep = env.step(actions)
    acum = 0
    for i, r in enumerate(timestep.reward):
        reward_hist[i] += float(r)
        acum += reward_hist[i]
    reward_hist["total"] = acum
    time.sleep(0.1)

