import os
import pickle
import dm_env
import numpy as np
from dmlab2d.ui_renderer import pygame
from ray.tune.registry import register_env
from ray.rllib.agents.registry import get_trainer_class
from adapters.env_creator import EnvCreator
from adapters.ray_model_policy import RayModelPolicy

experiment_name = "map_parolat_prob_meltingpot"
experiment_id = "A3C_meltingpot_d30c9_00000_0_2022-07-09_18-28-23"
checkpoint_id = 21

agent_algorithm = experiment_id.split("_")[0]
checkpoint_path = os.path.join(experiment_name, experiment_id,
                               f"checkpoint_{str(checkpoint_id).zfill(6)}",
                               f"checkpoint-{checkpoint_id}")
env_creator = EnvCreator()
register_env("meltingpot", env_creator.create_env)

stored_config = pickle.load(open(os.path.join("logs", experiment_name, experiment_id, "params.pkl"), 'rb'))
stored_config["evaluation_interval"] = None

trainer = get_trainer_class(agent_algorithm)(env="meltingpot", config=stored_config)
trainer.restore(os.path.join("logs", checkpoint_path))

env = env_creator.create_env(stored_config["env_config"]).get_dmlab2d_env()
bots = [RayModelPolicy(trainer, "av")] * stored_config["env_config"]["num_players"]

timestep = env.reset()
states = [bot.initial_state() for bot in bots]
actions = [0] * len(bots)

scale = 2
pygame.init()
pygame.display.set_caption('DM Lab2d')
shape = env.observation_spec()[0]['WORLD.RGB'].shape
game_display = pygame.display.set_mode((int(shape[1] * scale), int(shape[0] * scale)))

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
    if sim_step%100 == 0:
        print(f"step ", sim_step)
