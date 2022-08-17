"""This class was took from DeepMind's meltingpot original"""

import dmlab2d
from gym import spaces
from adapters import adapter_utils
from ray.rllib.env import multi_agent_env


PLAYER_STR_FORMAT = 'player_{index}'


class MeltingPotEnv(multi_agent_env.MultiAgentEnv):
  """An adapter between the Melting Pot substrates and RLLib MultiAgentEnv."""

  def __init__(self, env: dmlab2d.Environment):
    self.is_meltingpot_env = True
    self._env = env
    self._num_players = len(self._env.observation_spec())
    self._ordered_agent_ids = [
        PLAYER_STR_FORMAT.format(index=index)
        for index in range(self._num_players)
    ]
    self._agent_ids = set(self._ordered_agent_ids)
    self.observation_space = spaces.Dict({key: value for key, value in self.single_player_observation_space().items() if key == "RGB"})

    self.action_space = self.single_player_action_space()
    self.world_view = None
    super().__init__()

  def reset(self):
    """See base class."""
    timestep = self._env.reset()
    self.world_view = timestep.observation[0]["WORLD.RGB"]
    return adapter_utils.timestep_to_observations(timestep)

  def step(self, action):
    """See base class."""
    actions = [action[agent_id] for agent_id in self._ordered_agent_ids]
    timestep = self._env.step(actions)
    self.world_view = timestep.observation[0]["WORLD.RGB"]
    rewards = {
        agent_id: timestep.reward[index]
        for index, agent_id in enumerate(self._ordered_agent_ids)
    }
    done = {'__all__': True if timestep.last() else False}
    info = adapter_utils._timestep_to_world_observations(timestep)

    observations = adapter_utils.timestep_to_observations(timestep)
    return observations, rewards, done, info

  def close(self):
    """See base class."""
    self._env.close()

  def get_dmlab2d_env(self):
    """Returns the underlying DM Lab2D environment."""
    return self._env

  def single_player_observation_space(self) -> spaces.Space:
    """The observation space for a single player in this environment."""
    return adapter_utils.remove_world_observations_from_space(
        adapter_utils.spec_to_space(self._env.observation_spec()[0]))

  def single_player_action_space(self):
    """The action space for a single player in this environment."""
    return adapter_utils.spec_to_space(self._env.action_spec()[0])