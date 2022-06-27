"""This class was took from DeepMind's meltingpot original"""

import dm_env
from typing import Tuple
from ray.rllib.agents import trainer
from meltingpot.python.utils.bots import policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID


class RayModelPolicy(policy.Policy):
  """Policy wrapping an rllib model for inference.

  Note: Currently only supports a single input, batching is not enabled
  """

  def __init__(self,
               model: trainer.Trainer,
               policy_id: str = DEFAULT_POLICY_ID) -> None:
    """Initialize a policy instance.

    Args:
      model: An rllib.trainer.Trainer checkpoint.
      policy_id: Which policy to use (if trained in multi_agent mode)
    """
    self._model = model
    self._prev_action = 0
    self._policy_id = policy_id

  def step(self, timestep: dm_env.TimeStep,
           prev_state: policy.State) -> Tuple[int, policy.State]:
    """See base class."""
    observations = {
        key: value
        for key, value in timestep.observation.items()
        if 'WORLD' not in key
    }

    action, state, _ = self._model.compute_single_action(
        observations,
        prev_state,
        policy_id=self._policy_id,
        prev_action=self._prev_action,
        prev_reward=timestep.reward)

    self._prev_action = action
    return action, state

  def initial_state(self) -> policy.State:
    """See base class."""
    self._prev_action = 0
    return self._model.get_policy(self._policy_id).get_initial_state()

  def close(self) -> None:
    """See base class."""