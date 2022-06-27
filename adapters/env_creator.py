from ml_collections import config_dict
from meltingpot.python import substrate
from adapters.meltingpot_env import MeltingPotEnv


class EnvCreator:

    def __init__(self):
        pass

    def create_env(self, env_config):
        """Outputs an environment for registering."""
        env = substrate.build(config_dict.ConfigDict(env_config))
        env = MeltingPotEnv(env)
        return env