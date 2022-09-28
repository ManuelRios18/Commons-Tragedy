import pathlib
import ruamel.yaml as yaml
import dreamerv2.api_ma as dv2
import substrates as substrates_handler
import dreamerv2.common as dreamer_common
from adapters.env_creator import EnvCreator


substrate_name = "commons_harvest_uniandes"
substrate_config = {"prob_type": "meltingpot",
                    "map_name": "four_agents_small"}

env_creator = EnvCreator()
game = substrates_handler.get_game(substrate_name)
env = env_creator.create_env(game.get_config(substrate_config))
testing_env = env_creator.create_env(game.get_config(substrate_config))
default_config = yaml.safe_load(pathlib.Path('config/dreamer_configs.yaml').read_text())['defaults']
atari_config = {
    "logdir": "logs/dreamerv2_ma_4p",
    "task": "atari_pong",
    "encoder": {"mlp_keys": "$^", "cnn_keys": "RGB", "act": "elu", "norm": "none", "cnn_depth": 48, "cnn_kernels": [4, 4, 4, 4], "mlp_layers": [400, 400, 400, 400]},
    "decoder":  {"mlp_keys": '$^', "cnn_keys": 'RGB', "act": "elu", "norm": "none", "cnn_depth": 48, "cnn_kernels": [5, 5, 6, 6], "mlp_layers": [400, 400, 400, 400]},
    "time_limit": 27000,
    "action_repeat": 1,
    "steps": 5e7,
    "eval_every": 20e3,
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

dv2.train(env, testing_env, config)