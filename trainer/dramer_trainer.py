import collections
import functools
import logging
import os
import pathlib
import re
import sys
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

import numpy as np
import ruamel.yaml as yaml

import dreamerv2.agent as dreamer_agent
import dreamerv2.common as dreamer_common
import tensorflow as tf
from trainer.dreamer_callbacks import DreamerCallbacks


class DreamerTrainer:

    def __init__(self):
        configs = yaml.safe_load(pathlib.Path('config/dreamer_configs.yaml').read_text())

        #configs = yaml.safe_load((pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
        parsed, remaining = dreamer_common.Flags(configs=['defaults']).parse(known_only=True)

        config = dreamer_common.Config(configs['defaults'])
        config = config.update(dreamer_common.Config(configs['atari']))
        self.config = dreamer_common.Flags(config).parse(remaining)

        #self.config = dreamer_common.Config(configs["defaults"])
        self.logdir = pathlib.Path(self.config.logdir).expanduser()
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.config.save(self.logdir / 'config.yaml')
        print(self.config, '\n')
        print('Logdir', self.logdir)

        tf.config.experimental_run_functions_eagerly(not self.config.jit)
        message = 'No GPU found. To actually train on CPU remove this assert.'
        assert tf.config.experimental.list_physical_devices('GPU'), message
        for gpu in tf.config.experimental.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
        assert self.config.precision in (16, 32), self.config.precision
        if self.config.precision == 16:
            import tensorflow.keras.mixed_precision as prec
            #prec.set_global_policy(prec.Policy('mixed_float16'))
            print("precision was set")

        self.train_replay = dreamer_common.Replay(self.logdir / 'train_episodes', **self.config.replay)
        self.eval_replay = dreamer_common.Replay(self.logdir / 'eval_episodes', **dict(
            capacity=self.config.replay.capacity // 10,
            minlen=self.config.dataset.length,
            maxlen=self.config.dataset.length))
        self.step = dreamer_common.Counter(self.train_replay.stats['total_steps'])
        outputs = [
            dreamer_common.TerminalOutput(),
            dreamer_common.JSONLOutput(self.logdir),
            dreamer_common.TensorBoardOutput(self.logdir),
        ]
        self.logger = dreamer_common.Logger(self.step, outputs, multiplier=self.config.action_repeat)
        self.metrics = collections.defaultdict(list)

        self.should_train = dreamer_common.Every(self.config.train_every)
        self.should_log = dreamer_common.Every(self.config.log_every)
        print("Should log every ", self.should_log._every)
        self.should_video_train = dreamer_common.Every(self.config.eval_every)
        self.should_video_eval = dreamer_common.Every(self.config.eval_every)
        self.should_expl = dreamer_common.Until(self.config.expl_until // self.config.action_repeat)

        self.dreamer_callbacks = DreamerCallbacks(config=self.config, logger=self.logger, step=self.step,
                                                  should_video_train=self.should_video_train,
                                                  should_video_eval=self.should_video_eval,
                                                  should_train=self.should_train, should_log=self.should_log,
                                                  train_replay=self.train_replay, eval_replay=self.eval_replay,
                                                  metrics=self.metrics)
        print('Creating envs.')
        num_eval_envs = min(self.config.envs, self.config.eval_eps)
        self.train_envs = [self.dreamer_callbacks.make_env() for _ in range(self.config.envs)]
        self.eval_envs = [self.dreamer_callbacks.make_env() for _ in range(num_eval_envs)]

        act_space = self.train_envs[0].act_space
        obs_space = self.train_envs[0].obs_space

        self.train_driver = dreamer_common.Driver(self.train_envs)
        self.train_driver.on_episode(lambda ep: self.dreamer_callbacks.per_episode(ep, mode='train'))
        self.train_driver.on_step(lambda tran, worker: self.step.increment())
        self.train_driver.on_step(self.train_replay.add_step)
        self.train_driver.on_reset(self.train_replay.add_step)

        self.eval_driver = dreamer_common.Driver(self.eval_envs)
        self.eval_driver.on_episode(lambda ep: self.dreamer_callbacks.per_episode(ep, mode='eval'))
        self.eval_driver.on_episode(self.eval_replay.add_episode)

        prefill = max(0, self.config.prefill - self.train_replay.stats['total_steps'])
        if prefill:
            print(f'Prefill dataset ({prefill} steps).')
            random_agent = dreamer_common.RandomAgent(act_space)
            self.train_driver(random_agent, steps=prefill, episodes=1)
            self.eval_driver(random_agent, episodes=1)
            self.train_driver.reset()
            self.eval_driver.reset()

        print('Create agent.')

        self.train_dataset = iter(self.train_replay.dataset(**self.config.dataset))
        self.report_dataset = iter(self.train_replay.dataset(**self.config.dataset))
        self.eval_dataset = iter(self.eval_replay.dataset(**self.config.dataset))
        self.agent = dreamer_agent.Agent(self.config, obs_space, act_space, self.step)
        train_agent = dreamer_common.CarryOverState(self.agent.train)
        train_agent(next(self.train_dataset))
        if (self.logdir / 'variables.pkl').exists():
            self.agent.load(self.logdir / 'variables.pkl')
        else:
            print('Pretrain agent.')
            for _ in range(self.config.pretrain):
                train_agent(next(self.train_dataset))
        self.train_policy = lambda *args: self.agent.policy(*args, mode='explore' if self.should_expl(self.step) else 'train')
        self.eval_policy = lambda *args: self.agent.policy(*args, mode='eval')
        self.dreamer_callbacks.add_agent_and_dataset(agent=self.agent,  train_dataset=self.train_dataset,
                                                     train_agent=train_agent, report_dataset=self.report_dataset)
        self.train_driver.on_step(self.dreamer_callbacks.train_step)

        print("Done until here")

    def start_training(self):
        while self.step < self.config.steps:
            self.logger.write()
            print('Start evaluation.')
            self.logger.add(self.agent.report(next(self.eval_dataset)), prefix='eval')
            self.eval_driver(self.eval_policy, episodes=self.config.eval_eps)
            print('Start training.')
            self.train_driver(self.train_policy, steps=self.config.eval_every)
            self.agent.save(self.logdir / 'variables.pkl')
        for env in self.train_envs + self.eval_envs:
            try:
                env.close()
            except Exception:
                pass