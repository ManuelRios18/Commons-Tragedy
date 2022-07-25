import re
import numpy as np
import dreamerv2.common as common


class DreamerCallbacks:
    
    def __init__(self, config, logger, step, should_video_train, should_video_eval, should_train, should_log,
                 train_replay, eval_replay, metrics):
        self.config = config
        self.logger = logger
        self.step = step
        self.should_video_train = should_video_train
        self.should_video_eval = should_video_eval
        self.should_train = should_train
        self.should_log = should_log
        self.train_replay = train_replay
        self.eval_replay = eval_replay
        self.metrics = metrics
        self.agent = None
        self.train_dataset = None
        self.train_agent = None
        self.report_dataset = None

    def add_agent_and_dataset(self, agent, train_dataset, train_agent, report_dataset):
        self.agent = agent
        self.train_dataset = train_dataset
        self.train_agent = train_agent
        self.report_dataset = report_dataset

    def make_env(self):
        suite, task = self.config.task.split('_', 1)
        if suite == 'atari':
            env = common.Atari(task, self.config.action_repeat, self.config.render_size, self.config.atari_grayscale)
            env = common.OneHotAction(env)
        else:
            raise NotImplementedError(suite)
        env = common.TimeLimit(env, self.config.time_limit)
        return env
    
    def per_episode(self, ep, mode):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
        self.logger.scalar(f'{mode}_return', score)
        self.logger.scalar(f'{mode}_length', length)
        for key, value in ep.items():
            if re.match(self.config.log_keys_sum, key):
                self.logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
            if re.match(self.config.log_keys_mean, key):
                self.logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
            if re.match(self.config.log_keys_max, key):
                self.logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
        should = {'train': self.should_video_train, 'eval': self.should_video_eval}[mode]
        if should(self.step):
            for key in self.config.log_keys_video:
                self.logger.video(f'{mode}_policy_{key}', ep[key])
        replay = dict(train=self.train_replay, eval=self.eval_replay)[mode]
        self.logger.add(replay.stats, prefix=mode)
        self.logger.write()
    
    def train_step(self, tran, worker):
        if self.should_train(self.step):
            for _ in range(self.config.train_steps):
                mets = self.train_agent(next(self.train_dataset))
                [self.metrics[key].append(value) for key, value in mets.items()]
        #print("checkin if should log")
        if self.should_log(self.step):
            #print("Logging")
            for name, values in self.metrics.items():
                self.logger.scalar(name, np.array(values, np.float64).mean())
                self.metrics[name].clear()
            self.logger.add(self.agent.report(next(self.report_dataset)), prefix='train')
            self.logger.write(fps=True)
