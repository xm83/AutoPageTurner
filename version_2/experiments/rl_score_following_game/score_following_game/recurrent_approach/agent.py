import os
import sys
import time
import torch

import numpy as np

from collections import OrderedDict
from score_following_game.reinforcement_learning.torch_extentions.distributions.adapted_categorical import AdaptedCategorical


class Agent(object):

    def __init__(self, observation_space, model, n_actions=1, gamma=0.99, distribution=AdaptedCategorical,
                 use_cuda=torch.cuda.is_available(), log_writer=None, log_interval=10, evaluator=None, eval_interval=5000,
                 lr_scheduler=None, score_name=None, high_is_better=False, dump_interval=100000, dump_dir=None, buffer=None):

        self.observation_space = observation_space
        self.model = model

        self.log_writer = log_writer
        self.log_interval = log_interval

        self.evaluator = evaluator
        self.eval_interval = eval_interval

        self.lr_scheduler = lr_scheduler
        self.score_name = score_name

        self.dump_interval = dump_interval
        self.dump_dir = dump_dir

        self.use_cuda = use_cuda

        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.log_dict = dict()

        self.n_worker = 1
        self.update_cnt = 0
        self.step_cnt = 0
        self.now = self.after = None
        self.step_times = np.ones(11, dtype=np.float32) # 11 is just a random number to have a running avg

        self.distribution = distribution
        self.buffer = buffer

    def perform_update(self):

        # logging
        if self.update_cnt % self.log_interval == 0 and self.update_cnt > 0:
            self.log()

        # evaluation
        if self.evaluator is not None and self.update_cnt % self.eval_interval == 0 and self.update_cnt > 0:
            self.evaluate()

        # dump model regularly
        if self.update_cnt % self.dump_interval == 0 and self.update_cnt > 0:
            print('Saved model at update {}'.format(self.update_cnt))
            self.store_model('model_update_{}'.format(self.update_cnt), self.dump_dir)

        self.update_cnt += 1

        # estimate updates per second (running avg)
        self.step_times[0:-1] = self.step_times[1::]
        self.step_times[-1] = time.time() - self.after
        ups = 1.0 / self.step_times.mean()
        self.after = time.time()
        print("update %d @ %.1fups" % (np.mod(self.update_cnt, self.log_interval), ups), end="\r")
        sys.stdout.flush()

    def store_model(self, name, store_dir=None):

        if store_dir is not None:
            model_path = os.path.join(store_dir, name)
        else:
            model_path = name

        self.model.save_network(model_path)

    def log(self):

        self.log_dict['steps'] = self.step_cnt
        self.log_dict['learn_rate'] = self.model.get_learn_rate()

        print('-' * 32)
        print('| {:<15} {: 12d} |'.format('update', self.update_cnt))
        print('| {:<15} {: 12.1f} |'.format('duration(s)', time.time() - self.now))
        for log_key in self.log_dict:

            log_var = self.log_dict[log_key].cpu().item() if type(self.log_dict[log_key]) == torch.Tensor \
                else self.log_dict[log_key]

            if type(self.log_dict[log_key]) == int:
                print('| {:<15} {: 12d} |'.format(log_key, log_var))
            else:
                print('| {:<15} {: 12.5f} |'.format(log_key, log_var))

            if self.log_writer is not None:
                self.log_writer.add_scalar('training/{}'.format(log_key), log_var,
                                           int(self.update_cnt / self.log_interval))

        if self.log_writer is not None and self.log_writer.log_gradients:

            for tag, value in self.model.net.named_parameters():
                tag = tag.replace('.', '/')
                if value.grad is not None:
                    self.log_writer.add_histogram(tag + '/grad', value.grad.data.cpu().norm(2).item(),  int(self.update_cnt / self.log_interval))

        print('-' * 32)
        self.now = time.time()

    def evaluate(self):
        self.model.set_eval_mode()
        stats = self.evaluator.evaluate(self.trained_agent(self.model, self.use_cuda, distribution=self.distribution),
                                        self.log_writer, int(self.update_cnt / self.eval_interval))

        self.model.set_train_mode()

        if self.score_name is not None:
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(stats[self.score_name])
                if self.model.get_learn_rate() == 0:
                    print('Training stopped')

            improvement = (self.high_is_better and stats[self.score_name] >= self.best_score) or \
                          (not self.high_is_better and stats[self.score_name] <= self.best_score)

            if improvement:
                print('New best model at update {}'.format(self.update_cnt))
                self.store_model('best_model', self.dump_dir)
                self.best_score = stats[self.score_name]