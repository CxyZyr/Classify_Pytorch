import sys
import logging
import collections
import numpy as np


from copy import deepcopy
from torch import distributed as dist
from logging import handlers


def is_dist() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def merge(dict1, dict2):
    ''' Return a new dictionary by merging
        two dictionaries recursively.
    '''
    result = deepcopy(dict1)
    for key, value in dict2.items():
        if isinstance(value, collections.abc.Mapping):
            result[key] = merge(result.get(key, {}), value)
        else:
            result[key] = deepcopy(dict2[key])
    return result


class IterLoader:

    def __init__(self, dataloader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self):
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)

class LoggerBuffer():
    def __init__(self, name, path, headers, screen_intvl=1):
        self.logger = self.get_logger(name, path)
        self.history = []
        self.headers = headers
        self.screen_intvl = screen_intvl

    def get_logger(self, name, path):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
    
        # set project level
        msg_fmt = '[%(levelname)s] %(asctime)s, %(message)s'
        time_fmt = '%Y-%m-%d_%H-%M-%S'
        formatter = logging.Formatter(msg_fmt, time_fmt)
    
        # define file handler and set formatter
        file_handler = logging.FileHandler(path, 'w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

        # to avoid duplicated logging info in PyTorch >1.9
        if len(logger.root.handlers) == 0:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            logger.root.addHandler(stream_handler)
        # to avoid duplicated logging info in PyTorch >1.8
        for handler in logger.root.handlers:
            handler.setLevel(logging.WARNING)


        return logger

    def clean(self):
        self.history = {}

    def update(self,  msg):
        # get the iteration
        n = msg.pop('Iter')
        self.history.append(msg)

        # header expansion
        novel_heads = [k for k in msg if k not in self.headers]
        if len(novel_heads) > 0:
            self.logger.warning(
                    'Items {} are not defined.'.format(novel_heads))

        # missing items
        missing_heads = [k for k in self.headers if k not in msg]
        if len(missing_heads) > 0:
            self.logger.warning(
                    'Items {} are missing.'.format(missing_heads))

        if self.screen_intvl != 1:
            doc_msg = ['Iter: {:5d}'.format(n)]
            for k, fmt in self.headers.items():
                v = self.history[-1][k]
                doc_msg.append(('{}: {'+fmt+'}').format(k, v))
            doc_msg = ', '.join(doc_msg)
            self.logger.debug(doc_msg)

        '''
        construct message to show on screen every `self.screen_intvl` iters
        '''
        if n % self.screen_intvl == 0:
            screen_msg = ['Iter: {:5d}'.format(n)]
            for k, fmt in self.headers.items():
                vals = [msg[k] for msg in self.history[-self.screen_intvl:]
                        if k in msg]
                v = sum(vals) / len(vals)
                screen_msg.append(('{}: {'+fmt+'}').format(k, v))
                    
            screen_msg = ', '.join(screen_msg)
            self.logger.info(screen_msg)


class Logger(object):
    def __init__(self, name,path):
        self.logger = self.get_logger(name,path)

    def get_logger(self, name, path):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # set log level
        msg_fmt = '[%(levelname)s] %(asctime)s, %(message)s'
        time_fmt = '%Y-%m-%d_%H:%M:%S'
        formatter = logging.Formatter(msg_fmt, time_fmt)

        # define file handler and set formatter
        file_handler = logging.FileHandler(path, 'w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

        logger.removeHandler(stream_handler)

        return logger

class CosineDecayLR(object):
    def __init__(self, optimizer, T_max, lr_init, lr_min=0., warmup=0):
        """
        a cosine decay scheduler about steps, not epochs.
        :param optimizer: ex. optim.SGD
        :param T_max:  max steps, and steps=epochs * batches
        :param lr_max: lr_max is init lr.
        :param warmup: in the training begin, the lr is smoothly increase from 0 to lr_init, which means "warmup",
                        this means warmup steps, if 0 that means don't use lr warmup.
        """
        super(CosineDecayLR, self).__init__()
        self.__optimizer = optimizer
        self.__T_max = T_max # max_iter
        self.__lr_min = lr_min # 1e-6
        self.__lr_max = lr_init # 1e-4
        self.__warmup = warmup # 2


    def step(self, t):
        if self.__warmup and t < self.__warmup:
            lr = self.__lr_max / self.__warmup * t
        else:
            T_max = self.__T_max - self.__warmup
            t = t - self.__warmup
            lr = self.__lr_min + 0.5 * (self.__lr_max - self.__lr_min) * (1 + np.cos(t/T_max * np.pi))
        for param_group in self.__optimizer.param_groups:
            param_group["lr"] = lr

