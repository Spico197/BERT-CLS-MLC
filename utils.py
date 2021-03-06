import os
import json
import random
from ast import literal_eval
from configparser import ConfigParser

import torch
import numpy as np


def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config(object):
    def __init__(self, conf=None, **kwargs):
        super().__init__()

        if conf.endswith('.json'):
            config = json.loads(open(conf).read())
            self.update(config)
        elif conf.endswith('.ini'):
            config = ConfigParser()
            config.read(conf or [])
            self.update({**dict((name, literal_eval(value))
                                for section in config.sections()
                                for name, value in config.items(section)),
                        **kwargs})
        else:
            raise ValueError('Not supported config file')

    def __repr__(self):
        s = line = "-" * 20 + "-+-" + "-" * 30 + "\n"
        s += f"{'Param':20} | {'Value':^30}\n" + line
        for name, value in vars(self).items():
            s += f"{name:20} | {str(value):^30}\n"
        s += line

        return s

    def __getitem__(self, key):
        return getattr(self, key)

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def keys(self):
        return vars(self).keys()

    def items(self):
        return vars(self).items()

    def update(self, kwargs):
        for key in ('self', 'cls', '__class__'):
            kwargs.pop(key, None)
        kwargs.update(kwargs.pop('kwargs', dict()))
        for name, value in kwargs.items():
            setattr(self, name, value)

        return self

    def pop(self, key, val=None):
        return self.__dict__.pop(key, val)


if __name__ == "__main__":
    config = Config("config.ini")
    print(config.num_epoch)
