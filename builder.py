from __future__ import print_function
import copy
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Sampler
from utils import get_world_size, get_rank
from importlib import import_module
from sys import getsizeof, stderr
from itertools import chain
from collections import deque

try:
    from reprlib import repr
except ImportError:
    pass


def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.
    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:
        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)

def merge_weight(pretrain_dict):
    origin_dict = pretrain_dict
    merge_dict = {}

    last_model_dict = {}
    for k, v in origin_dict.items():
        if 'module' in k[:8]:
            single_key = k.replace('module.', '', 1)
            last_model_dict[single_key] = v
        else:
            last_model_dict[k] = v

    for k, v in last_model_dict.items():
        if 'linear.weight' == k:
            weight = v
        elif 'bn.weight' == k:
            gamma = v
        elif 'bn.bias' == k:
            beta = v
        elif 'bn.running_mean' == k:
            running_mean = v
        elif 'bn.running_var' == k:
            running_var = v
        elif 'bn.num_batches_tracked' == k:
            continue
        else:
            merge_dict[k] = v

    std = (running_var + 1e-5).sqrt()
    t = (gamma / std).reshape(-1, 1)

    merged_weight = weight * t
    merged_bias = beta - running_mean * gamma / std
    merge_dict['merge_linear.weight'] = merged_weight
    merge_dict['merge_linear.bias'] = merged_bias

    return merge_dict


def build_from_cfg(cfg, module):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        object: The constructed object.
    """
    if not isinstance(cfg, dict):
        raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
    if 'type' not in cfg:
        raise KeyError(f'`cfg` must contain the key "type", but got {cfg}')

    args = cfg.copy()

    obj_type = args.pop('type')
    if not isinstance(obj_type, str):
        raise TypeError(f'type must be a str, but got {type(obj_type)}')
    else:
        obj_cls = getattr(import_module(module), obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not in the {module} module')

    return obj_cls(**args)




def build_dataloader(cfg):
    """ build a dataloader or a list of dataloaders
    Args:
        cfg: a dict for a dataloader,
             or a list or a tuple of dicts for multiple dataloaders.
    Returns:
        dataloader(s): PyTorch dataloader(s)
    """
    if isinstance(cfg, (list, tuple)):
        return [build_dataloader(c) for c in cfg]

    if 'dataset' not in cfg:
        raise KeyError('Missing key "dataset" in `cfg`', cfg)
    if 'dataloader' not in cfg:
        raise KeyError('Missing key "dataloader" in `cfg`', cfg)
    args = copy.deepcopy(cfg)
    # dataset
    args['dataset']['data_dir'] = args['data_dir']
    args['dataset']['ann_path'] = args['ann_path']
    data = build_from_cfg(args['dataset'], 'datasets')
    world_size = get_world_size()
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            data, shuffle=cfg['dataloader']['shuffle'])
        args['dataloader']['shuffle'] = False  # shuffle is now done by sampler
    else:
        sampler = None
    args['dataloader']['dataset'] = data
    args['dataloader']['sampler'] = sampler
    dataloader = build_from_cfg(args['dataloader'], 'torch.utils.data')

    return dataloader, sampler

def build_norm_dataloader(cfg):
    """ build a dataloader or a list of dataloaders
    Args:
        cfg: a dict for a dataloader,
             or a list or a tuple of dicts for multiple dataloaders.
    Returns:
        dataloader(s): PyTorch dataloader(s)
    """
    if isinstance(cfg, (list, tuple)):
        return [build_dataloader(c) for c in cfg]

    if 'dataset' not in cfg:
        raise KeyError('Missing key "dataset" in `cfg`', cfg)
    if 'dataloader' not in cfg:
        raise KeyError('Missing key "dataloader" in `cfg`', cfg)
    args = copy.deepcopy(cfg)
    # dataset
    args['dataset']['data_dir'] = args['data_dir']
    args['dataset']['ann_path'] = args['ann_path']
    data = build_from_cfg(args['dataset'], 'datasets')
    args['dataloader']['dataset'] = data
    args['dataloader']['sampler'] = None
    dataloader = build_from_cfg(args['dataloader'], 'torch.utils.data')

    return dataloader


def build_model(cfg):
    if 'net' not in cfg:
        raise KeyError(f'`cfg` must contain the key "net", but got {cfg}')
    rank = get_rank()
    net = build_from_cfg(cfg['net'], 'networks')
    net = net.to(rank)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    if 'pretrained' in cfg:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pretrain_dict = torch.load(cfg['pretrained'],map_location=map_location)
        last_model_dict = {}
        for k, v in pretrain_dict.items():
            if 'module' not in k[:8]:
                last_model_dict['module.' + k] = v
            else:
                last_model_dict[k] = v
        net.load_state_dict(last_model_dict)
        print('-----Load pretrain is Successful-----')

    if 'clip_grad_norm' not in cfg:
        cfg['clip_grad_norm'] = 1e5
        warnings.warn('`clip_grad_norm` is not set. The default is 1e5')

    return net

def build_norm_model(cfg):
    if 'net' not in cfg:
        raise KeyError(f'`cfg` must contain the key "net", but got {cfg}')
    net = build_from_cfg(cfg['net'], 'networks')

    if 'pretrained' in cfg:
        pretrain_dict = torch.load(cfg['pretrained'],map_location='cpu')
        if 'QATMiniFAS' in cfg['net']['type']:
            pretrain_dict = merge_weight(pretrain_dict)
            print('-----Merge BatchNorm1d and Linear for QAT training task if use MiniFAS type-----')
        last_model_dict = {}
        for k, v in pretrain_dict.items():
            if 'module' in k[:8]:
                single_key = k.replace('module.','',1)
                last_model_dict[single_key] = v
            else:
                last_model_dict[k] = v
        net.load_state_dict(last_model_dict)
        print('-----Load FP32 is Successful-----')

    if 'clip_grad_norm' not in cfg:
        cfg['clip_grad_norm'] = 1e5
        warnings.warn('`clip_grad_norm` is not set. The default is 1e5')

    return net


