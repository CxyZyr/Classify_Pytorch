import os
import yaml
import argparse
import random
import torch
import torch.distributed as dist

from runner import IterRunner
from qat_runner import QIterRunner


def parse_args():
    parser = argparse.ArgumentParser(
            description='A PyTorch project for classify.')
    parser.add_argument('--config',default='./config/qat/Head.yml',
            help='norm config file path')
    parser.add_argument('--local_rank',default=0,
            help='rank')
    args = parser.parse_args()

    return args


def norm_worker(config):

    # init seed
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # init processes
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    #init runner and run
    runner = IterRunner(config)
    runner.train()

    # clean up
    dist.destroy_process_group()

def qat_worker(config):
    # init seed
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # init runner and run
    runner = QIterRunner(config)
    runner.train()


if __name__ == '__main__':
    # get arguments and config
    args = parse_args()

    with open(args.config,'r') as f:
        config = yaml.load(f,yaml.SafeLoader)

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        raise KeyError('Devices IDs have to be specified.''CPU mode is not supported yet')

    if config['train']['do_qat']:
        qat_worker(config)
    else:
        norm_worker(config)

    # CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nnodes=1 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=12346 train.py --config=config/norm/Head.yml
    # CUDA_VISIBLE_DEVICES=0 python train.py --config=config/qat/Head.yml

