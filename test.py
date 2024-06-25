import argparse
import os.path
import yaml
import torch
from importlib import import_module
from torch.utils.data import DataLoader
from tqdm import tqdm
from builder import build_norm_dataloader,build_norm_model

# Set up warnings
import warnings

warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)

@torch.no_grad()
def val(model, dataloader, device):
    correct = 0
    total = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total

    return acc

def main(config):
    if config['device'] != 'cpu':
        torch.cuda.set_device(config['device'])
        device = torch.device("cuda:{}".format(config['device']))
    else:
        device = torch.device("cpu")

    # init model
    if config['val']['quant_model']:
        model = torch.jit.load(config['model']['pretrained'])
    else:
        model = build_norm_model(config['model'])

    # init data
    val_loader = build_norm_dataloader(config['val'])

    acc = val(model, val_loader, device)
    print('ACC: {:.2f}'.format(acc))


if __name__ == '__main__':
    # get arguments and config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/val/Head.yml', help='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.SafeLoader)
    main(config)



