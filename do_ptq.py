import argparse
import os
import torch
import sys
import yaml

script_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from importlib import import_module
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.utils.data import DataLoader
from tqdm import tqdm
from builder import build_norm_dataloader, build_norm_model

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

def calibrate(model, data_loader, calib_num, batch_size):
    model.eval()
    count = 0
    with torch.no_grad():
        for image in tqdm(data_loader):
            image.cpu()
            model(image)
            count += 1 * batch_size
            if count >= calib_num:
                break
    if calib_num % batch_size != 0:
        calib_real = ((calib_num // batch_size) + 1)*batch_size
        print('calib_num was changed from {} to {}'.format(calib_num,calib_real))
    print('Calib Done')

def ptq_pytorch(config):
    # init model
    quant_model = build_norm_model(config['model'])
    quant_model.eval()
    quant_model.apply_to_all_custom_conv()
    quant_model.cpu()
    quant_model.qconfig = torch.quantization.get_default_qconfig(config['ptq']['backend'])
    prepare_model = torch.quantization.prepare(quant_model, inplace=True)

    # Calibrate - Use representative (validation) data.
    ptq_loader = build_norm_dataloader(config['val'])
    calibrate(prepare_model,ptq_loader,config['ptq']['backend']['calib_num'],config['val']['dataloader']['batch_size'])

    # quantize
    print('Quant fx start')
    quantized_model = torch.quantization.convert(quant_model)
    print('Quant fx end')

    # save
    pretrained_path = config['model']['pretrained']
    save_path = os.path.join(os.path.dirname(pretrained_path),'quant'+os.path.basename(pretrained_path))
    input_size = config['val']['dataset']['input_size']
    input_tensor = torch.Tensor(1,3,input_size,input_size)
    traced_model = torch.jit.trace(quantized_model, input_tensor)
    torch.jit.save(traced_model, save_path)

if __name__ == '__main__':
    # get arguments and config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', nargs='+', type=str, default='config/ptq/Head.yml',help='')
    args = parser.parse_args()
    with open(args.config,'r') as f:
        config = yaml.load(f,yaml.SafeLoader)
    ptq_pytorch(config)




