import os
import os.path as osp
import time
import copy
import yaml
import warnings
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from builder import build_norm_dataloader, build_norm_model
from utils import Logger,CosineDecayLR

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

class QIterRunner():
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.distill = self.config['distill']['distill_train']

        # init dataloader
        self.train_dataloader = build_norm_dataloader(self.config['train'])
        self.val_dataloader = build_norm_dataloader(self.config['val'])

        # init model
        self.model = build_norm_model(config['model'])

        # init distillation
        if self.distill:
            self.teacher_model = build_norm_model(self.config['distill']['distill_model'])
            self.teacher_model.eval()
            self.distill_criterion = self.loss_fn_kd

        # init project
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.project_dir = osp.join(config['common']['save_log_dir'],timestamp)
        os.makedirs(self.project_dir,exist_ok=True)

        print('')
        print('The training log and models are saved to ' + self.project_dir)
        print('')

        # save cfg
        save_cfg_path = osp.join(self.project_dir,config['common']['save_cfg_name'])
        with open(save_cfg_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=None)

        # save log
        save_log_dir = osp.join(self.project_dir, 'log')
        os.makedirs(save_log_dir, exist_ok=True)
        self.train_log = Logger(name='norm', path="{}/{}_train.log".format(save_log_dir,time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())))
        self.val_log = Logger(name='val', path="{}/{}_val.log".format(save_log_dir,time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime())))

        #save weight
        self.save_weights_dir = osp.join(self.project_dir,'weights')
        os.makedirs(self.save_weights_dir, exist_ok=True)

        # init common and norm arguments
        self.epoch = self.config['train']['epoch']
        self.test_first = self.config['common']['test_first']
        self.screen_intvl = self.config['common']['screen_intvl']
        self.val_intvl = self.config['common']['val_intvl']
        self.iter_step = self.config['train']['optim']['iter_step']
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.input_size = config['train']['dataset']['input_size']
        self.optimizer = None
        self.scheduler = None
        self.scheduler_type = None
        self.acc = 0
        self.iter = 0

    def set_optimizer_scheduler(self,config):
        if config['optim']['type'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=config['optim']['lr_init'],
                                        weight_decay=config['optim']['weight_decay'])
        else:
            self.optimizer = optim.SGD(self.model.parameters(),
                          lr=config['optim']['lr_init'],
                          momentum=config['optim']['momentum'],
                          weight_decay=config['optim']['weight_decay'])

        if config['scheduler']['type'] == 'CosineDecayLR':
            self.scheduler_type = 'CosineDecayLR'
            self.scheduler = CosineDecayLR(
                self.optimizer,
                T_max=config['epoch']*len(self.train_dataloader),
                lr_init=config['optim']['lr_init'],
                lr_min=config['scheduler']['lr_end'],
                warmup=config['scheduler']['warm_up_epoch']*len(self.train_dataloader)
            )
        if config['scheduler']['type'] == 'MultiStepLR':
            self.scheduler_type = 'MultiStepLR'
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                config['scheduler']['milestones'],
                config['scheduler']['gamma'],
                -1
            )

    def loss_fn_kd(self,outputs, labels, teacher_outputs, alpha, T):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha

        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """
        KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                                 F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
                  F.cross_entropy(outputs, labels) * (1. - alpha)

        return KD_loss

    def init_qat(self):
        self.model.eval()
        self.model.apply_to_all_custom_conv()
        self.model.train()
        quantization_config = torch.quantization.get_default_qat_qconfig('fbgemm')
        self.model.qconfig = quantization_config
        torch.quantization.prepare_qat(self.model, inplace=True)
        self.model.to(self.device)

    def set_model(self, test_mode):
        if test_mode:
            self.model.eval()
        else:
            self.model.train()

    def update_model(self, i):
        if i % self.iter_step == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler_type == 'CosineDecayLR':
                self.scheduler.step(self.iter)
            else:
                self.scheduler.step()


    def save_model(self,input_size,acc):
        model_int8_name = 'ConvertInt8_Iter{}_ACC{:.2f}.pth'.format(str(self.iter + 1), acc)
        model_int8_path = osp.join(self.save_weights_dir, model_int8_name)
        model_int8 = copy.deepcopy(self.model).cpu()
        model_int8 = torch.quantization.convert(model_int8, inplace=False)
        traced_model = torch.jit.trace(model_int8, torch.Tensor(1, 3, input_size, input_size).cpu())
        torch.jit.save(traced_model, model_int8_path)
        model_name = 'TrueQuant_Iter{}_ACC{:.2f}.pth'.format(str(self.iter + 1), acc)
        model_path = osp.join(self.save_weights_dir, model_name)
        torch.save(model_int8.state_dict(), model_path)
        del model_int8,traced_model

    @torch.no_grad()
    def val(self):
        # switch to test mode
        self.set_model(test_mode=True)
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
        self.val_log.logger.info("Processing Val Iter:{} ACC : {}]".format(self.iter+1, acc))
        # if model have acc better in the test data,save the model
        if acc >= self.acc:
            self.save_model(input_size=self.input_size,acc=acc)
            self.acc = acc

    def train(self):
        if self.test_first:
            self.val()
        self.init_qat()
        self.set_optimizer_scheduler(self.config['train'])
        for epoch in range(self.epoch):
            Loss = 0
            self.set_model(test_mode=False)
            for i,(images,labels) in enumerate(self.train_dataloader):
                # Implementing gradient accumulation for multi-GPU training on a single machine.

                images, labels = images.to(self.device), labels.to(self.device)
                # forward
                pred = self.model(images)
                loss = self.criterion(pred, labels)

                if self.distill:
                    teacher_pred = self.teacher_model(images)
                    distill_loss = self.distill_criterion(pred, labels, teacher_pred, 0.9, 20)
                    loss = distill_loss

                # backward
                loss.backward()
                Loss = (Loss * i + loss.item()) / (i + 1)

                # update model
                self.iter = epoch*len(self.train_dataloader)+i
                self.update_model(i)

                if (i + 1) % self.screen_intvl == 0 or (i + 1) == len(self.train_dataloader):

                    self.train_log.logger.info("Processing Training Epoch:[{} | {}] Batch:[{} | {}] Lr:{:.6f} Loss:{:.4f} "
                        .format(epoch+1, self.epoch, i+1, len(self.train_dataloader), self.optimizer.param_groups[0]['lr'], Loss))

                # do test and save
                if (i + 1) % self.val_intvl == 0 or (i + 1) == len(self.train_dataloader):
                    self.val()

                # Delete temporary variables to prevent memory leaks.
                del loss
                del images, labels

                if self.distill:
                    del teacher_pred, distill_loss







