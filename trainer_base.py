import abc
import torch
import os
import wandb
import math

from models import dataloader as dataloader_hub
from models import lr_scheduler
from models import model_implements
from models import losses as loss_hub
from models import utils
from models import metrics

from datetime import datetime


class TrainerBase:
    __metaclass__ = abc.ABCMeta

    def __init__(self, args, now=None, k_fold=0):
        self.args = args
        now_time = now if now is not None else datetime.now().strftime("%Y%m%d %H%M%S")
        self.saved_model_directory = self.args.saved_model_directory + '/' + now_time
        self.k_fold = k_fold

        # save hyper-parameters
        if not self.args.debug:
            with open(self.args.config_path, 'r') as f_r:
                file_path = self.args.saved_model_directory + '/' + now_time
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                with open(os.path.join(file_path, self.args.config_path.split('/')[-1]), 'w') as f_w:
                    f_w.write(f_r.read())

        # Check cuda available and assign to device
        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.model = self.init_model(self.args.model_name, self.device, self.args)
        self.optimizer = self.init_optimizer(self.args.optimizer, self.model, self.args.lr)
        self.criterion = self.init_criterion(self.args.criterion)
        self.metric_train = self.init_metric(self.args.task, self.args.num_class)
        self.metric_val = self.init_metric(self.args.task, self.args.num_class)

        if hasattr(self.args, 'model_path'):
            if self.args.model_path != '':
                if 'imagenet' in self.args.model_path.lower():
                    self.model.module.load_pretrained_imagenet(self.args.model_path)
                    print('Model loaded successfully!!! (ImageNet)')
                else:
                    self.model.module.load_pretrained(self.args.model_path)
                    print('Model loaded successfully!!! (Custom)')
                self.model.to(self.device)

        if self.args.wandb:
            wandb.init(project='{}'.format(args.project_name), config=args, name=now_time,
                       settings=wandb.Settings(start_method="fork"))
            wandb.watch(self.model)

        self.model_post_path_dict = {}
        self.last_saved_epoch = 0
        self.callback = utils.TrainerCallBack()
        if hasattr(self.model.module, 'train_callback'):
            self.callback.train_callback = self.model.module.train_callback

    @abc.abstractmethod
    def _train(self, epoch):
        pass

    @abc.abstractmethod
    def _validate(self, model, epoch):
        pass

    @abc.abstractmethod
    def run(self):
        pass

    def save_model(self, model, model_name, epoch, metric=None, best_flag=False, metric_name='metric'):
        file_path = self.saved_model_directory + '/'

        file_format = file_path + model_name + '-Epoch_' + str(epoch) + '-' + metric_name + '_' + str(metric)[:6] + '-folds_' + str(self.k_fold) + '.pt'

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        if best_flag:
            if metric_name in self.model_post_path_dict.keys():
                os.remove(self.model_post_path_dict[metric_name])
            self.model_post_path_dict[metric_name] = file_format

        torch.save(model.state_dict(), file_format)

        print(f'{utils.Colors.LIGHT_RED}{file_format} model saved!!{utils.Colors.END}')
        self.last_saved_epoch = epoch

    def init_data_loader(self,
                         batch_size,
                         mode,
                         dataloader_name,
                         x_path=None,
                         y_path=None,
                         csv_path=None):

        if dataloader_name == 'Image2Image':
            loader = dataloader_hub.Image2ImageDataLoader(x_path=x_path,
                                                          y_path=y_path,
                                                          batch_size=batch_size,
                                                          num_workers=self.args.worker,
                                                          pin_memory=self.args.pin_memory,
                                                          mode=mode,
                                                          args=self.args)
        elif dataloader_name == 'Image2Vector':
            loader = dataloader_hub.Image2VectorDataLoader(csv_path=csv_path,
                                                           batch_size=batch_size,
                                                           num_workers=self.args.worker,
                                                           pin_memory=self.args.pin_memory,
                                                           mode=mode,
                                                           args=self.args)
        else:
            raise Exception('No dataloader named', dataloader_name)

        return loader

    def set_scheduler(self, optimizer, scheduler_name, data_loader, batch_size):
        scheduler = None
        steps_per_epoch = math.ceil((data_loader.__len__() / batch_size))

        if hasattr(self.args, 'scheduler'):
            if scheduler_name == 'WarmupCosine':
                scheduler = lr_scheduler.WarmupCosineSchedule(optimizer=optimizer,
                                                              warmup_steps=steps_per_epoch * self.args.warmup_epoch,
                                                              t_total=self.args.epoch * steps_per_epoch,
                                                              cycles=self.args.epoch / self.args.cycles,
                                                              last_epoch=-1)
            elif scheduler_name == 'CosineAnnealing':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.cycles, eta_min=self.args.lr_min)
            elif scheduler_name == 'Constant':
                scheduler = lr_scheduler.ConstantLRSchedule(optimizer, last_epoch=-1)
            elif scheduler_name == 'WarmupConstant':
                scheduler = lr_scheduler.WarmupConstantSchedule(optimizer, warmup_steps=steps_per_epoch * self.args.warmup_epoch)
            else:
                print(f'{utils.Colors.LIGHT_PURPLE}No scheduler found --> {scheduler_name}{utils.Colors.END}')
        else:
            pass

        return scheduler

    @staticmethod
    def init_model(model_name, device, args):
        model = getattr(model_implements, model_name)(**vars(args)).to(device)

        return torch.nn.DataParallel(model)

    def init_criterion(self, criterion_name):
        criterion = getattr(loss_hub, criterion_name)().to(self.device)

        return criterion

    def init_optimizer(self, optimizer_name, model, lr):
        optimizer = None

        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.weight_decay)

        return optimizer

    @staticmethod
    def init_metric(task_name, num_class):
        if task_name == 'segmentation':
            metric = metrics.StreamSegMetrics_segmentation(num_class)
        elif task_name == 'classification':
            metric = metrics.StreamSegMetrics_classification(num_class)
        elif task_name == 'regression':
            metric = metrics.StreamSegMetrics_classification(num_class)
        else:
            raise Exception('No task named', task_name)

        return metric
