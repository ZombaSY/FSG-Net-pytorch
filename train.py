import torch
import time
import os
import wandb
import numpy as np
import sys
import math

from models import dataloader as dataloader_hub
from models import lr_scheduler
from models import model_implements
from models import losses as loss_hub
from models import metrics


class Trainer_seg:
    def __init__(self, args, now_time=None):
        self.start_time = time.time()
        self.args = args

        # save hyper-parameters
        if not self.args.debug:
            with open(self.args.config_path, 'r') as f_r:
                file_path = self.args.saved_model_directory + '/' + now_time
                if not os.path.exists(file_path):
                    os.makedirs(file_path)
                with open(os.path.join(file_path, self.args.config_path.split('/')[-1]), 'w') as f_w:
                    f_w.write(f_r.read())

        if args.wandb:
            # wandb.login(key='your_WandB_key')
            wandb.init(project='{}'.format(args.project_name), config=args, name=now_time,
                    settings=wandb.Settings(start_method="fork"))

        # Check cuda available and assign to device
        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        # 'init' means that this variable must be initialized.
        # 'set' means that this variable is available of being set, not must.
        self.loader_train = self.__init_data_loader(self.args.train_x_path,
                                                    self.args.train_y_path,
                                                    self.args.batch_size,
                                                    mode='train')
        self.loader_val = self.__init_data_loader(self.args.val_x_path,
                                                  self.args.val_y_path,
                                                  batch_size=1,
                                                  mode='validation')

        self.model = self.init_model(self.args.model_name, self.device, self.args)
        self.optimizer = self.__init_optimizer(self.args.optimizer, self.model, self.args.lr)
        self.scheduler = self.__set_scheduler(self.optimizer, self.args.scheduler, self.loader_train, self.args.batch_size)

        if hasattr(self.args, 'model_path'):
            if self.args.model_path != '':
                self.model.load_state_dict(torch.load(self.args.model_path))
                print('Model loaded successfully!!! (Custom)')
                self.model.to(self.device)

        self.criterion = self.__init_criterion(self.args.criterion)

        if self.args.wandb:
            if self.args.mode == 'train':
                wandb.watch(self.model)


        self.saved_model_directory = self.args.saved_model_directory + '/' + now_time

        self.metric_best = {'f1_score': 0}  # the 'value' follows the metric on validation
        self.model_post_path_dict = {}
        self.last_saved_epoch = 0

        self.__validate_interval = 1 if (self.loader_train.__len__() // self.args.train_fold) == 0 else self.loader_train.__len__() // self.args.train_fold

    def _train(self, epoch):
        self.model.train()
        batch_losses = []
        f1_list = []

        print('Start Train')
        for batch_idx, (x_in, target) in enumerate(self.loader_train.Loader):
            if (x_in[0].shape[0] / torch.cuda.device_count()) <= torch.cuda.device_count():   # if has 1 batch per GPU
                break   # avoid BN issue

            x_in, _ = x_in
            target, _ = target
            x_in = x_in.to(self.device)
            target = target.long().to(self.device)

            output = self.model(x_in)

            if isinstance(output, tuple) or isinstance(output, list):  # condition for deep supervision
                loss = sum([self.criterion(item, target) for item in output]) / len(output)
                output = output[0]  # please check that the first element of tuple is not interpolated in network.
            else:
                loss = self.criterion(output, target)

            if not torch.isfinite(loss):
                raise Exception('Loss is NAN. End training.')

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            output_argmax = torch.where(output > 0.5, 1, 0).cpu()
            metric_result = metrics.metrics_np(output_argmax[:, 0], target.squeeze(0).detach().cpu().numpy(), b_auc=False)
            f1_list.append(metric_result['f1'])
            batch_losses.append(loss.item())

            if hasattr(self.args, 'train_fold'):
                if batch_idx != 0 and (batch_idx % self.__validate_interval) == 0:
                    self._validate(self.model, epoch)

            if (batch_idx != 0) and (batch_idx % (self.args.log_interval // self.args.batch_size) == 0):
                loss_mean = np.mean(batch_losses)
                print('{} epoch / Train Loss {} : {}, lr {}'.format(epoch,
                                                                    self.args.criterion,
                                                                    loss_mean,
                                                                    self.optimizer.param_groups[0]['lr']))

        loss_mean = np.mean(batch_losses)
        print('{} epoch / Train Loss {} : {}, lr {}'.format(epoch,
                                                            self.args.criterion,
                                                            loss_mean,
                                                            self.optimizer.param_groups[0]['lr']))
        f1_score = sum(f1_list) / len(f1_list)
        print('{} epoch / Train f1_score: {}'.format(epoch, f1_score))
        if self.args.wandb:
            wandb.log({'Train Loss {}'.format(self.args.criterion): loss_mean,
                       'Train f1_score': f1_score},
                      step=epoch)

    def _validate(self, model, epoch):
        model.eval()
        f1_list = []

        with torch.no_grad():
            for batch_idx, (x_in, target) in enumerate(self.loader_val.Loader):
                x_in, _ = x_in
                target, _ = target
                x_in = x_in.to(self.device)
                target = target.long().to(self.device)

                output = model(x_in)

                if isinstance(output, tuple) or isinstance(output, list):  # condition for Deep supervision
                    output = output[0]

                output_argmax = torch.where(output > 0.5, 1, 0).cpu()
                metric_result = metrics.metrics_np(output_argmax[:, 0], target.squeeze(0).detach().cpu().numpy(), b_auc=False)
                f1_list.append(metric_result['f1'])

        f1_score = sum(f1_list) / len(f1_list)
        print('{} epoch / Val f1_score: {}'.format(epoch, f1_score))
        if self.args.wandb:
            wandb.log({'Val f1_score': f1_score},
                      step=epoch)

        model_metrics = {'f1_score': f1_score}

        for key in model_metrics.keys():
            if model_metrics[key] > self.metric_best[key]:
                self.metric_best[key] = model_metrics[key]
                self.save_model(model, self.args.model_name, epoch, model_metrics[key], best_flag=True, metric_name=key)

        if (epoch - self.last_saved_epoch) > self.args.cycles * 4:
            print('The model seems to be converged. Early stop training.')
            print(f'Best F1 -----> {self.metric_best["f1_score"]}')
            wandb.log({f'Best F1': self.metric_best['f1_score']},
                      step=epoch)
            sys.exit()  # safe exit

    def start_train(self):
        for epoch in range(1, self.args.epoch + 1):
            self._train(epoch)
            self._validate(self.model, epoch)

            print('### {} / {} epoch ended###'.format(epoch, self.args.epoch))

    def save_model(self, model, model_name, epoch, metric=None, best_flag=False, metric_name='metric'):
        file_path = self.saved_model_directory + '/'

        file_format = file_path + model_name + '-Epoch_' + str(epoch) + '-' + metric_name + '_' + str(metric) + '.pt'

        if not os.path.exists(file_path):
            os.mkdir(file_path)

        if best_flag:
            if metric_name in self.model_post_path_dict.keys():
                os.remove(self.model_post_path_dict[metric_name])
            self.model_post_path_dict[metric_name] = file_format

        torch.save(model.state_dict(), file_format)

        print(file_format + '\t model saved!!')
        self.last_saved_epoch = epoch

    def __init_data_loader(self,
                           x_path,
                           y_path,
                           batch_size,
                           mode):

        if self.args.dataloader == 'Image2Image_zero_pad':
            loader = dataloader_hub.Image2ImageDataLoader_zero_pad(x_path=x_path,
                                                                   y_path=y_path,
                                                                   batch_size=batch_size,
                                                                   num_workers=self.args.worker,
                                                                   pin_memory=self.args.pin_memory,
                                                                   mode=mode,
                                                                   args=self.args)
        else:
            raise Exception('No dataloader named', self.args.dataloader)

        return loader

    @staticmethod
    def init_model(model_name, device, args):
        model = getattr(model_implements, model_name)(**vars(args)).to(device)

        return torch.nn.DataParallel(model)

    def __init_criterion(self, criterion_name):
        criterion = getattr(loss_hub, criterion_name)().to(self.device)

        return criterion

    def __init_optimizer(self, optimizer_name, model, lr):
        optimizer = None

        if optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.weight_decay)

        return optimizer

    def __set_scheduler(self, optimizer, scheduler_name, data_loader, batch_size):
        scheduler = None
        steps_per_epoch = math.ceil((data_loader.__len__() / batch_size))

        if hasattr(self.args, 'scheduler'):
            if scheduler_name == 'WarmupCosine':
                scheduler = lr_scheduler.WarmupCosineSchedule(optimizer=optimizer,
                                                              warmup_steps=steps_per_epoch * self.args.warmup_epoch,
                                                              t_total=self.args.epoch * steps_per_epoch,
                                                              cycles=self.args.epoch / self.args.cycles,
                                                              last_epoch=-1)
            elif scheduler_name == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.cycles, eta_min=self.args.lr / 100)   # min: lr / 100, max: lr
            elif scheduler_name == 'ConstantLRSchedule':
                scheduler = lr_scheduler.ConstantLRSchedule(optimizer, last_epoch=-1)
            elif scheduler_name == 'WarmupConstantSchedule':
                scheduler = lr_scheduler.WarmupConstantSchedule(optimizer, warmup_steps=steps_per_epoch * self.args.warmup_epoch)
            else:
                raise Exception('No scheduler named', scheduler_name)
        else:
            pass

        return scheduler
