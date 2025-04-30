import torch
import time
import numpy as np
import os

from models import metrics
from models import utils
from models import dataloader as dataloader_hub
from models import model_implements
from train import Trainer_seg
from PIL import Image


class Inferencer:
    def __init__(self, args):
        self.start_time = time.time()
        self.args = args

        use_cuda = self.args.cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if use_cuda else 'cpu')

        self.loader_form = self.__init_data_loader(self.args.val_x_path,
                                                   self.args.val_y_path,
                                                   batch_size=1,
                                                   mode='validation')

        self.model = Trainer_seg.init_model(self.args.model_name, self.device, self.args)
        self.model.load_state_dict(torch.load(args.model_path))
        self.model.eval()

        self.metric = self._init_metric(self.args.task, self.args.n_classes)

        self.image_mean = self.loader_form.image_loader.image_mean
        self.image_std = self.loader_form.image_loader.image_std
        self.fn_list = []

    def start_inference_segmentation(self):
        f1_list = []
        acc_list = []
        auc_list = []
        sen_list = []
        mcc_list = []

        for batch_idx, (img, target) in enumerate(self.loader_form.Loader):
            with torch.no_grad():
                x_in, img_id = img
                target, origin_size = target

                x_in = x_in.to(self.device)
                x_img = x_in
                target = target.long().to(self.device)

                output = self.model(x_in)

                if isinstance(output, tuple) or isinstance(output, list):  # condition for deep supervision
                    output = output[0]

                metric_result = self.post_process(output, target, x_img, img_id)
                f1_list.append(metric_result['f1'])
                acc_list.append(metric_result['acc'])
                auc_list.append(metric_result['auc'])
                sen_list.append(metric_result['sen'])
                mcc_list.append(metric_result['mcc'])

        metrics = self.metric.get_results()
        cIoU = [metrics['Class IoU'][i] for i in range(self.args.n_classes + 1)]
        mIoU = sum(cIoU) / (self.args.n_classes + 1)

        print('mean mIoU', mIoU)
        print('mean F1 score:', sum(f1_list) / len(f1_list))
        print('mean Accuracy', sum(acc_list) / len(acc_list))
        print('mean AUC', sum(auc_list) / len(auc_list))
        print('mean Sensitivity', sum(sen_list) / len(sen_list))
        print('mean MCC', sum(mcc_list) / len(mcc_list))

    def post_process(self, output, target, x_img, img_id):
        # reconstruct original image
        x_img = x_img.squeeze(0).data.cpu().numpy()
        x_img = np.transpose(x_img, (1, 2, 0))
        x_img = x_img * np.array(self.image_std)
        x_img = x_img + np.array(self.image_mean)
        x_img = x_img * 255.0
        x_img = x_img.astype(np.uint8)

        output = utils.remove_center_padding(output)
        target = utils.remove_center_padding(target)

        output_argmax = torch.where(output > 0.5, 1, 0).cpu().detach()
        self.metric.update(target.squeeze(1).cpu().detach().numpy(), output_argmax.numpy())

        path, fn = os.path.split(img_id[0])
        img_id, ext = os.path.splitext(fn)
        dir_path, fn = os.path.split(self.args.model_path)
        fn, ext = os.path.splitext(fn)
        save_dir = dir_path + '/' + fn + '/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        Image.fromarray(x_img).save(save_dir + img_id + '.png', quality=100)
        Image.fromarray((output_argmax.squeeze().numpy() * 255).astype(np.uint8)).save(save_dir + img_id + f'_argmax.png', quality=100)
        # Image.fromarray(output_heatmap.astype(np.uint8)).save(save_dir + img_id + f'_heatmap_overlay.png', quality=100)

        metric_result = metrics.metrics_np(output_argmax[None, :], target.squeeze(0).detach().cpu().numpy(), b_auc=True)
        print(f'{img_id} \t Done !!')

        return metric_result

    def __init_model(self, model_name):
        if model_name == 'UNet':
            model = model_implements.UNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'UNet2P':
            model = model_implements.UNet2P(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'UNet3P_Deep':
            model = model_implements.UNet3P_Deep(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'ResUNet':
            model = model_implements.ResUNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'ResUNet2P':
            model = model_implements.ResUNet2P(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'SAUNet':
            model = model_implements.SAUNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'ATTUNet':
            model = model_implements.ATTUNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'DCSAU_UNet':
            model = model_implements.DCSAU_UNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'AGNet':
            model = model_implements.AGNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'R2UNet':
            model = model_implements.R2UNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'ConvUNeXt':
            model = model_implements.ConvUNeXt(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'FRUNet':
            model = model_implements.FRUNet(n_classes=1, in_channels=self.args.input_channel)
        elif model_name == 'FSGNet':
            model = model_implements.FSGNet(n_classes=1, in_channels=self.args.input_channel)
        else:
            raise Exception('No model named', model_name)

        return torch.nn.DataParallel(model)

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
        elif self.args.dataloader == 'Image2Image_resize':
            loader = dataloader_hub.Image2ImageDataLoader_resize(x_path=x_path,
                                                                 y_path=y_path,
                                                                 batch_size=batch_size,
                                                                 num_workers=self.args.worker,
                                                                 pin_memory=self.args.pin_memory,
                                                                 mode=mode,
                                                                 args=self.args)
        else:
            raise Exception('No dataloader named', self.args.dataloader)

        return loader

    def _init_metric(self, task_name, num_class):
        if task_name == 'segmentation':
            metric = metrics.StreamSegMetrics_segmentation(num_class + 1)
        else:
            raise Exception('No task named', task_name)

        return metric
