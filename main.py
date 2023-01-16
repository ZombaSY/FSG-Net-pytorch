import torch
import os
import argparse
import wandb
import yaml
import numpy as np
import random
import ast

from train import Trainer_seg
from inference import Inferencer
from torch.cuda import is_available
from datetime import datetime


# fix seed for reproducibility
seed = 3407
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)  # raise error if CUDA >= 10.2
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


def conf_to_args(args, **kwargs):
    var = vars(args)

    for key, value in kwargs.items():
        var[key] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default=None, type=str)
    arg, unknown_arg = parser.parse_known_args()

    if arg.config_path is not None:
        with open(arg.config_path, 'rb') as f:
            conf = yaml.load(f.read(), Loader=yaml.Loader)  # load the config file
            conf['config_path'] = arg.config_path
    else:
        # make unrecognized args to dict
        conf = {'config_path': 'configs/sweep_config.yaml'}
        for item in unknown_arg:
            item = item.strip('--')
            key, value = item.split('=')
            if key != 'CUDA_VISIBLE_DEVICES':
                try:
                    if value == 'true' or value == 'false':
                        value = value.title()
                    value = ast.literal_eval(value)
                except ValueError:
                    if value.isalpha(): pass
                except SyntaxError as e:
                    if '/' in value: pass
                    else: raise e
            conf[key] = value

    args = argparse.Namespace()
    conf_to_args(args, **conf)  # pass in keyword args

    now_time = datetime.now().strftime("%Y-%m-%d %H%M%S")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES

    if args.wandb:
        wandb.login(key='your_WandB_key')
        wandb.init(project='{}'.format(args.project_name), config=args, name=now_time,
                   settings=wandb.Settings(start_method="fork"))

    print('Use CUDA :', args.cuda and is_available())
    if args.mode in 'train':

        # save hyper-parameters
        with open(args.config_path, 'r') as f_r:
            file_path = args.saved_model_directory + '/' + now_time
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            with open(os.path.join(file_path, args.config_path.split('/')[-1]), 'w') as f_w:
                f_w.write(f_r.read())

        if args.mode == 'train':
            if args.task == 'segmentation':
                trainer = Trainer_seg(args, now_time)
        else:
            raise Exception('Invalid mode')

        trainer.start_train()

    elif args.mode in 'inference':
        inferencer = Inferencer(args)

        if args.inference_mode == 'segmentation':
            inferencer.start_inference_segmentation()
        else:
            raise ValueError('Please select correct inference_mode !!!')
    else:
        print('No mode supported.')


if __name__ == "__main__":
    main()
