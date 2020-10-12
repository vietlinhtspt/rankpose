# -*- coding: utf-8
import yaml
import torch
import pickle
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from losses import Criterion
from models import load_model
import albumentations as albu
from dataset import laod_dataset
from collections import OrderedDict
from logger.log import debug_logger
from logger.plot import history_ploter
from torch.utils.data import DataLoader
from utils.metrics import calculate_diff
from utils.optimizer import create_optimizer
from torch.utils.data.sampler import  WeightedRandomSampler
from glob import glob

seed = 2020
np.random.seed(seed)
random.seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('config_path')
args = parser.parse_args()

def main():
    config_path = Path(args.config_path)
    config = yaml.load(open(config_path))

    net_config = config['Net']
    # loss_config = config['Loss']
    # opt_config = config['Optimizer']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_class = net_config['n_class']
    val_dir = '../data'
    val_name = 'biwi_dataset_list.txt'
    val_type = 'BIWI'
    use_bined = False
    num_workers = 4

    pretrained_path = ["/home/linhnv/projects/RankPose/model/headpose_resnet/model_epoch_77_8.775894704750192.pth"]

    # models_path = glob("/home/linhnv/projects/RankPose/model/headpose_resnet/*")
    # models_path = [x for x in models_path if x.startswith("/home/linhnv/projects/RankPose/model/headpose_resnet/model_epoch")]
    # print(models_path)
    for pretrained_path in pretrained_path:
        print(f"[INFO] Pretrained path: {pretrained_path}")

        model = load_model(**net_config)
        # To device
        model = model.to(device)

        modelname = config_path.stem
        output_dir = Path('../model') / modelname
        output_dir.mkdir(exist_ok=True)
        log_dir = Path('../logs') / modelname
        log_dir.mkdir(exist_ok=True)

        # logger = debug_logger(log_dir)
        # logger.debug(config)
        # logger.info(f'Device: {device}')
    

        params = model.parameters()
        

        valid_dataset = laod_dataset(data_type=val_type, split='valid', base_dir=val_dir, filename=val_name, 
                                use_bined=False, n_class=n_class)

        # top_10 = len(train_dataset) // 10
        # top_30 = len(train_dataset) // 3.33
        # train_weights = [ 3 if idx<top_10 else 2 if idx<top_30 else 1 for idx in train_dataset.labels_sort_idx]
        # train_sample = WeightedRandomSampler(train_weights, num_samples=len(train_dataset), replacement=True)

        # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sample, num_workers=num_workers,
        #                           pin_memory=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True)

        if torch.cuda.is_available():
            model = nn.DataParallel(model)

        # logger.info(f'Load pretrained from {pretrained_path}')
        param = torch.load(pretrained_path, map_location='cpu')
        if "state_dict" in param:
            model.load_state_dict(param['state_dict'], strict=False)
        else:
            model.load_state_dict(param)
        del param

        valid_losses = []
        valid_diffs = []
        model.eval()
        with torch.no_grad():
            with tqdm(valid_loader) as _tqdm:
                for batched in _tqdm:
                    if use_bined:
                        images, labels, yaw_labels, pitch_labels, roll_labels = batched
                    
                        images, labels = images.to(device), labels.to(device)
                        # yaw_labels, pitch_labels, roll_labels = yaw_labels.to(device), pitch_labels.to(device), roll_labels.to(device)

                        preds, y_pres, p_pres, r_pres = model(images, use_bined)
                    
                        # loss = loss_fn([preds, y_pres, p_pres, r_pres], [labels, yaw_labels, pitch_labels, roll_labels])

                        diff = calculate_diff(preds, labels)
                    else:
                        images, labels = batched
                    
                        images, labels = images.to(device), labels.to(device)

                        preds = model(images, use_bined)
                    
                        # loss = loss_fn([preds], [labels])

                        diff = calculate_diff(preds, labels)
                    
                    _tqdm.set_postfix(OrderedDict(mae=f'{diff:.2f}'))
                    # _tqdm.set_postfix(OrderedDict(loss=f'{loss.item():.3f}', d_y=f'{np.mean(diff[:,0]):.1f}', d_p=f'{np.mean(diff[:,1]):.1f}', d_r=f'{np.mean(diff[:,2]):.1f}'))
                    valid_losses.append(0)
                    valid_diffs.append(diff)

        valid_loss = np.mean(valid_losses)
        valid_diff = np.mean(valid_diffs)
        print(f'valid diff: {valid_diff}')
        # logger.info(f'valid seg loss: {valid_loss}')
        # logger.info(f'valid diff: {valid_diff}')

if __name__=='__main__':
    main()