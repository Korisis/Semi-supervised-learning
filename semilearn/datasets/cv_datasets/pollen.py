# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data


mean, std = {}, {}
mean['pollen'] = [220.83]

std['pollen'] = [35.47]


def get_pollen(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True, logger=None):
    
    data_dir = os.path.join(data_dir, name.lower())
    data_path = os.path.join(data_dir, 'wods_pollen_data_uint8.pkl')

    df = pd.read_pickle(data_path)

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df['species'])
    
    train_data = np.array(df_train['holo_image_0'])
    train_targets = np.array(df_train['species'], dtype=torch.LongTensor)
    
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5, exclude_color_aug=True),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name],)
    ])

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, train_data, train_targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)
    
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1

    if logger is not None:
        logger("lb count: {}".format(lb_count))
        logger("ulb count: {}".format(ulb_count))
    else:
        print("lb count: {}".format(lb_count))
        print("ulb count: {}".format(ulb_count))

    if alg == 'fullysupervised':
        lb_data = train_data
        lb_targets = train_targets

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

    test_data = np.array(df_test['holo_image_0'])
    test_targets = np.array(df_test['species'], dtype=torch.LongTensor)
    eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset
