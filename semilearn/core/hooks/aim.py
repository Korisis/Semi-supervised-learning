# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import aim
from .hook import Hook
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def get_confusion_matrix_aim_figure(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    _ = disp.plot(values_format='.2f')
    fig = disp.figure_
    plt.close(disp.figure_)
    return aim.Image(fig)


class AimHook(Hook):
    """
    Aim Hook
    """

    def __init__(self):
        super().__init__()
        self.log_key_list = ['train/sup_loss', 'train/unsup_loss', 'train/total_loss', 'train/util_ratio',
                             'train/ulb_targets_true', 'train/ulb_targets_false', 'train/ulb_targets_true_ratio',
                             'train/run_time', 'train/prefetch_time', 'lr',
                             'eval/loss', 'eval/top-1-acc', 'eval/precision', 'eval/recall', 'eval/F1']

    def before_run(self, algorithm):
        # initialize aim run
        name = algorithm.save_name
        project = algorithm.save_dir.split('/')[-1]
        self.run = aim.Run(experiment=name, repo='/mnt/c/Users/Korisis/switchdrive/wods/Semi-supervised-learning/saved_models/classic_cv')

        # set configuration
        self.run['hparams'] = algorithm.args.__dict__


        # set tag
        benchmark = f'benchmark: {project}'
        dataset = f'dataset: {algorithm.args.dataset}'
        data_setting = f'setting: {algorithm.args.dataset}_lb{algorithm.args.num_labels}_{algorithm.args.lb_imb_ratio}_ulb{algorithm.args.ulb_num_labels}_{algorithm.args.ulb_imb_ratio}'
        alg = f'alg: {algorithm.args.algorithm}'
        imb_alg = f'imb_alg: {algorithm.args.imb_algorithm}'
        self.run.add_tag(benchmark)
        self.run.add_tag(dataset)
        self.run.add_tag(data_setting)
        self.run.add_tag(alg)
        self.run.add_tag(imb_alg)

    def after_train_step(self, algorithm):
        if self.every_n_iters(algorithm, algorithm.num_log_iter):
            for key, item in algorithm.log_dict.items():
                if key in self.log_key_list:
                    self.run.track(item, name=key, step=algorithm.it)
        
        if self.every_n_iters(algorithm, algorithm.num_eval_iter):
            self.run.track(algorithm.best_eval_acc, name='eval/best-acc', step=algorithm.it)
            self.run.track(get_confusion_matrix_aim_figure(algorithm.log_dict.get('eval/confusion_matrix')), name='confusion matrix', step=algorithm.it)
