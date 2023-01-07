import os
import time
import logging

import cv2
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm.autonotebook import tqdm

import lib.dataset
import lib.loss
import lib.model
import lib.utils

import random

# PyTorch基础设置
torch.manual_seed(random.randint(1,99999))  # 设置随机种子
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# 定义训练流程
class PipeLine(object):
    def __init__(self, config):
        super().__init__()
        self.cfg = config

        self.is_t1 = "-t1" in self.cfg.data_root


        # 初始化DataLoader
        self.ds_type = lib.dataset.__dict__[self.cfg.dataset_type]

        self.train_loader = lib.utils.DataLoaderX(
            self.ds_type(self.cfg.data_root, k=self.cfg.fold_id, cfg=self.cfg, is_train=False),
            batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers,
            shuffle=False, pin_memory=True, drop_last=False)

        # 初始化模型
        print(self.cfg.model_type)
        self.net = lib.model.__dict__[self.cfg.model_type]().cuda()
        logging.info('Model Type: {}, Total Params: {:.2f}M'.format(
            self.cfg.model_type, sum(p.numel() for p in self.net.parameters())/1e6)
        )
        # if self.cfg.pretrain_model is not None:
        print(self.cfg.pretrain_model)
        self.net = lib.utils.load_model(self.net, self.cfg.checkpoint)


    def run(self):
        self.results_one_epoch()

    def results_one_epoch(self):
        print('================================>check results in save path')
        progress_bar = tqdm(self.train_loader)
        load_t0 = time.time()
        aa = 0
        for data in progress_bar:
            aa += 1
            # load data
            data = [v.cuda(non_blocking=True) for v in data]

            # data = data.cuda(non_blocking=True)
            outputs = self.net(data[0])
            for i in range(outputs.shape[0]):
                # print(outputs[i])
                img = outputs[i].permute(1, 2, 0)
                img = img.detach().cpu().numpy()
                cv2.imwrite(os.path.join(self.cfg.save_root, str(aa) + '_' + str(i) + 'mae.jpg'), img * 255)





if __name__ == '__main__':
    import argparse

    # 设置日志消息格式
    logging.basicConfig(format="%(asctime)s-%(message)s", level=logging.INFO)

    # 读取运行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str)
    args = parser.parse_args()

    # 加载配置文件
    cfg = lib.utils.load_yaml_config(args.config_path)

    # 实例化训练流程并开始训练
    pipeline = PipeLine(cfg)
    pipeline.run()
