# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Houwen Peng and  Zhipeng Zhang
# Email: houwen.peng@microsoft.com
# Details: SiamFC training script
# ------------------------------------------------------------------------------

import _init_paths
import argparse
import pprint

from torch.utils.data import DataLoader

from core.config import config, update_config
from dataset.siamfc import SiamFCDataset
from utils.utils import create_logger

eps = 1e-5
def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Train SiamFC')
    # general
    parser.add_argument('--cfg', required=True, type=str, default='/home/syh/siamdw/experiments/train/SiamFC.yaml', help='yaml configure file name')

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    parser.add_argument('--gpus', type=str, help='gpus')
    parser.add_argument('--workers', type=int, help='num of dataloader workers')

    args = parser.parse_args()

    return args


def reset_config(config, args):
    """
    set gpus and workers
    """
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers

def main():
    # [*] args, loggers and tensorboard
    args = parse_args()
    reset_config(config, args)

    logger, _, tb_log_dir = create_logger(config, 'SIAMFC', 'train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # parallel
    gpus = [int(i) for i in config.GPUS.split(',')]
    gpu_num = len(gpus)
    logger.info('GPU NUM: {:2d}'.format(len(gpus)))
    logger.info('model prepare done')

    # build dataloader, benefit to tracking
    train_set = SiamFCDataset(config)
    train_loader = DataLoader(train_set, batch_size=config.SIAMFC.TRAIN.BATCH * gpu_num, num_workers=config.WORKERS,pin_memory=True, sampler=None)

    nCount=0
    for iter, input in enumerate(train_loader):
        # measure data loading time

        # input and output/loss
        label_cls = input[2]
        template = input[0]
        search = input[1]
        nCount=nCount+1

        if nCount==5:
            break

        print('=====')



if __name__ == '__main__':
    main()




