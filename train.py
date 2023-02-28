import os
import yaml
import argparse
import datetime
from tensorboardX import SummaryWriter

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from mvmm import build_model
from helpers.dataloader_helper import build_train_loader
from helpers.dataloader_helper import build_test_loader
from helpers.optimizer_helper import build_optimizer
from helpers.logger_helper import create_logger
from helpers.logger_helper import log_cfg
from helpers.random_seed_helper import set_random_seed
from helpers.train_helper import Trainer
from helpers.test_helper import Tester


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/configs/ResNet_VFE.yaml',
                        help='path to the config file')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='batch size for training and evaluating')
    parser.add_argument('--epochs', type=int, default=None,
                        help='number of epochs for training')
    parser.add_argument('--result_dir', type=str, default='outputs/data',
                        help='path to save detection results')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='path to the checkpoint for resuming training')
    args = parser.parse_args()
    return args


def main():
    args = parse_config()
    assert os.path.exists(args.cfg_file)
    cfg = yaml.load(open(args.cfg_file, 'r'), Loader=yaml.Loader)

    if args.batch_size is not None:
        cfg['dataset']['batch_size'] = args.batch_size

    if args.epochs is not None:
        cfg['trainer']['epochs'] = args.epochs
        cfg['trainer']['save_frequency'] = min(args.epochs, cfg['trainer']['save_frequency'])
        cfg['tester']['checkpoint'] = 'checkpoints/checkpoint_epoch_%d.pth' % args.epochs

    if args.resume_checkpoint is not None:
        cfg['trainer']['resume_checkpoint'] = args.resume_checkpoint

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)
    log_cfg(args, cfg, logger)

    tb_logger = SummaryWriter(log_dir=log_dir)

    logger.info('###################  Training  ###################')
    set_random_seed(cfg['random_seed'])

    train_loader = build_train_loader(cfg['dataset'], cfg['trainer']['split'])
    test_loader = build_test_loader(cfg['dataset'], cfg['tester']['split'])

    model = build_model(cfg['model'], dataset=train_loader.dataset)

    total_iters_each_epoch = len(train_loader)
    total_epochs = cfg['trainer']['epochs']
    optimizer, lr_scheduler = build_optimizer(cfg['optimizer'], model, total_iters_each_epoch, total_epochs)

    trainer = Trainer(
        cfg=cfg['trainer'],
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        data_loader=train_loader,
        logger=logger,
        tb_logger=tb_logger,
    )
    trainer.train()

    logger.info('###################  Evaluation  ###################')
    tester = Tester(
        cfg=cfg['tester'],
        model=model,
        data_loader=test_loader,
        result_dir=args.result_dir,
        logger=logger
    )
    tester.test()


if __name__ == '__main__':
    main()
