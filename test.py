import os
import yaml
import argparse
import datetime

try:
    import cv2
except ImportError:
    import sys
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
    import cv2

from mvmm import build_model
from helpers.dataloader_helper import build_test_loader
from helpers.logger_helper import create_logger
from helpers.logger_helper import log_cfg
from helpers.random_seed_helper import set_random_seed
from helpers.test_helper import Tester


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='data/configs/ResNet_VFE.yaml',
                        help='path to the config file')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='batch size for evaluating')
    parser.add_argument('--result_dir', type=str, default='outputs/data',
                        help='path to save detection results')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to the checkpoint for evaluating')
    args = parser.parse_args()
    return args


def main():
    args = parse_config()
    assert os.path.exists(args.cfg_file)
    cfg = yaml.load(open(args.cfg_file, 'r'), Loader=yaml.Loader)

    if args.batch_size is not None:
        cfg['dataset']['batch_size'] = args.batch_size

    if args.checkpoint is not None:
        cfg['tester']['checkpoint'] = args.checkpoint

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)
    log_cfg(args, cfg, logger)

    logger.info('###################  Evaluation Only  ###################')
    set_random_seed(cfg['random_seed'])

    test_loader = build_test_loader(cfg['dataset'], cfg['tester']['split'])

    model = build_model(cfg['model'], dataset=test_loader.dataset)

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
