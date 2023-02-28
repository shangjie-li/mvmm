import os
import tqdm
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from helpers.checkpoint_helper import save_checkpoint
from helpers.checkpoint_helper import load_checkpoint


class Trainer():
    def __init__(self, cfg, model, optimizer, lr_scheduler, data_loader, logger, tb_logger):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.data_loader = data_loader
        self.logger = logger
        self.tb_logger = tb_logger
        self.epoch = 0
        self.iter = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        if cfg.get('resume_checkpoint') is not None:
            assert os.path.exists(cfg['resume_checkpoint'])
            self.epoch = load_checkpoint(
                file_name=cfg['resume_checkpoint'],
                model=self.model,
                optimizer=self.optimizer,
                map_location=self.device,
                logger=self.logger,
            )
            assert self.epoch is not None
            self.iter = self.epoch * len(self.data_loader)

    def train(self):
        start_epoch = self.epoch
        progress_bar = tqdm.tqdm(
            range(start_epoch, self.cfg['epochs']), dynamic_ncols=True, leave=True,
            desc='epochs'
        )

        for epoch in range(start_epoch, self.cfg['epochs']):
            np.random.seed(np.random.get_state()[1][0] + epoch)
            self.train_one_epoch()
            self.epoch += 1

            if self.epoch % self.cfg['save_frequency'] == 0:
                ckpt_dir = 'checkpoints'
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_name = os.path.join(ckpt_dir, 'checkpoint_epoch_%d.pth' % self.epoch)
                save_checkpoint(ckpt_name, self.model, self.optimizer, self.epoch)

            progress_bar.update()
        progress_bar.close()

    def train_one_epoch(self):
        self.model.train()
        progress_bar = tqdm.tqdm(
            total=len(self.data_loader), dynamic_ncols=True, leave=(self.epoch + 1 == self.cfg['epochs']),
            desc='iters'
        )

        for idx, batch_dict in enumerate(self.data_loader):
            batch_dict = self.data_loader.dataset.load_data_to_gpu(batch_dict, self.device)

            self.lr_scheduler.step(self.iter)

            try:
                cur_lr = float(self.optimizer.lr)
            except:
                cur_lr = self.optimizer.param_groups[0]['lr']
            self.optimizer.zero_grad()

            total_loss, stats_dict = self.model(batch_dict)

            total_loss.backward()
            clip_grad_norm_(self.model.parameters(), 10)
            self.optimizer.step()

            self.tb_logger.add_scalar('learning_rate/learning_rate', cur_lr, self.iter)
            self.tb_logger.add_scalar('loss/loss', total_loss.item(), self.iter)
            for key, val in stats_dict.items():
                self.tb_logger.add_scalar('sub_loss/' + key, val, self.iter)
            self.iter += 1

            progress_bar.update()
        progress_bar.close()
