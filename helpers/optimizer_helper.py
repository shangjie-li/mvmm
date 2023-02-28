from functools import partial

import torch.nn as nn
import torch.optim as optim

from utils.optim_wrapper_utils import OptimWrapper
from utils.optim_wrapper_utils import OneCycle


def build_optimizer(cfg, model, total_iters_each_epoch, total_epochs):
    if cfg['type'] == 'AdamOneCycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, cfg['lr'], get_layer_groups(model), wd=cfg['weight_decay'], true_wd=True, bn_wd=True
        )

        total_steps = total_iters_each_epoch * total_epochs
        moms = [0.95, 0.85]
        div_factor = 10
        pct_start = 0.4
        lr_scheduler = OneCycle(
            optimizer, total_steps, cfg['lr'], moms, div_factor, pct_start
        )

    else:
        raise NotImplementedError

    return optimizer, lr_scheduler
