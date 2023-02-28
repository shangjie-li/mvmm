try:
    from collections.abc import Iterable
except:
    from collections import Iterable

from functools import partial
import numpy as np
import torch
from torch import nn
from torch._utils import _unflatten_dense_tensors
from torch.nn.utils import parameters_to_vector


bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


def split_bn_bias(layer_groups):
    "Split the layers in `layer_groups` into batchnorm (`bn_types`) and non-batchnorm groups."
    split_groups = []
    for l in layer_groups:
        l1, l2 = [], []
        for c in l.children():
            if isinstance(c, bn_types):
                l2.append(c)
            else:
                l1.append(c)
        split_groups += [nn.Sequential(*l1), nn.Sequential(*l2)]
    return split_groups


def get_master(layer_groups, flat_master: bool = False):
    "Return two lists, one for the model parameters in FP16 and one for the master parameters in FP32."
    split_groups = split_bn_bias(layer_groups)
    model_params = [[param for param in lg.parameters() if param.requires_grad] for lg in split_groups]
    if flat_master:
        master_params = []
        for lg in model_params:
            if len(lg) != 0:
                mp = parameters_to_vector([param.data.float() for param in lg])
                mp = torch.nn.Parameter(mp, requires_grad=True)
                if mp.grad is None: mp.grad = mp.new(*mp.size())
                master_params.append([mp])
            else:
                master_params.append([])
        return model_params, master_params
    else:
        master_params = [[param.clone().float().detach() for param in lg] for lg in model_params]
        for mp in master_params:
            for param in mp: param.requires_grad = True
        return model_params, master_params


def model_g2master_g(model_params, master_params, flat_master: bool = False) -> None:
    "Copy the `model_params` gradients to `master_params` for the optimizer step."
    if flat_master:
        for model_group, master_group in zip(model_params, master_params):
            if len(master_group) != 0:
                master_group[0].grad.data.copy_(parameters_to_vector([p.grad.data.float() for p in model_group]))
    else:
        for model_group, master_group in zip(model_params, master_params):
            for model, master in zip(model_group, master_group):
                if model.grad is not None:
                    if master.grad is None: master.grad = master.data.new(*master.data.size())
                    master.grad.data.copy_(model.grad.data)
                else:
                    master.grad = None


def master2model(model_params, master_params, flat_master: bool = False) -> None:
    "Copy `master_params` to `model_params`."
    if flat_master:
        for model_group, master_group in zip(model_params, master_params):
            if len(model_group) != 0:
                for model, master in zip(model_group, _unflatten_dense_tensors(master_group[0].data, model_group)):
                    model.data.copy_(master)
    else:
        for model_group, master_group in zip(model_params, master_params):
            for model, master in zip(model_group, master_group): model.data.copy_(master.data)


def listify(p=None, q=None):
    "Make `p` listy and the same length as `q`."
    if p is None:
        p = []
    elif isinstance(p, str):
        p = [p]
    elif not isinstance(p, Iterable):
        p = [p]
    n = q if type(q) == int else len(p) if q is None else len(q)
    if len(p) == 1: p = p * n
    assert len(p) == n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


def trainable_params(m: nn.Module):
    "Return list of trainable params in `m`."
    res = filter(lambda p: p.requires_grad, m.parameters())
    return res


def is_tuple(x) -> bool: return isinstance(x, tuple)


class OptimWrapper():
    "Basic wrapper around `opt` to simplify hyper-parameters changes."

    def __init__(self, opt, wd, true_wd: bool = False, bn_wd: bool = True):
        self.opt, self.true_wd, self.bn_wd = opt, true_wd, bn_wd
        self.opt_keys = list(self.opt.param_groups[0].keys())
        self.opt_keys.remove('params')
        self.read_defaults()
        self.wd = wd

    @classmethod
    def create(cls, opt_func, lr,
               layer_groups, **kwargs):
        "Create an `optim.Optimizer` from `opt_func` with `lr`. Set lr on `layer_groups`."
        split_groups = split_bn_bias(layer_groups)
        opt = opt_func([{'params': trainable_params(l), 'lr': 0} for l in split_groups])
        opt = cls(opt, **kwargs)
        opt.lr, opt.opt_func = listify(lr, layer_groups), opt_func
        return opt

    def new(self, layer_groups):
        "Create a new `OptimWrapper` from `self` with another `layer_groups` but the same hyper-parameters."
        opt_func = getattr(self, 'opt_func', self.opt.__class__)
        split_groups = split_bn_bias(layer_groups)
        opt = opt_func([{'params': trainable_params(l), 'lr': 0} for l in split_groups])
        return self.create(opt_func, self.lr, layer_groups, wd=self.wd, true_wd=self.true_wd, bn_wd=self.bn_wd)

    def __repr__(self) -> str:
        return f'OptimWrapper over {repr(self.opt)}.\nTrue weight decay: {self.true_wd}'

    def step(self) -> None:
        "Set weight decay and step optimizer."
        # weight decay outside of optimizer step (AdamW)
        if self.true_wd:
            for lr, wd, pg1, pg2 in zip(self._lr, self._wd, self.opt.param_groups[::2], self.opt.param_groups[1::2]):
                for p in pg1['params']:
                    # When some parameters are fixed:  Shaoshuai Shi
                    if p.requires_grad is False:
                        continue
                    p.data.mul_(1 - wd * lr)
                if self.bn_wd:
                    for p in pg2['params']:
                        # When some parameters are fixed:  Shaoshuai Shi
                        if p.requires_grad is False:
                            continue
                        p.data.mul_(1 - wd * lr)
            self.set_val('weight_decay', listify(0, self._wd))
        self.opt.step()

    def zero_grad(self) -> None:
        "Clear optimizer gradients."
        self.opt.zero_grad()

    def __getattr__(self, k: str):
        return getattr(self.opt, k, None)

    def clear(self):
        "Reset the state of the inner optimizer."
        sd = self.state_dict()
        sd['state'] = {}
        self.load_state_dict(sd)

    @property
    def lr(self) -> float:
        return self._lr[-1]

    @lr.setter
    def lr(self, val: float) -> None:
        self._lr = self.set_val('lr', listify(val, self._lr))

    @property
    def mom(self) -> float:
        return self._mom[-1]

    @mom.setter
    def mom(self, val: float) -> None:
        if 'momentum' in self.opt_keys:
            self.set_val('momentum', listify(val, self._mom))
        elif 'betas' in self.opt_keys:
            self.set_val('betas', (listify(val, self._mom), self._beta))
        self._mom = listify(val, self._mom)

    @property
    def beta(self) -> float:
        return None if self._beta is None else self._beta[-1]

    @beta.setter
    def beta(self, val: float) -> None:
        "Set beta (or alpha as makes sense for given optimizer)."
        if val is None: return
        if 'betas' in self.opt_keys:
            self.set_val('betas', (self._mom, listify(val, self._beta)))
        elif 'alpha' in self.opt_keys:
            self.set_val('alpha', listify(val, self._beta))
        self._beta = listify(val, self._beta)

    @property
    def wd(self) -> float:
        return self._wd[-1]

    @wd.setter
    def wd(self, val: float) -> None:
        "Set weight decay."
        if not self.true_wd: self.set_val('weight_decay', listify(val, self._wd), bn_groups=self.bn_wd)
        self._wd = listify(val, self._wd)

    def read_defaults(self) -> None:
        "Read the values inside the optimizer for the hyper-parameters."
        self._beta = None
        if 'lr' in self.opt_keys: self._lr = self.read_val('lr')
        if 'momentum' in self.opt_keys: self._mom = self.read_val('momentum')
        if 'alpha' in self.opt_keys: self._beta = self.read_val('alpha')
        if 'betas' in self.opt_keys: self._mom, self._beta = self.read_val('betas')
        if 'weight_decay' in self.opt_keys: self._wd = self.read_val('weight_decay')

    def set_val(self, key: str, val, bn_groups: bool = True):
        "Set `val` inside the optimizer dictionary at `key`."
        if is_tuple(val): val = [(v1, v2) for v1, v2 in zip(*val)]
        for v, pg1, pg2 in zip(val, self.opt.param_groups[::2], self.opt.param_groups[1::2]):
            pg1[key] = v
            if bn_groups: pg2[key] = v
        return val

    def read_val(self, key: str):
        "Read a hyperparameter `key` in the optimizer dictionary."
        val = [pg[key] for pg in self.opt.param_groups[::2]]
        if is_tuple(val[0]): val = [o[0] for o in val], [o[1] for o in val]
        return val


class LRSchedulerStep(object):
    def __init__(self, fai_optimizer: OptimWrapper, total_step, lr_phases,
                 mom_phases):
        self.optimizer = fai_optimizer
        self.total_step = total_step
        self.lr_phases = []

        for i, (start, lambda_func) in enumerate(lr_phases):
            if len(self.lr_phases) != 0:
                assert self.lr_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(lr_phases) - 1:
                self.lr_phases.append((int(start * total_step), int(lr_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.lr_phases.append((int(start * total_step), total_step, lambda_func))
        assert self.lr_phases[0][0] == 0
        self.mom_phases = []
        for i, (start, lambda_func) in enumerate(mom_phases):
            if len(self.mom_phases) != 0:
                assert self.mom_phases[-1][0] < start
            if isinstance(lambda_func, str):
                lambda_func = eval(lambda_func)
            if i < len(mom_phases) - 1:
                self.mom_phases.append((int(start * total_step), int(mom_phases[i + 1][0] * total_step), lambda_func))
            else:
                self.mom_phases.append((int(start * total_step), total_step, lambda_func))
        assert self.mom_phases[0][0] == 0

    def step(self, step):
        for start, end, func in self.lr_phases:
            if step >= start:
                self.optimizer.lr = func((step - start) / (end - start))
        for start, end, func in self.mom_phases:
            if step >= start:
                self.optimizer.mom = func((step - start) / (end - start))


def annealing_cos(start, end, pct):
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start - end) / 2 * cos_out


class OneCycle(LRSchedulerStep):
    def __init__(self, fai_optimizer, total_step, lr_max, moms, div_factor,
                 pct_start):
        self.lr_max = lr_max
        self.moms = moms
        self.div_factor = div_factor
        self.pct_start = pct_start
        a1 = int(total_step * self.pct_start)
        a2 = total_step - a1
        low_lr = self.lr_max / self.div_factor
        lr_phases = ((0, partial(annealing_cos, low_lr, self.lr_max)),
                     (self.pct_start,
                      partial(annealing_cos, self.lr_max, low_lr / 1e4)))
        mom_phases = ((0, partial(annealing_cos, *self.moms)),
                      (self.pct_start, partial(annealing_cos,
                                               *self.moms[::-1])))
        fai_optimizer.lr, fai_optimizer.mom = low_lr, self.moms[0]
        super().__init__(fai_optimizer, total_step, lr_phases, mom_phases)
