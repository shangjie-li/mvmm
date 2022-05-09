import os
import sys
import torch


def strip_optimizer(f='output/ckpt/checkpoint_epoch_80.pth', s=''):
    # Strip optimizer from 'f' to finalize training, optionally save as 's'
    x = torch.load(f, map_location=torch.device('cpu'))
    for k in 'epoch', 'it', 'optimizer_state', 'version':  # keys
        if k in x:
            del x[k]
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print(f"Optimizer stripped from {f},{(' saved as %s,' % s) if s else ''} {mb:.1f}MB")


if __name__ == '__main__':
    # Usage: python strip_optimizer.py path_to_your_ckpt
    if len(sys.argv) == 2:
        weights = sys.argv[1]
        print('Stripping optimizer in weights: %s...' % weights)
    else:
        print('Error: Only the parameter path_to_your_ckpt is needed.')
        exit()
    strip_optimizer(weights)

