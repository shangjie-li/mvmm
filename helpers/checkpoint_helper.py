import os
import torch


def save_checkpoint(file_name, model, optimizer=None, epoch=None):
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict() if optimizer is not None else None

    state_dict = {
        'model_state': model_state,
        'optimizer_state': optimizer_state,
        'epoch': epoch,
    }

    torch.save(state_dict, file_name)


def load_checkpoint(file_name, model, optimizer, map_location, logger=None):
    if not os.path.isfile(file_name):
        raise FileNotFoundError

    if logger is not None:
        logger.info('==> Loading from the checkpoint "{}"...'.format(file_name))

    checkpoint = torch.load(file_name, map_location)
    if model is not None and checkpoint['model_state'] is not None:
        model.load_state_dict(checkpoint['model_state'])
    if optimizer is not None and checkpoint['optimizer_state'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    if logger is not None:
        logger.info('==> Done.')

    return checkpoint.get('epoch')
