import logging

__all__ = ["scale_lr", "set_lr"]


def scale_lr(optimizer, k):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] *= k
    logging.info('lr multiply {}'.format(k))


def set_lr(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
    logging.info('lr set to {}'.format(lr))
