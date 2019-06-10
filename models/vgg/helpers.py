import torch
import torchvision
import fcn  # pip install fcn
import numpy as np
import os
import os.path as osp

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


def VGG16(pretrained=False, folder='data/pretrained_models'):
    model = torchvision.models.vgg16(pretrained=False)
    if not pretrained:
        return model
    model_file = _get_vgg16_pretrained_model(folder)

    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    return model


def _get_vgg16_pretrained_model(folder):
    path_model = osp.join(os.getcwd(), folder, 'vgg16_from_caffe.pth')
    return fcn.data.cached_download(
        url='http://drive.google.com/uc?id=0B9P1L--7Wd2vLTJZMXpIRkVVRFk',
        path=path_model,
        md5='aa75b158f4181e7f6230029eb96c1b13',
    )


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
    )
    for idx, m in enumerate(model.modules()):
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped) or idx == 0:
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))


def prepare_optim(opts, model):
    optim = torch.optim.SGD([
        {
            'params': get_parameters(model, bias=False)
        },
        {
            'params': get_parameters(model, bias=True),
            'lr': opts.cfg['lr'] * 2,
            'weight_decay': 0
        },
    ],
                            lr=opts.cfg['lr'],
                            momentum=opts.cfg['momentum'],
                            weight_decay=opts.cfg['weight_decay'])
    if opts.resume:
        checkpoint = torch.load(opts.resume)
        optim.load_state_dict(checkpoint['optim_state_dict'])
    return optim
