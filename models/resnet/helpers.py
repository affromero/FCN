import torch


def prepare_optim(opts, model):
    optim = torch.optim.SGD(model.parameters(),
                            lr=opts.cfg['lr'],
                            momentum=opts.cfg['momentum'],
                            weight_decay=opts.cfg['weight_decay'])
    if opts.resume:
        checkpoint = torch.load(opts.resume)
        optim.load_state_dict(checkpoint['optim_state_dict'])
    return optim
