from torchvision.models.segmentation import fcn_resnet101
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, n_class=21):

        super(FCN, self).__init__()
        self.fcn = fcn_resnet101(pretrained=False, num_classes=21)
        # Uses bilinear interpolation for upsampling
        # https://github.com/pytorch/vision/blob/master/
        # torchvision/models/segmentation/_utils.py

    def forward(self, x, debug=False):
        return self.fcn(x)['out']

    def resume(self, file, test=False):
        import torch
        if test and not file:
            self.fcn = fcn_resnet101(pretrained=True, num_classes=21)
            return
        if file:
            print('Loading checkpoint from: ' + file)
            checkpoint = torch.load(file)
            checkpoint = checkpoint['model_state_dict']
            self.load_state_dict(checkpoint)
