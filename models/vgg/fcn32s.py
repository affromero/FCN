import torch.nn as nn
import os
import fcn


class FCN(nn.Module):

    pretrained_model = os.path.expanduser(
        os.getcwd() + '/data/pretrained_models/fcn32s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vM2oya3k0Zlgtekk',
            path=cls.pretrained_model,
            md5='8acf386d722dc3484625964cbe2aba49',
        )

    def __init__(self, n_class=21):

        super(FCN, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class,
                                          n_class,
                                          64,
                                          stride=32,
                                          bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        from models.vgg.helpers import get_upsampling_weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(m.in_channels,
                                                       m.out_channels,
                                                       m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x, debug=False):
        h = x
        if debug:
            print(h.data.shape)
        h = self.relu1_1(self.conv1_1(h))
        if debug:
            print(h.data.shape)
        h = self.relu1_2(self.conv1_2(h))
        if debug:
            print(h.data.shape)
        h = self.pool1(h)
        if debug:
            print(h.data.shape)

        h = self.relu2_1(self.conv2_1(h))
        if debug:
            print(h.data.shape)
        h = self.relu2_2(self.conv2_2(h))
        if debug:
            print(h.data.shape)
        h = self.pool2(h)
        if debug:
            print(h.data.shape)

        h = self.relu3_1(self.conv3_1(h))
        if debug:
            print(h.data.shape)
        h = self.relu3_2(self.conv3_2(h))
        if debug:
            print(h.data.shape)
        h = self.relu3_3(self.conv3_3(h))
        if debug:
            print(h.data.shape)
        h = self.pool3(h)
        if debug:
            print(h.data.shape)

        h = self.relu4_1(self.conv4_1(h))
        if debug:
            print(h.data.shape)
        h = self.relu4_2(self.conv4_2(h))
        if debug:
            print(h.data.shape)
        h = self.relu4_3(self.conv4_3(h))
        if debug:
            print(h.data.shape)
        h = self.pool4(h)
        if debug:
            print(h.data.shape)

        h = self.relu5_1(self.conv5_1(h))
        if debug:
            print(h.data.shape)
        h = self.relu5_2(self.conv5_2(h))
        if debug:
            print(h.data.shape)
        h = self.relu5_3(self.conv5_3(h))
        if debug:
            print(h.data.shape)
        h = self.pool5(h)
        if debug:
            print(h.data.shape)

        h = self.relu6(self.fc6(h))
        if debug:
            print(h.data.shape)
        h = self.drop6(h)
        if debug:
            print(h.data.shape)

        h = self.relu7(self.fc7(h))
        if debug:
            print(h.data.shape)
        h = self.drop7(h)
        if debug:
            print(h.data.shape)

        h = self.score_fr(h)
        if debug:
            print(h.data.shape)

        h = self.upscore(h)
        if debug:
            print(h.data.shape)
        h = h[:, :, 19:19 + x.size()[2], 19:19 + x.size()[3]].contiguous()
        if debug:
            print(h.data.shape)

        return h

    def copy_params_from_vgg16(self, vgg16):
        features = [
            self.conv1_1,
            self.relu1_1,
            self.conv1_2,
            self.relu1_2,
            self.pool1,
            self.conv2_1,
            self.relu2_1,
            self.conv2_2,
            self.relu2_2,
            self.pool2,
            self.conv3_1,
            self.relu3_1,
            self.conv3_2,
            self.relu3_2,
            self.conv3_3,
            self.relu3_3,
            self.pool3,
            self.conv4_1,
            self.relu4_1,
            self.conv4_2,
            self.relu4_2,
            self.conv4_3,
            self.relu4_3,
            self.pool4,
            self.conv5_1,
            self.relu5_1,
            self.conv5_2,
            self.relu5_2,
            self.conv5_3,
            self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())

    def resume(self, file, test=False):
        import torch
        if test and not file:
            file = 'data/pretrained_models/fcn32s_from_caffe.pth'
            if not os.path.isfile(file):
                self.download()
        if file:
            print('Loading checkpoint from: ' + file)
            checkpoint = torch.load(file)
            try:
                self.load_state_dict(checkpoint)
            except RuntimeError:
                checkpoint = checkpoint['model_state_dict']
                self.load_state_dict(checkpoint)
            self.load_state_dict(checkpoint)
        else:
            from models.vgg.helpers import VGG16
            vgg16 = VGG16(pretrained=True)  # It takes a while
            self.copy_params_from_vgg16(vgg16)
