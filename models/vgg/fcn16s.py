import torch.nn as nn
import fcn
import os


class FCN(nn.Module):

    pretrained_model = os.path.expanduser(
        os.getcwd() + '/data/pretrained_models/fcn16s_from_caffe.pth')

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vVGE3TkRMbWlNRms',
            path=cls.pretrained_model,
            md5='991ea45d30d632a01e5ec48002cac617',
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
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(n_class,
                                           n_class,
                                           4,
                                           stride=2,
                                           bias=False)
        self.upscore16 = nn.ConvTranspose2d(n_class,
                                            n_class,
                                            32,
                                            stride=16,
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
            print('pool4: {}'.format(h.data.shape))
        pool4 = h  # 1/16 #<------------------------------------

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
        h = self.upscore2(h)
        if debug:
            print('upscore2: {}'.format(h.data.shape))
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        if debug:
            print('score_pool4: {}'.format(h.data.shape))
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        if debug:
            print('score_pool4c: {}'.format(h.data.shape))
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c
        if debug:
            print('upscore2+score_pool4c: {}'.format(h.data.shape))

        h = self.upscore16(h)
        if debug:
            print('upscore16: {}'.format(h.data.shape))
        h = h[:, :, 27:27 + x.size()[2], 27:27 + x.size()[3]].contiguous()
        if debug:
            print('upscore16 rearranged: {}'.format(h.data.shape))

        return h

    def copy_params_from_fcn32s(self, fcn32s):
        for name, l1 in fcn32s.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            assert l1.weight.size() == l2.weight.size()
            assert l1.bias.size() == l2.bias.size()
            l2.weight.data.copy_(l1.weight.data)
            l2.bias.data.copy_(l1.bias.data)

    def resume(self, file, test=False):
        import torch
        if test and not file:
            file = 'data/pretrained_models/fcn16s_from_caffe.pth'
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
            from models.vgg.fcn32s import FCN
            fcn32s = FCN()
            fcn32s_weights = FCN.download()  # Original FCN32 pretrained model
            fcn32s.load_state_dict(torch.load(fcn32s_weights))
            self.copy_params_from_fcn32s(fcn32s)
