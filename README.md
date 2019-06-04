# Pytorch - fcn

PyTorch implementation of [Fully Convolutional Networks](https://github.com/shelhamer/fcn.berkeleyvision.org) including VGG and RESNET backbones.


## Requirements

- [pytorch](https://github.com/pytorch/pytorch) >= 1.1.0
- [torchvision](https://github.com/pytorch/vision) >= 0.3.0
- [fcn](https://github.com/wkentaro/fcn) >= 6.1.5
- [Pillow](https://github.com/python-pillow/Pillow)
- [tqdm](https://github.com/tqdm/tqdm)


## Training

`python main.py --gpu_id=0 --backbone=vgg --fcn=32s --root_dataset=./data/Pascal_VOC --mode=train`
- backbone = vgg or resnet.
- fcn = 32s, 16s, 8s for vgg, and 101, 50 for resnet.
- mode = train, val, demo.

### Acknowledgement
This repo is inspired in the [wkentaro](https://github.com/wkentaro/pytorch-fcn/blob/master/README.md)'s repo and some snippets are a copy of it. 
