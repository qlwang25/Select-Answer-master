# -*- coding: utf-8 -*-

#!/usr/bin/env python
# @Time    : 18-1-24 下午2:45
# @Author  : wang shen
# @web    : 
# @File    : test.py

import torch
from torch.autograd import Variable
import torch.nn as nn


def test():
    input = torch.rand(20, 100, 100)
    input = Variable(input)
    print('input size: ', input.size())
    # size : (N, C, W , H)
    input = torch.unsqueeze(input, 1)
    print('input size: ', input.size(), '\n')

    # W_new = ( W + padding[0] * 2 - (kernel_size[0] - 1) * dilation[0] ) / stride[0] + 1
    # H_new = ( H + padding[1] * 2 - (kernel_size[1] - 1) * dilation[1] ) / stride[1] + 1
    layer1 = nn.Conv2d(1, 100, kernel_size=(5, 100), stride=1, padding=(0, 50))
    layer1_out = layer1(input)
    print('layer1 out size :', layer1_out.size())

    layer1_pool = nn.MaxPool2d((4, 1), stride=2, padding=0)
    layer1_pool_out = layer1_pool(layer1_out)
    print('layer1 pool out size :', layer1_pool_out.size())
    pass


if __name__ == '__main__':
    test()