# *----------------------------------------------------------------------------*
# * Copyright (C) 2023 Politecnico di Torino, Italy                            *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Beatrice Alessandra Motetti <beatrice.motetti@polito.it>          *
# *----------------------------------------------------------------------------*
from math import ceil
import torch.nn as nn


class TempConvBlock(nn.Module):
    """
    Temporal Convolutional Block composed of one temporal convolutional layer.
    The block is composed of :
    - Conv1d
    - BatchNorm1d
    - ReLU
    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param dil: Amount of dilation
    :param pad: Amount of padding
    :param groups: Channel groups, defaults to 1.
    """

    def __init__(self, ch_in, ch_out, k_size, dil, pad, groups=1):
        super(TempConvBlock, self).__init__()
        self.tcn = nn.Conv2d(in_channels=ch_in, out_channels=ch_out,
                             kernel_size=k_size, dilation=dil,
                             padding=pad, groups=groups)

        self.bn = nn.BatchNorm2d(num_features=ch_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.tcn(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    """
    Convolutional Block composed of:
    - Conv1d
    - BatchNorm1d
    - ReLU
    - MaxPool1d
    :param ch_in: Number of input channels
    :param ch_out: Number of output channels
    :param k_size: Kernel size
    :param s: Amount of stride
    :param pad: Amount of padding
    :param groups: Channel groups, defaults to 1.
    """

    def __init__(self, ch_in, ch_out, k_size, s, pad, dilation=1, groups=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=ch_in, out_channels=ch_out,
                              kernel_size=k_size, stride=s,
                              dilation=dilation, padding=pad, groups=groups)
        self.bn = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class TEMPONet(nn.Module):
    """
    TEMPONet architecture:
    Three repeated instances of TemporalConvBlock and ConvBlock organized as follows:
    - TemporalConvBlock
    - ConvBlock
    Two instances of FCBlock followed by a final Linear layer with a single neuron.
    """

    def __init__(self):
        super().__init__()

        # Parameters
        self.dil = [2, 2, 1, 4, 4, 8, 8]
        self.rf = [5, 5, 5, 9, 9, 17, 17]
        self.ch = [32, 32, 64, 64, 64, 128, 128, 128, 128, 256, 128]

        # 1st instance of two TempConvBlocks and ConvBlock
        k_tcb00 = ceil(self.rf[0]/self.dil[0])
        self.tcb00 = TempConvBlock(ch_in=8, ch_out=self.ch[0],
                                   k_size=(k_tcb00, 1), dil=(self.dil[0], 1),
                                   pad=(((k_tcb00-1)*self.dil[0]+1)//2, 0))
        k_tcb01 = ceil(self.rf[1]/self.dil[1])
        self.tcb01 = TempConvBlock(ch_in=self.ch[0], ch_out=self.ch[1],
                                   k_size=(k_tcb01, 1), dil=(self.dil[1], 1),
                                   pad=(((k_tcb01-1)*self.dil[1]+1)//2, 0))
        k_cb0 = ceil(self.rf[2]/self.dil[2])
        self.cb0 = ConvBlock(ch_in=self.ch[1], ch_out=self.ch[2],
                             k_size=(k_cb0, 1), s=1, dilation=(self.dil[2], 1),
                             pad=(((k_cb0-1)*self.dil[2]+1)//2, 0))

        # 2nd instance of two TempConvBlocks and ConvBlock
        k_tcb10 = ceil(self.rf[3]/self.dil[3])
        self.tcb10 = TempConvBlock(ch_in=self.ch[2], ch_out=self.ch[3],
                                   k_size=(k_tcb10, 1), dil=(self.dil[3], 1),
                                   pad=(((k_tcb10-1)*self.dil[3]+1)//2, 0))
        k_tcb11 = ceil(self.rf[4]/self.dil[4])
        self.tcb11 = TempConvBlock(ch_in=self.ch[3], ch_out=self.ch[4],
                                   k_size=(k_tcb11, 1), dil=(self.dil[4], 1),
                                   pad=(((k_tcb11-1)*self.dil[4]+1)//2, 0))
        self.cb1 = ConvBlock(ch_in=self.ch[4], ch_out=self.ch[5],
                             k_size=(5, 1), s=(2, 1), pad=(2, 0))

        # 3td instance of TempConvBlock and ConvBlock
        k_tcb20 = ceil(self.rf[5]/self.dil[5])
        self.tcb20 = TempConvBlock(ch_in=self.ch[5], ch_out=self.ch[6],
                                   k_size=(k_tcb20, 1), dil=(self.dil[5], 1),
                                   pad=(((k_tcb20-1)*self.dil[5]+1)//2, 0))
        k_tcb21 = ceil(self.rf[6]/self.dil[6])
        self.tcb21 = TempConvBlock(ch_in=self.ch[6], ch_out=self.ch[7],
                                   k_size=(k_tcb21, 1), dil=(self.dil[6], 1),
                                   pad=(((k_tcb21-1)*self.dil[6]+1)//2, 0))
        self.cb2 = ConvBlock(ch_in=self.ch[7], ch_out=self.ch[8],
                             k_size=(5, 1), s=(4, 1), pad=(4, 0))

        # Final classifier
        self.fc0 = nn.Linear(in_features=self.ch[8]*5, out_features=self.ch[9])
        self.bn0 = nn.BatchNorm1d(num_features=self.ch[9])
        self.relu0 = nn.ReLU()
        self.fc1 = nn.Linear(in_features=self.ch[9], out_features=self.ch[10])
        self.bn1 = nn.BatchNorm1d(num_features=self.ch[10])
        self.relu1 = nn.ReLU()
        self.out_layer = nn.Linear(in_features=self.ch[10], out_features=9)

    def forward(self, input):
        x = self.tcb00(input)
        x = self.tcb01(x)
        x = self.cb0(x)

        x = self.tcb10(x)
        x = self.tcb11(x)
        x = self.cb1(x)

        x = self.tcb20(x)
        x = self.tcb21(x)
        x = self.cb2(x)

        x = x.flatten(1)

        x = self.relu0(self.bn0(self.fc0(x)))
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.out_layer(x)
        return x
