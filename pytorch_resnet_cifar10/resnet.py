'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

import numpy as np

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

USE_BATCH_NORM              = True
USE_CONV_BIAS               = False

USE_PER_CHANNEL_ZM_INIT     = False

USE_PER_CHANNEL_RESCALING   = False
USE_PER_FILTER_L1_RESCALING = False

#WEIGHT_INIT             = 'WEIGHT_INIT_HE'
WEIGHT_INIT             = 'WEIGHT_INIT_XAVIER'

# only used for per channel options
WEIGHT_INIT_SCALING     = 0.75

def _weights_init(m):

    classname = m.__class__.__name__

    if isinstance(m, nn.Linear) :
        if WEIGHT_INIT=='WEIGHT_INIT_HE' :
            init.kaiming_normal_ ( m.weight, mode='fan_out', nonlinearity='relu' )
        elif WEIGHT_INIT=='WEIGHT_INIT_XAVIER' :
            init.xavier_normal_(m.weight)
            
    elif isinstance(m, nn.Conv2d) :
        if WEIGHT_INIT=='WEIGHT_INIT_HE' :
            init.kaiming_normal_ ( m.weight, mode='fan_out', nonlinearity='relu' )
        elif WEIGHT_INIT=='WEIGHT_INIT_XAVIER' :
            init.xavier_normal_(m.weight)
        elif WEIGHT_INIT=='WEIGHT_INIT_NORMAL_RUFF1' :
            # uniform distribution of range -0.5 to +0.5 has area of 1 noting that in comparison to He and Xavier they have much smaller values in general
            init.uniform(m.weight, a=-0.5, b=0.5)
        else :
            assert False, "Weight init %s not supported" % (WEIGHT_INIT)

        # normalise each 2D kernel (if not 1x1) to zero mean
        WeightData = m.weight.data
        IsConv1x1 = WeightData.shape[2:]==(1,1)

        # DEBUG - measure filter asymmetry
        L1_All = torch.mean ( torch.abs(WeightData), dim=(0,1,2,3), keepdim=True )
        L1_PerFilter = torch.mean ( torch.abs(WeightData), dim=(1,2,3), keepdim=True )
        L1Scale = (L1_All / L1_PerFilter)
        L1Scale = L1Scale.reshape(WeightData.shape[0])

        if not IsConv1x1 :
            if USE_PER_CHANNEL_ZM_INIT :
                WeightData -= torch.mean (WeightData,dim=(2,3), keepdim=True)

            # optionally rescale weights per channel to set a standard range to the max weight scaled by 1/FilterDepth
            # note - this ensures that all channels have similar scale as some may coincidentally have very low amplitude which may delay their learning (or be a good thing, who knows ?)
            if USE_PER_CHANNEL_RESCALING :
                # normalise to the max magnitude 
                AbsWeightData = torch.abs(WeightData)
                # take max over dim=(2,3) noting that torch only supports single dimension max and returns tuple(max,argmax)
                MaxAbsWeightData = torch.max ( torch.max ( AbsWeightData, dim=3, keepdim=True, out=None )[0], dim=2, keepdim=True, out=None )[0]
                # compute filter gamma as 1/sqrt(D) which is based on the Euclidean norm of unit input channels as a vector
                FilterDepth = WeightData.shape[1]
                FilterGamma = np.sqrt(FilterDepth)
                # rescale gamma/per_chan_maxabs
                OrigWeightData = WeightData
                WeightData = WeightData / MaxAbsWeightData
                # OPT1 : restore the L1 over all axes of the weights by rescaling to L1_orig/L1_maxgammascaled
                # OPT2 : per filter energy restored which keeps energy asymmetry between fitlers
                # OPT3 : per filter per channel energy restored which keeps filter asymmetry between channels
                opt=2
                if opt==1:
                    L1_orig = torch.mean ( torch.abs ( OrigWeightData) )
                    L1_rescaled = torch.mean ( torch.abs(WeightData) )
                elif opt==2:
                    L1_orig = torch.mean ( torch.abs ( OrigWeightData), dim=(1,2,3), keepdim=True )
                    L1_rescaled = torch.mean ( torch.abs(WeightData), dim=(1,2,3), keepdim=True )
                else :
                    L1_orig = torch.mean ( torch.abs ( OrigWeightData), dim=(2,3), keepdim=True )
                    L1_rescaled = torch.mean ( torch.abs(WeightData), dim=(2,3), keepdim=True )
                #L1Scale = L1_orig.item() / L1_rescaled.item()
                L1Scale = L1_orig / L1_rescaled
                WeightData *= L1Scale * WEIGHT_INIT_SCALING

            elif USE_PER_FILTER_L1_RESCALING :
                # each filter is given the same L1 energy which is estimated as the L1 across all filters
                L1_All = torch.mean ( torch.abs(WeightData), dim=(0,1,2,3), keepdim=True )
                L1_PerFilter = torch.mean ( torch.abs(WeightData), dim=(1,2,3), keepdim=True )
                L1Scale = L1_All / L1_PerFilter
                WeightData *= L1Scale * 0.79 # WEIGHT_INIT_SCALING
                #WeightData = WeightData / L1Scale * 0.68     # variant that increases the L1 asymmetry between filters

            else :
                # scale weights to reduce fwd gain
                #WeightData *= WEIGHT_INIT_SCALING
                pass

        else :
            pass

        # if there is a bias initialise it to zero
        if m.bias is not None :
            m.bias.data.fill_(0.0)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', HasBatchNorm=USE_BATCH_NORM, HasConvBias=USE_CONV_BIAS):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=HasConvBias)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=HasConvBias)
        if HasBatchNorm :
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        else :
            self.bn1 = lambda x : x
            self.bn2 = lambda x : x

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                if HasBatchNorm :
                    self.shortcut = nn.Sequential (
                                                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=HasConvBias),
                                                     nn.BatchNorm2d(self.expansion * planes)
                                                  )
                else :
                    self.shortcut = nn.Sequential (
                                                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=HasConvBias)
                                                  )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, HasBatchNorm=USE_BATCH_NORM, HasConvBias=USE_CONV_BIAS):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=HasConvBias)

        if HasBatchNorm :
            self.bn1 = nn.BatchNorm2d(16)
        else :
            self.bn1 = lambda x : x

        self.layer1 = self._MakeResolutionBlock(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._MakeResolutionBlock(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._MakeResolutionBlock(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _MakeResolutionBlock(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        # shortcut for hooking layers up in the order in the list as a single function (i.e. sub-graph) that can be called from the computational graph
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3])
def resnet32(**kwargs):
    return ResNet(BasicBlock, [5, 5, 5])
def resnet44(**kwargs):
    return ResNet(BasicBlock, [7, 7, 7])
def resnet56(**kwargs):
    return ResNet(BasicBlock, [9, 9, 9])
def resnet110(**kwargs):
    return ResNet(BasicBlock, [18, 18, 18])
def resnet1202(**kwargs):
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()