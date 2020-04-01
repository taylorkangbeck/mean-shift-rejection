'''

'''
import torch
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from functools import partial

from ScalingLayer import ScalingLayer
from PerMapBiasLayer import PerMapBiasLayer
from PerMapRandomScalingLayer import PerMapRandomScalingLayer
from FilterEuclideanRescalingLayer import FilterEuclideanRescalingLayer
from ConvScalingLayer import ConvScalingLayer
from MinibatchNoiseLayer import MinibatchNoiseLayer

from torch.nn import BatchNorm2d
from torch.nn.utils import weight_norm

import ActFunc

from Common import *
import TorchConfig as Config

from collections import namedtuple

import InitUtil

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

WEIGHT_INIT_XAVIER = 'WEIGHT_INIT_XAVIER'
WEIGHT_INIT_METHOD = WEIGHT_INIT_XAVIER

# only used for per channel options
WEIGHT_INIT_SCALING     = 0.75

DEFAULT_BLOCK_WIDTHS            = [16,32,64]
#DEFAULT_BLOCK_WIDTHS            = [16,25,52]

SCALING_LINEAR_RANGE_INIT_LOW   = 1.0 #0.3
SCALING_LINEAR_RANGE_INIT_HIGH  = 1.0 #0.3

###############
# DEFINITIONS
###############

tupWeightScaling = namedtuple('tupWeightScaling', 'WeightName Scaling Weight UseExpoScaling')


def _weights_init ( m, UseZMI, UsePerFilterL2Rescaling ) :

    assert False, "automatic weight init is no longer used both as it is not controllable per layer (no layer names) and because PyTorch BatchNorm scaling init is incorrect"
    classname = m.__class__.__name__

    if isinstance(m, nn.Linear) :
        if not UseZMI :
            init.xavier_normal_(m.weight)
        else :
            init.xavier_normal_(m.weight)

            if False :
                # linear (fully connected) layer has 2 dimensions (filter_num, input_depth)
                WeightData = m.weight.data
                # note - current best practice is little difference between per filter zero mean normal distribution and ZM uniform 2x distribution.
                # At ep3 UniX2 wins but final accuracy is not conclusive, and likely is simply due to higher LR by larger weights so norm x2 would also be faster.
                # However by ep18 some lead was taken by 2xUni so this is currently the default FC init.
                if True :
                    L1_All = torch.mean ( torch.abs(WeightData), dim=(0,1), keepdim=True )
                    # remake the weights as uniform with the same L1
                    UniformWeightData  = np.random.uniform ( -1.0, 1.0, size=m.weight.data.shape ).astype('float32')
                    UniformWeightData -= np.mean ( UniformWeightData, axis=(1,), keepdims=True )
                    UniformWeightData = torch.from_numpy(UniformWeightData)
                    L1_PerFilter = torch.mean ( torch.abs(UniformWeightData), dim=(1,), keepdim=True )
                    L1Scale = L1_All / L1_PerFilter
                    m.weight.data.copy_(UniformWeightData * L1Scale )
                else :
                    # keep spread but adjust per filter mean to zero
                    m.weight.data[...] = 1.0 * ( m.weight.data - torch.mean ( m.weight.data, dim=(1,), keepdim=True ) )

    elif isinstance(m, nn.Conv2d) :
        if WEIGHT_INIT=='WEIGHT_INIT_HE' :
            init.kaiming_normal_ ( m.weight, mode='fan_out', nonlinearity='relu' )
        elif WEIGHT_INIT=='WEIGHT_INIT_XAVIER' :
            init.xavier_normal_(m.weight)
        elif WEIGHT_INIT=='WEIGHT_INIT_UNIFORM' :
            # uniform distribution of range -0.5 to +0.5 has area of 1 noting that in comparison to He and Xavier they have much smaller values in general
            init.uniform(m.weight, a=-1.0, b=1.0)
        else :
            assert False, "Weight init %s not supported" % (WEIGHT_INIT)

        # normalise each 2D kernel (if not 1x1) to zero mean
        WeightData = m.weight.data
        IsConv1x1 = WeightData.shape[2:]==(1,1)

        # TODO: conv1x1 have cross channel zmi

        if IsConv1x1 :
            pass
        else :
            if UseZMI :
                WeightData -= torch.mean (WeightData,dim=(2,3), keepdim=True)

            if UsePerFilterMagBalancing :
                if True :
                    # - the filter is given uniform zero mean distribution with same L1 energy as the mean of the original filter bank
                    # - each filter is given the same L1 energy which is estimated as the L1 across all filters
                    L1_All = torch.mean ( torch.abs(WeightData), dim=(0,1,2,3), keepdim=True )
                    # remake the weights as uniform with the same L1
                    UniformWeightData  = np.random.uniform ( -1.0, 1.0, size=m.weight.data.shape ).astype('float32')
                    UniformWeightData -= np.mean ( UniformWeightData, axis=(2,3), keepdims=True )
                    UniformWeightData = torch.from_numpy(UniformWeightData)
                    L1_PerFilter = torch.mean ( torch.abs(UniformWeightData), dim=(1,2,3), keepdim=True )
                    L1Scale = L1_All / L1_PerFilter
                    m.weight.data.copy_(UniformWeightData * L1Scale )
                    #m.weight.data = UniformWeightData * L1Scale * 0.97 # 0.79 # WEIGHT_INIT_SCALING

                else :
                    # each filter is given the same L1 energy which is estimated as the L1 across all filters
                    L1_All = torch.mean ( torch.abs(WeightData), dim=(0,1,2,3), keepdim=True )
                    L1_PerFilter = torch.mean ( torch.abs(WeightData), dim=(1,2,3), keepdim=True )
                    L1Scale = L1_All / L1_PerFilter
                    WeightData *= L1Scale * 0.97 # 0.79 # WEIGHT_INIT_SCALING

    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) :
        if m.bias is not None :
            m.bias.data.fill_(0.0)

"""
Lambda layer wraps the passed function lambd as a PyTorch module aka layer so operates much like a Python lambda as a lightweight layerless layer.
"""
#class LambdaLayer(nn.Module):
#    def __init__( self, func ) :
#        super(LambdaLayer, self).__init__()
#        self.func = func

#    def forward(self, x):
#        return self.func(x)

class ElementwiseAddLayer(nn.Module):
    def __init__(self):
        super(ElementwiseAddLayer, self).__init__()

    def forward(self, input):
        return input[0] + input[1]


"""
Group of layers that form a residual block.
"""
class ResBlock(nn.Module):

    """
    MODS:
    - init now requires the name of the block to be passed so that all created layers can have their outputs referenced within an attribute dictionary of layers that is accessible
        via block.Layers. The layer names are constructed as block_name followed by ::layername eg MyBlock::Conv1
    """
    def __init__ (  self,
                    block_name,
                    in_planes,
                    planes,
                    stride                  = 1,
                    option                  = 'A',
                    HasBatchNorm            = False,
                    ScalingInitLow        = 1.0,
                    ScalingInitHigh       = 1.0,
                    HasBias                 = False,
                    HasPreBias              = False,
                    HasScaling              = False,
                    UseExpoScaling          = False,
                    SharedScaling           = False,
                    NoConv1Scaling          = False,
                    NoConv2Scaling          = False,
                    HasSkipScaling          = False,
                    HasReluConv1x1          = False,
                    expansion               = 1,
                    ConvNoiseScale          = 0.0,
                    ConvNoiseShift          = 0.0,
                    ConvNoiseMode           = 'full',
                    ScalingNoise            = 0.0,
                    InputNoise              = False,            # now command line
                    Conv1Noise              = False,            # hard coded here - alternative is -scalingnoise which affects on res block conv1 and conv2 by scaling the kernel by 1 +/- scalingnoise
                    Conv2Noise              = False,            # hard coded here
                    ReluNoise               = False,            # hard coded here
                    LRDict                  = None,
                    UseWeightNorm           = False,
                    UseFilterL2Rescale      = False,
                    UseZMI                  = False,
                    UsePerFilterMagBalancing = False,
                    ConvWeightInitMethod    = init.xavier_normal_,
                    ConvInitMagnitudeScale  = None,     # default is to keep whatever magnitude is supplied by the weight init method
                    Conv1x1InitMagnitudeScale = None,
                    ScalingInitMagnitudeScale = None,
                    EpsBN                   = 1e-5,
                    ConvBN                  = None,
                    WeightScaling           = None,
                    AdaptLR2Scaling         = False,
                    UseConvScaling          = False,
                    Conv2ZeroInit           = False,
                    NonLinLay               = partial ( LambdaLayer, func=F.relu ),
                    HasActivationEuclidNorm = False,
                    Conv2KernelSize         = 3,
                    Conv2Padding            = 1,
                   ) :

        super(ResBlock, self).__init__()

        # class var : here instead of class static as partial use does not see the class static (no idea why)
        self.expansion = expansion

        # record the base name of the list of layers
        self.Name = block_name
        self.LRDict = LRDict
        self.HasBatchNorm = HasBatchNorm
        self.ConvNoiseScale = ConvNoiseScale
        self.ConvNoiseShift = ConvNoiseShift
        self.HasScaling = HasScaling
        self.SharedScaling = SharedScaling
        self.UseExpoScaling = UseExpoScaling
        self.NoConv1Scaling = NoConv1Scaling
        self.NoConv2Scaling = NoConv2Scaling
        self.HasSkipScaling = HasSkipScaling
        self.HasBias = HasBias
        self.HasPreBias = HasPreBias
        self.InputNoise = InputNoise
        self.Conv1Noise = Conv1Noise
        self.Conv2Noise = Conv2Noise
        self.ReluNoise = ReluNoise
        self.ScalingNoise = ScalingNoise
        self.UseWeightNorm = UseWeightNorm
        self.UseFilterL2Rescale = UseFilterL2Rescale
        self.UseZMI = UseZMI
        self.UsePerFilterMagBalancing = UsePerFilterMagBalancing
        self.ConvWeightInitMethod = ConvWeightInitMethod
        self.ConvInitMagnitudeScale = ConvInitMagnitudeScale
        self.HasReluConv1x1 = HasReluConv1x1
        self.AdaptLR2Scaling = AdaptLR2Scaling
        self.UseConvScaling = UseConvScaling
        self.Conv2ZeroInit = Conv2ZeroInit
        self.NonLinLay = NonLinLay
        self.HasActivationEuclidNorm = HasActivationEuclidNorm
        self.Conv2KernelSize = Conv2KernelSize
        self.Conv2Padding = Conv2Padding

        # scaling init if any
        if ScalingInitMagnitudeScale is not None :
            ScalingInitPartial = partial(InitUtil.TensorConstantInit, Value=ScalingInitMagnitudeScale)
        elif ScalingInitLow != ScalingInitHigh :
            ScalingInitPartial = partial(InitUtil.RandUniformInit, Low=ScalingInitLow, High=ScalingInitHigh)
        else :
            ScalingInitPartial = partial(InitUtil.TensorConstantInit, Value=ScalingInitHigh)

        # conv layer weight init - if use zmi then init is uniform else Xavier. ZMI removes a lot of the asymmetry in the kernel and normal distribution has too many near zero draws hence uniform distribution
        if UseZMI :
            ConvWeightInitPartial = partial ( InitUtil.ConvWeightInit, Method=InitUtil.ConvParamUniformInit, UseZMI=UseZMI, UsePerFilterMagBalancing=UsePerFilterMagBalancing, InitMagnitudeScale=ConvInitMagnitudeScale )
            #ConvWeightInitPartial = partial ( InitUtil.ConvWeightInit, Method=InitUtil.ConvParamUniformInit, UseZMI=UseZMI, UsePerFilterMagBalancing=UsePerFilterMagBalancing, InitMaxScale=1.0 )
        else :
            ConvWeightInitPartial = partial ( InitUtil.ConvWeightInit, Method=ConvWeightInitMethod, UseZMI=UseZMI, UsePerFilterMagBalancing=UsePerFilterMagBalancing, InitMagnitudeScale=ConvInitMagnitudeScale )

        if Conv2ZeroInit :
            ConvWeightInitPartial1 = ConvWeightInitPartial
            ConvWeightInitPartial2 =  partial ( InitUtil.TensorConstantInit, Value=0.0 )
        else :
            ConvWeightInitPartial1 = ConvWeightInitPartial
            ConvWeightInitPartial2 = ConvWeightInitPartial

        # either use PyTorch Conv2d module or ConvScaling module

        if UseConvScaling :
            self.conv1 = ConvScalingLayer ( Shape=(planes,in_planes,3,3), Stride=stride, Padding=1, WeightInit=ConvWeightInitPartial1, UseExpoScaling=UseExpoScaling, SharedScaling=SharedScaling, HasScaling = not NoConv1Scaling, ScalingInit=ScalingInitPartial, ScalingNoise=ScalingNoise )
            self.conv2 = ConvScalingLayer ( Shape=(planes,planes,Conv2KernelSize,Conv2KernelSize), Padding=Conv2Padding, WeightInit=ConvWeightInitPartial2, UseExpoScaling=UseExpoScaling, SharedScaling=SharedScaling, HasScaling = not NoConv2Scaling, ScalingInit=ScalingInitPartial, ScalingNoise=ScalingNoise )

            if not NoConv1Scaling :
                WeightScaling.append ( tupWeightScaling ( self.Name+'.conv1.weight', self.conv1.Scaling, self.conv1.weight, UseExpoScaling ) )
            if not NoConv2Scaling :
                WeightScaling.append ( tupWeightScaling ( self.Name+'.conv2.weight', self.conv2.Scaling, self.conv2.weight, UseExpoScaling ) )
        else :
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=Conv2KernelSize, stride=1, padding=1, bias=False)
            ConvWeightInitPartial ( self.conv1.weight.data )
            ConvWeightInitPartial ( self.conv2.weight.data )

        if UseWeightNorm :
            self.conv1 = weight_norm ( self.conv1, ParamName='weight', dim=0 )
            self.conv2 = weight_norm ( self.conv2, ParamName='weight', dim=0 )

        if self.UseFilterL2Rescale :
            self.conv1_L2Rescale = FilterEuclideanRescalingLayer ( self.conv1.weight )
            self.conv2_L2Rescale = FilterEuclideanRescalingLayer ( self.conv2.weight )

        if HasBatchNorm :
            self.bn1 = BatchNorm2d(planes, eps=EpsBN)
            self.bn2 = BatchNorm2d(planes, eps=EpsBN)
            BatchNormInit (self.bn1, ScalingInit=ScalingInitPartial )
            BatchNormInit (self.bn2, ScalingInit=ScalingInitPartial )
        else :
            self.bn1 = lambda x : x
            self.bn2 = lambda x : x

        if self.InputNoise and ConvNoiseScale>0.0 :
            self.noise = PerMapRandomScalingLayer ( in_planes, delta=ConvNoiseScale, shift=ConvNoiseShift, mode=ConvNoiseMode )

        if self.Conv1Noise : # and ConvNoiseScale>0.0 :
            #self.conv1_noise = PerMapRandomScalingLayer ( planes, delta=ConvNoiseScale, shift=ConvNoiseShift, mode=ConvNoiseMode )
            self.conv1_noise = MinibatchNoiseLayer ( planes, Momentum=0.9 )

        if self.Conv2Noise : # and ConvNoiseScale>0.0 :
            #self.conv2_noise = PerMapRandomScalingLayer ( planes, delta=ConvNoiseScale, shift=ConvNoiseShift, mode=ConvNoiseMode )
            self.conv2_noise = MinibatchNoiseLayer ( planes, Momentum=0.9 )

        if self.ReluNoise and ConvNoiseScale>0.0 :
            self.relunoise = PerMapRandomScalingLayer ( planes, delta=ConvNoiseScale, shift=ConvNoiseShift, mode=ConvNoiseMode )

        if HasPreBias :
            self.conv1_prebias = PerMapBiasLayer ( in_planes, BiasParamName="PreBias" )
            self.conv2_prebias = PerMapBiasLayer ( planes, BiasParamName="PreBias" )

        if HasBias :
            self.conv1_bias = PerMapBiasLayer ( planes )
            self.conv2_bias = PerMapBiasLayer ( planes )

        if HasScaling :
            self.conv1_scaling = ScalingLayer ( planes, UseExpoScaling=UseExpoScaling, Init=ScalingInitMagnitudeScale )
            self.conv2_scaling = ScalingLayer ( planes, UseExpoScaling=UseExpoScaling, Init=ScalingInitMagnitudeScale )

        if HasSkipScaling :
            # for simplicity scaling is applied to the zero extended skip
            self.skip_scaling = ScalingLayer ( planes, UseExpoScaling=True, Init=1.0, ScalingParamName='SkipScaling' )

        # optional relu + conv1x1 projection after 2nd conv block (including any BN)
        if HasReluConv1x1 :
            self.conv2_nl = self.NonLinLay()
            self.conv3 = nn.Conv2d ( planes, planes, kernel_size=1, stride=1, padding=0, bias=False )
            ConvWeightInit ( self.conv3.weight.data, Method=ConvWeightInitMethod, UseZMI=False, UsePerFilterMagBalancing=UsePerFilterMagBalancing, InitMagnitudeScale=Conv1x1InitMagnitudeScale )

        # nonlinearity layers for conv1 and the block output - note that these get the corresponding layer's output width passed in case that is needed e.g. for trelu, and is ignored if not needed e.g. for relu
        self.conv1_nl = self.NonLinLay ( planes=planes )
        self.output_nl = self.NonLinLay ( planes=planes )

        # SHORTCUT - if strided or change of width then zero padd
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                # strided subsample by 2 in X and Y and pad the depth by the difference between the output conv2 width and the input depth and here arbitrarilly this is padded after the planes of the input
                self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 0, planes-in_planes), "constant", 0))
            elif option == 'B':
                if HasBatchNorm :
                    self.shortcut = nn.Sequential (
                                                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                                                     BatchNorm2d(self.expansion * planes)
                                                  )
                else :
                    self.shortcut = nn.Sequential (
                                                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
                                                  )
        else :
            # empty sequential module passes the input to the output
            self.shortcut = nn.Sequential()

        # residual + skip elementwise addition layer
        self.ResAdd = ElementwiseAddLayer()

        # optionally the conv+bn layers are associated in a tuple appended to the ConvBN list passed
        if ConvBN is not None and HasBatchNorm :
            ConvBN.append ( (self.conv1, self.bn1) )
            ConvBN.append ( (self.conv2, self.bn2) )

    def forward(self, x):

        # reset layers list since every time the net is performed either in inference or training then forward is called and a new list of layers with activations is created
        Layers = [ tupLayerInfo(self.Name+"_input", None, None, x, "ResNetInput") ]

        if self.InputNoise and self.ConvNoiseScale>0.0 :
            Layers.append ( HookUpLayer ( name=self.Name+'.noise', module=self.noise, input=x, tags=['noise'] ) )
            # if there is shift noise then clip range to be +ve since -ve input is meaningless
            if self.ConvNoiseShift > 0.0 :
                Layers.append ( HookUpLayer ( name=self.Name+'.noiserelu', module=self.NonLin, input=Layers[-1].Out, tags=['relu'] ) )
            input = Layers[-1].Out

        else :
            input = x

        if self.HasPreBias :
            Layers.append ( HookUpLayer ( name=self.Name+'.conv1_prebias', module=self.conv1_prebias, input=Layers[-1].Out, tags=['prebias'] ) )
        # conv layer 1
        Layers.append ( HookUpLayer ( name=self.Name+'.conv1', module=self.conv1, input=Layers[-1].Out, tags=['conv'] ) )
        if self.HasActivationEuclidNorm :   # optionally apply activation normalization
            Layers.append ( HookUpLayer ( name='an1', module=LambdaLayer(ActFunc.ChannelEuclidNorm), input=Layers[-1].Out, tags=['an'] ) )
        if self.HasBatchNorm :
            Layers.append ( HookUpLayer ( name=self.Name+'.bn1', module=self.bn1, input=Layers[-1].Out, tags=['bn'] ) )
        if self.HasBias :
            Layers.append ( HookUpLayer ( name=self.Name+'.conv1_bias', module=self.conv1_bias, input=Layers[-1].Out, tags=['bias'] ) )
        if self.HasScaling :
            Layers.append ( HookUpLayer ( name=self.Name+'.conv1_scaling', module=self.conv1_scaling, input=Layers[-1].Out, tags=['scaling'] ) )
        if self.UseFilterL2Rescale :
            Layers.append ( HookUpLayer ( name=self.Name+'.conv1_L2Rescale', module=self.conv1_L2Rescale, input=Layers[-1].Out, tags=['L2rescale'] ) )
        if self.Conv1Noise :
            Layers.append ( HookUpLayer ( name=self.Name+'.conv1_noise', module=self.conv1_noise, input=Layers[-1].Out, tags=['noise'] ) )
        Layers.append ( HookUpLayer ( name=self.Name+'.conv1_nl', module=self.conv1_nl, input=Layers[-1].Out, tags=['conv1_nl'] ) )

        # 2nd conv layer (TODO - put this into a conv block)
        if self.HasPreBias :
            Layers.append ( HookUpLayer ( name=self.Name+'.conv2_prebias', module=self.conv2_prebias, input=Layers[-1].Out, tags=['prebias'] ) )
        # conv layer 2
        Layers.append ( HookUpLayer ( name=self.Name+'.conv2', module=self.conv2, input=Layers[-1].Out, tags=['conv'] ) )
        if self.HasActivationEuclidNorm :   # optionally apply activation normalization
            Layers.append ( HookUpLayer ( name='an2', module=LambdaLayer(ActFunc.ChannelEuclidNorm), input=Layers[-1].Out, tags=['an'] ) )
        if self.HasBatchNorm :
            Layers.append ( HookUpLayer ( name=self.Name+'.bn2', module=self.bn2, input=Layers[-1].Out, tags=['bn'] ) )
        if self.HasBias :
            Layers.append ( HookUpLayer ( name=self.Name+'.conv2_bias', module=self.conv2_bias, input=Layers[-1].Out, tags=['bias'] ) )
        if self.HasScaling :
            Layers.append ( HookUpLayer ( name=self.Name+'.conv2_scaling', module=self.conv2_scaling, input=Layers[-1].Out, tags=['scaling'] ) )
        if self.UseFilterL2Rescale :
            Layers.append ( HookUpLayer ( name=self.Name+'.conv2_L2Rescale', module=self.conv2_L2Rescale, input=Layers[-1].Out, tags=['L2rescale'] ) )
        if self.Conv2Noise :
            Layers.append ( HookUpLayer ( name=self.Name+'.conv2_noise', module=self.conv2_noise, input=Layers[-1].Out, tags=['noise'] ) )

        if self.HasReluConv1x1 :
            Layers.append ( HookUpLayer ( name=self.Name+'.conv2_nl', module=self.conv2_relu, input=Layers[-1].Out, tags=['conv2_nl'] ) )
            Layers.append ( HookUpLayer ( name=self.Name+'.conv3', module=self.conv3, input=Layers[-1].Out, tags=['conv','resproj'] ) )

        # remember the output of the residual branch
        ResBranchLastLayer = Layers[-1]
        # skip
        Layers.append ( HookUpLayer ( name=self.Name+'.skip', module=self.shortcut, input=input, tags=['skip'] ) )

        if self.HasSkipScaling :
            Layers.append ( HookUpLayer ( name=self.Name+'.skipscaling', module=self.skip_scaling, input=Layers[-1].Out, tags=['resskip'] ) )

        # residual add
        Layers.append ( HookUpLayer ( name=self.Name+'.resadd', module=self.ResAdd, input=[Layers[-1].Out,ResBranchLastLayer.Out], tags=['resadd'] ) )

        if self.ReluNoise and self.ConvNoiseScale>0.0 :
            Layers.append ( HookUpLayer ( name=self.Name+'.relu_noise', module=self.relunoise, input=Layers[-1].Out, tags=['noise'] ) )

        # block output nonlinearity
        Layers.append ( HookUpLayer ( name=self.Name+'.nl', module=self.output_nl, input=Layers[-1].Out, tags=['nl'] ) )

        # Create LRDict items for conv weights if there is scaling
        if self.HasScaling and self.AdaptLR2Scaling :
            # scaling is broadcast to (depth,kerny,kernx)
            Scaling1BroadcastShape = ( self.conv1_scaling.Scaling.shape[0], 1, 1, 1 )
            Scaling2BroadcastShape = ( self.conv2_scaling.Scaling.shape[0], 1, 1, 1 )
            if self.UseExpoScaling :
                conv1_scaling_view = self.conv1_scaling.Scaling.exp().view(Scaling1BroadcastShape)
                conv2_scaling_view = self.conv2_scaling.Scaling.exp().view(Scaling2BroadcastShape)
            else :
                conv1_scaling_view = ( self.conv1_scaling.Scaling + 0.001 ).view(Scaling1BroadcastShape)
                conv2_scaling_view = ( self.conv2_scaling.Scaling + 0.001 ).view(Scaling2BroadcastShape)

            self.LRDict[self.Name+'.conv1.weight'] = 1.0 / conv1_scaling_view   #.sqrt()   #.pow(2)
            self.LRDict[self.Name+'.conv2.weight'] = 1.0 / conv2_scaling_view   #.sqrt()   #.pow(2)

        elif self.UseConvScaling and self.AdaptLR2Scaling :
            # scaling is broadcast to (depth,kerny,kernx)
            Scaling1BroadcastShape = ( self.conv1.Scaling.shape[0], 1, 1, 1 )
            Scaling2BroadcastShape = ( self.conv2.Scaling.shape[0], 1, 1, 1 )
            if self.UseExpoScaling :
                conv1_scaling_view = self.conv1.Scaling.exp().view(Scaling1BroadcastShape)
                conv2_scaling_view = self.conv2.Scaling.exp().view(Scaling2BroadcastShape)
            else :
                conv1_scaling_view = ( self.conv1.Scaling + 0.001 ).view(Scaling1BroadcastShape)
                conv2_scaling_view = ( self.conv2.Scaling + 0.001 ).view(Scaling2BroadcastShape)

            self.LRDict[self.Name+'.conv1.weight'] = 1.0 / conv1_scaling_view   #.sqrt()   #.pow(2)
            self.LRDict[self.Name+'.conv2.weight'] = 1.0 / conv2_scaling_view   #.sqrt()   #.pow(2)

        elif self.UseWeightNorm and self.AdaptLR2Scaling :
            Scaling1BroadcastShape = ( self.conv1.weight_g.shape[0], 1, 1, 1 )
            Scaling2BroadcastShape = ( self.conv2.weight_g.shape[0], 1, 1, 1 )
            conv1_scaling_view = ( self.conv1.weight_g + 0.001 ).view(Scaling1BroadcastShape)
            conv2_scaling_view = ( self.conv2.weight_g + 0.001 ).view(Scaling2BroadcastShape)
            self.LRDict[self.Name+'.conv1.weight_v'] = 1.0 / conv1_scaling_view   #.sqrt()   #.pow(2)
            self.LRDict[self.Name+'.conv2.weight_v'] = 1.0 / conv2_scaling_view   #.sqrt()   #.pow(2)
            assert False, 'CHECK INITIAL SCALING AFTER HOOKUP - when is the init called ??'

        elif self.HasBatchNorm and self.AdaptLR2Scaling :
            Scaling1Data = self.bn1.weight.data
            Scaling2Data = self.bn2.weight.data
            # scaling is broadcast to (depth,kerny,kernx)
            Scaling1BroadcastShape = ( Scaling1Data.shape[0], 1, 1, 1 )
            Scaling2BroadcastShape = ( Scaling2Data.shape[0], 1, 1, 1 )
            if self.UseExpoScaling :
                Conv1Scaling = Scaling1Data.exp()
                Conv2Scaling = Scaling2Data.exp()
            else :
                Conv1Scaling = Scaling1Data
                Conv2Scaling = Scaling2Data
            self.LRDict[self.Name+'.conv1.weight'] = 1.0 / ( Conv1Scaling + 0.001).view(Scaling1BroadcastShape)
            self.LRDict[self.Name+'.conv2.weight'] = 1.0 / ( Conv2Scaling + 0.001).view(Scaling2BroadcastShape)


        return Layers

    def ORIGforward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):

    """
    Notes:
    - LRDict if passed is populated with any learning rate adjustments to parameters in which the parameter is the index. This is just a quick hack as a net independent general method should be externally applied
        to associate for instance scaling with its convolutional layer.
    """
    def __init__ (  self,
                    block,
                    num_blocks,
                    num_classes                 = 10,
                    UseBatchNorm                = False,
                    ScalingInitLow              = 1.0,
                    ScalingInitHigh             = 1.0,
                    HasBias                     = False,
                    InputHasBias                = False,
                    ClassHasBias                = False,
                    HasScaling                  = False,
                    UseConvScaling              = False,
                    NoConv1Scaling              = False,
                    NoConv2Scaling              = False,
                    InputConvHasScaling         = False,
                    SharedScaling               = False,
                    UseExpoScaling              = False,
                    HasReluConv1x1              = False,
                    UsePerFilterMagBalancing    = False,
                    UseZMI                      = False,
                    HasPreBias                  = False,
                    ClassHasScaling             = False,
                    ConvNoiseScale              = 0.0,
                    ConvNoiseShift              = 0.0,
                    ConvNoiseMode               = 'full',
                    UseResBlockInpuNoise        = False,
                    Conv1Noise                  = False,
                    Conv2Noise                  = False,
                    ScalingNoise                = 0.0,
                    LRDict                      = None,
                    UseWeightNorm               = False,        # use PyTorch in-built weight_now on conv3x3 layers
                    AdaptLR2Scaling             = False,
                    UseFilterL2Rescale          = False,
                    ClassifierWeightInitMethod  = init.xavier_normal_,
                    ClassifierInitMagnitudeScale = None,
                    ConvInitMagnitudeScale      = None,
                    Conv1x1InitMagnitudeScale   = None,
                    ScalingInitMagnitudeScale   = None,
                    ConvWeightInitMethod        = init.xavier_normal_,  # ConvParamUniformInit, 
                    EpsBN                       = 1e-5,
                    BlockWidth                  = DEFAULT_BLOCK_WIDTHS,       # NB widths must be divisible by 4 for the padding layer though this is just a code oversight that can be corrected
                    Conv2ZeroInit               = False,
                    NonLinLay                   = partial ( LambdaLayer, func=F.relu ),
                    HasActivationEuclidNorm     = False,                        # y' = y / ( EuclidMag (y) + eps ) where y = conv(x)
                ) :

        super(ResNet, self).__init__()

        # output the architecture's block widths to ensure this is clear
        print ( "Architecture block widths = " + str(BlockWidth))

        # remember the passed LRDict for use in the forward pass : this is a reference and not a copy
        self.LRDict = LRDict

        # in_planes is standard PyTorch naming for the depth of the first conv layer
        self.BlockWidth = BlockWidth
        self.Conv1Width = BlockWidth[0]

        self.HasReluConv1x1 = HasReluConv1x1
        self.UseBatchNorm = UseBatchNorm
        self.UseWeightNorm = UseWeightNorm
        self.HasBias = HasBias
        self.HasPreBias = HasPreBias
        self.ClassHasBias = ClassHasBias
        self.InputHasBias = InputHasBias
        self.HasScaling = HasScaling
        self.SharedScaling = SharedScaling
        self.NoConv1Scaling = NoConv1Scaling
        self.NoConv2Scaling = NoConv2Scaling
        self.UseExpoScaling = UseExpoScaling
        self.ClassHasScaling = ClassHasScaling
        self.HasBatchNorm = UseBatchNorm
        self.ConvNoiseScale = ConvNoiseScale
        self.ConvNoiseShift = ConvNoiseShift
        self.HasConvNoise = ConvNoiseScale>0.0 or ConvNoiseShift>0.0
        self.ConvNoiseMode = ConvNoiseMode
        self.ScalingNoise = ScalingNoise
        self.UseResBlockInpuNoise = UseResBlockInpuNoise
        self.UsePerFilterMagBalancing = UsePerFilterMagBalancing
        self.UseZMI = UseZMI
        self.UseFilterL2Rescale = UseFilterL2Rescale
        self.AdaptLR2Scaling = AdaptLR2Scaling
        self.UseConvScaling = UseConvScaling
        self.ScalingInitMagnitudeScale = ScalingInitMagnitudeScale
        self.ScalingInitLow = ScalingInitLow
        self.ScalingInitHigh = ScalingInitHigh
        self.NonLinLay = NonLinLay
        self.HasActivationEuclidNorm = HasActivationEuclidNorm
        self.InputConvHasScaling = InputConvHasScaling

        # list of conv+bn layer tuples so that the BN for a conv is associated
        self.ConvBN = []
        # list of filter weight and corresponding scaling (if any) tuples
        self.WeightScaling = []

        # conv layer weight init - if use zmi then init is uniform else Xavier. ZMI removes a lot of the asymmetry in the kernel and normal distribution has too many near zero draws hence uniform distribution
        if UseZMI :
            ConvWeightInitPartial = partial ( InitUtil.ConvWeightInit, Method=InitUtil.ConvParamUniformInit, UseZMI=UseZMI, UsePerFilterMagBalancing=UsePerFilterMagBalancing, InitMagnitudeScale=ConvInitMagnitudeScale )
            Conv1WeightInitPartial = partial ( InitUtil.ConvWeightInit, Method=InitUtil.ConvParamUniformInit, UseZMI=UseZMI, UsePerFilterMagBalancing=UsePerFilterMagBalancing, InitMagnitudeScale=1.0 )
        else :
            ConvWeightInitPartial = partial ( InitUtil.ConvWeightInit, Method=ConvWeightInitMethod, UseZMI=UseZMI, UsePerFilterMagBalancing=UsePerFilterMagBalancing, InitMagnitudeScale=ConvInitMagnitudeScale )
            Conv1WeightInitPartial = partial ( InitUtil.ConvWeightInit, Method=ConvWeightInitMethod, UseZMI=UseZMI, UsePerFilterMagBalancing=UsePerFilterMagBalancing, InitMagnitudeScale=1.0 )

        # scaling init if any
        if ScalingInitMagnitudeScale is not None :
            ScalingInitPartial = partial(InitUtil.TensorConstantInit, Value=ScalingInitMagnitudeScale)
        elif ScalingInitLow != ScalingInitHigh :
            ScalingInitPartial = partial(InitUtil.RandUniformInit, Low=ScalingInitLow, High=ScalingInitHigh)
        else :
            ScalingInitPartial = partial(InitUtil.RandUniformInit, Low=ScalingInitLow, High=ScalingInitHigh)
            ScalingInitPartial = partial(InitUtil.TensorConstantInit, Value=ScalingInitHigh)

        # fill in standard parameters of res block using partial
        BlockPartial = partial (    block,
                                    UseZMI                  = UseZMI,
                                    UsePerFilterMagBalancing= UsePerFilterMagBalancing,
                                    ConvWeightInitMethod    = ConvWeightInitPartial,    #ConvWeightInitMethod,
                                    ConvInitMagnitudeScale  = ConvInitMagnitudeScale,
                                    Conv1x1InitMagnitudeScale = Conv1x1InitMagnitudeScale,
                                    ScalingInitMagnitudeScale = ScalingInitMagnitudeScale,
                                    HasReluConv1x1          = HasReluConv1x1,
                                    EpsBN                   = EpsBN,
                                    ConvBN                  = self.ConvBN,
                                    WeightScaling           = self.WeightScaling,
                                    AdaptLR2Scaling         = AdaptLR2Scaling,
                                    UseConvScaling          = UseConvScaling,
                                    UseExpoScaling          = UseExpoScaling,
                                    ScalingInitLow          = ScalingInitLow,
                                    ScalingInitHigh         = ScalingInitHigh,
                                    HasPreBias              = HasPreBias,
                                    Conv2ZeroInit           = Conv2ZeroInit,
                                    NonLinLay               = self.NonLinLay,
                                    HasActivationEuclidNorm = HasActivationEuclidNorm,
                                    InputNoise              = UseResBlockInpuNoise,
                                    Conv1Noise              = Conv1Noise,
                                    Conv2Noise              = Conv2Noise,
                                    ConvNoiseScale          = self.ConvNoiseScale,
                                    ConvNoiseShift          = self.ConvNoiseShift,
                                    ConvNoiseMode           = self.ConvNoiseMode,
                                    ScalingNoise            = self.ScalingNoise,
                                )

        # first conv layer with RGB input
        if UseConvScaling :
            # shared scaling : self.conv1 = ConvScalingLayer ( Shape=(self.Conv1Width,3,3,3), Padding=1, WeightInit=ConvWeightInitPartial, UseExpoScaling=UseExpoScaling, ScalingInit=ScalingInitPartial, SharedScaling=SharedScaling )
            # special case first layer - independent scaling initialized to unity, scaling param also named after first layer to avoid it being regularized as with residual block scaling
            # init=1: self.conv1 = ConvScalingLayer ( Shape=(self.Conv1Width,3,3,3), Padding=1, WeightInit=ConvWeightInitPartial, UseExpoScaling=UseExpoScaling, ScalingParamName='Conv1Scaling', ScalingInit=partial(InitUtil.TensorConstantInit, Value=1.0), SharedScaling=False )
            if True :
                self.conv1 = ConvScalingLayer ( Shape=(self.Conv1Width,3,3,3), Padding=1, WeightInit=Conv1WeightInitPartial, UseExpoScaling=UseExpoScaling, ScalingParamName='Conv1Scaling', WeightParamName='Conv1Weight', ScalingInit=partial(InitUtil.TensorConstantInit, Value=1.0), SharedScaling=SharedScaling, HasScaling=InputConvHasScaling )
                #self.conv1 = ConvScalingLayer ( Shape=(self.Conv1Width,3,3,3), Padding=1, WeightInit=ConvWeightInitPartial, UseExpoScaling=UseExpoScaling, ScalingParamName='Conv1Scaling', WeightParamName='Conv1Weight', ScalingInit=ScalingInitPartial, SharedScaling=SharedScaling, HasScaling=InputConvHasScaling )
            else :
                # input conv layer conv1 has no scaling otherwise inflates
                self.conv1 = ConvScalingLayer ( Shape=(self.Conv1Width,3,3,3), Padding=1, WeightInit=Conv1WeightInitPartial, UseExpoScaling=UseExpoScaling, WeightParamName='Conv1Weight', HasScaling=False )

            if InputConvHasScaling :
                self.WeightScaling.append ( tupWeightScaling ( 'conv1.Conv1Weight', self.conv1.Conv1Scaling, self.conv1.Conv1Weight, UseExpoScaling ) )

        else :
            self.conv1 = ConvScalingLayer ( Shape=(self.Conv1Width,3,3,3), Padding=1, WeightInit=ConvWeightInitPartial, WeightParamName='Conv1Weight', HasScaling=False )
            # self.conv1 = nn.Conv2d ( 3, self.Conv1Width, kernel_size=3, stride=1, padding=1, bias=False)
            # ConvWeightInitPartial ( self.conv1.weight.data )

        # optional extend conv1 with pre-hooked in-built weight norm
        if UseWeightNorm :
            self.conv1 = weight_norm ( self.conv1, ParamName='weight', dim=0 )

        if UseFilterL2Rescale :
            self.conv1_L2Rescale = FilterEuclideanRescalingLayer ( self.conv1.weight )

        if UseBatchNorm :
            self.bn1 = BatchNorm2d(self.Conv1Width, eps=EpsBN)
            BatchNormInit (self.bn1, ScalingInit=ScalingInitPartial )
            self.ConvBN.append ( (self.conv1,self.bn1) )
        else :
            self.bn1 = lambda x : x

        if self.HasConvNoise and False :
            self.conv1_noise = PerMapRandomScalingLayer ( self.Conv1Width, delta=ConvNoiseScale, shift=ConvNoiseShift, mode=self.ConvNoiseMode )

        if HasScaling :
            self.conv1_scaling = ScalingLayer ( self.Conv1Width, UseExpoScaling=UseExpoScaling, Init=1.0 )

        if HasBias or InputHasBias :
            self.conv1_bias = PerMapBiasLayer ( self.Conv1Width )

        self.conv1_nl = self.NonLinLay ( planes=self.Conv1Width )

        # create the resolution blocks noting that each block is a list of residual blocks i.e. a list of list of layers (modules)
        self.Block1LayerList = self._MakeResolutionBlock ( 'block1', BlockPartial, BlockWidth[0], input_planes=self.Conv1Width, num_blocks=num_blocks[0], stride=1 )
        self.Block2LayerList = self._MakeResolutionBlock ( 'block2', BlockPartial, BlockWidth[1], input_planes=BlockWidth[0],num_blocks=num_blocks[1], stride=2 )
        self.Block3LayerList = self._MakeResolutionBlock ( 'block3', BlockPartial, BlockWidth[2], input_planes=BlockWidth[1], num_blocks=num_blocks[2], stride=2 )

        # put all layers (i.e. modules) within the net into the class dictionary according to their layer name so that self.apply picks up on these to cause the weight initialisation to happen
        PutBlockListIntoModuleDict ( self, self.Block1LayerList )
        PutBlockListIntoModuleDict ( self, self.Block2LayerList )
        PutBlockListIntoModuleDict ( self, self.Block3LayerList )

        # CLASSIFIER - note the bias is separate to allow for its LR to be modified by naming it ClassifierBias to distinguish it from conv layer biases
        self.linear = nn.Linear(BlockWidth[2], num_classes, bias=False )
        InitUtil.LinearWeightInit ( self.linear.weight.data, Method=ClassifierWeightInitMethod, InitMagnitudeScale=ClassifierInitMagnitudeScale, UseZMI=False )
        assert self.linear.bias is None, "DO NOT USE A BIAS IN A LINEAR MODULE - instead create a separate bias module so that the bias is named ClassifierBias"

        if ClassHasBias :
            self.classifier_bias = PerMapBiasLayer ( num_classes, BiasParamName='ClassifierBias' )
#            self.classifier_bias = PerMapBiasLayer ( num_classes, BiasParamName='Bias' )
        if UseFilterL2Rescale and ClassHasScaling :
            self.classifier_L2Rescale = FilterEuclideanRescalingLayer ( self.linear.weight )
        if ClassHasScaling :
            self.classifier_scaling = ScalingLayer ( num_classes, ScalingParamName='ClassifierScaling', UseExpoScaling=UseExpoScaling, Init=1.5 )

        # single function call that applies the same weight init to all layers using module base apply() method which presumably causes all nn.modules referenced by class members to get initialised ? Weird !
        # TODO : ensure all parameters are initialised in the layer constructors with explicitly passed init methods
        # self.apply ( partial(_weights_init, UseZMI=UseZMI, UsePerFilterMagBalancing=UsePerFilterMagBalancing) )
        # REMOVED - layers are now directly initialised by function call

    def _MakeResolutionBlock ( self, block_name, block, planes, input_planes, num_blocks, stride ) :

        LayersList = []
        strides = [stride] + [1]*(num_blocks-1)
        LastBlockWidth = input_planes
        for (thisstride,idx) in zip(strides,range(len(strides))) :
            ThisBlock = block ( block_name+"_"+str(idx),
                                LastBlockWidth,
                                planes,
                                thisstride,
                                HasBatchNorm=self.HasBatchNorm,
                                HasBias=self.HasBias,
                                HasScaling=self.HasScaling,
                                SharedScaling=self.SharedScaling,
                                UseExpoScaling=self.UseExpoScaling,
                                NoConv1Scaling=self.NoConv1Scaling,
                                NoConv2Scaling=self.NoConv2Scaling,
                                LRDict=self.LRDict,
                                UseWeightNorm=self.UseWeightNorm,
                                UseFilterL2Rescale=self.UseFilterL2Rescale )

            LayersList.append ( ThisBlock )
            LastBlockWidth = planes * ThisBlock.expansion

        # shortcut for hooking layers up in the order in the list as a single function (i.e. sub-graph) that can be called from the computational graph
        return LayersList

    # creates list of layers in order of activation so that these may be hooked up in order in the forward call
    def _MakeResolutionBlockSequential ( self, block, planes, num_blocks, stride ) :

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append ( block ( self.in_planes, planes, stride, HasBatchNorm=self.HasBatchNorm, HasBias=self.HasBias, HasScaling=self.HasScaling, ConvNoiseScale=self.ConvNoiseScale, ConvNoiseShift=self.ConvNoiseShift, ConvNoiseMode=self.ConvNoiseMode, LRDict=self.LRDict ) )
            self.in_planes = planes * block.expansion

        # shortcut for hooking layers up in the order in the list as a single function (i.e. sub-graph) that can be called from the computational graph
        return nn.Sequential(*layers)


    def forward(self, x):

        # reset layers list since every time the net is performed either in inference or training then forward is called and a new list of layers with activations is created
        # note - Layers[] is a complete list of residual blocks in the ResNet each of which blocks has a group of layers in the architecture of a residual block and the blocks are added as units by name e.g. block1_4 which contains for instance block1_4.conv1
        Layers = []

        # first conv layer (TODO - put this into a conv block)
        Layers.append ( HookUpLayer ( name='conv1', module=self.conv1, input=x, tags=['conv'] ) )

        # optionally apply activation normalization
        if self.HasActivationEuclidNorm :
            Layers.append ( HookUpLayer ( name='an1', module=LambdaLayer(ActFunc.ChannelEuclidNorm), input=Layers[-1].Out, tags=['an'] ) )

        if self.UseBatchNorm :
            Layers.append ( HookUpLayer ( name='bn1', module=self.bn1, input=Layers[-1].Out, tags=['bn'] ) )
        if self.HasBias or self.InputHasBias :
            Layers.append ( HookUpLayer ( name='conv1_bias', module=self.conv1_bias, input=Layers[-1].Out, tags=['bias'] ) )
        if self.HasScaling :
            Layers.append ( HookUpLayer ( name='conv1_scaling', module=self.conv1_scaling, input=Layers[-1].Out, tags=['scaling'] ) )
        if self.UseFilterL2Rescale :
            Layers.append ( HookUpLayer ( name='conv1_L2Rescale', module=self.conv1_L2Rescale, input=Layers[-1].Out, tags=['L2rescale'] ) )
        if self.HasConvNoise and False :
            Layers.append ( HookUpLayer ( name='conv1_noise', module=self.conv1_noise, input=Layers[-1].Out, tags=['noise'] ) )
        Layers.append ( HookUpLayer ( name='conv1_nl', module=self.conv1_nl, input=Layers[-1].Out, tags=['input_nl'] ) )

        # call the blocks of layers one layer at a time with the proviso that a block is a linear set of layers in a daisychain
        Layers += HookUpLayers ( name='block1', modules=self.Block1LayerList, input=Layers[-1].Out, tags=['block1'] )
        Layers += HookUpLayers ( name='block2', modules=self.Block2LayerList, input=Layers[-1].Out, tags=['block2'] )
        Layers += HookUpLayers ( name='block3', modules=self.Block3LayerList, input=Layers[-1].Out, tags=['block3'] )

        # create a standard named hook for the encoder output tensor so that this can be accessed by name
        Layers.append ( tupLayerInfo ( 'enc_out', None, Layers[-1].Out, Layers[-1].Out, 'enc_out' ) )

        # final average pool and classifier - graph rather than packed into module
        AvgPoolOut = F.avg_pool2d(Layers[-1].Out, Layers[-1].Out.size()[3])
        AvgPoolOutResize = AvgPoolOut.view(AvgPoolOut.size(0), -1)

        Layers.append ( HookUpLayer ( name='classifier', module=self.linear, input=AvgPoolOutResize, tags=['classifier'] ) )
        if self.ClassHasBias :
            Layers.append ( HookUpLayer ( name='classifier_bias', module=self.classifier_bias, input=Layers[-1].Out, tags=['classbias'] ) )
        if self.ClassHasScaling :
            Layers.append ( HookUpLayer ( name='classifier_scaling', module=self.classifier_scaling, input=Layers[-1].Out, tags=['classscaling'] ) )
        if self.UseFilterL2Rescale and self.ClassHasScaling :
            Layers.append ( HookUpLayer ( name='classifier_L2Rescale', module=self.classifier_L2Rescale, input=Layers[-1].Out, tags=['classL2rescale'] ) )

        ###############################
        # per parameter LR adjustments - these may not be enabled and so could be unused
        # notes :
        # - gradients are not tracked
        ###############################

        if self.HasScaling and self.AdaptLR2Scaling :
            ScalingData = self.conv1_scaling.Scaling.data
            # scaling is broadcast to (depth,kerny,kernx)
            ScalingBroadcastShape = ( ScalingData.shape[0], 1, 1, 1 )
            if self.UseExpoScaling :
                Conv1Scaling = ScalingData.exp()
            else :
                Conv1Scaling = ScalingData
            self.LRDict['conv1.weight'] = 1.0 / ( Conv1Scaling + 0.01).view(ScalingBroadcastShape)

        if self.UseConvScaling and self.AdaptLR2Scaling and self.InputConvHasScaling :
            ScalingData = self.conv1.Scaling.data
            # scaling is broadcast to (depth,kerny,kernx)
            ScalingBroadcastShape = ( ScalingData.shape[0], 1, 1, 1 )
            if self.UseExpoScaling :
                Conv1Scaling = ScalingData.exp()
            else :
                Conv1Scaling = ScalingData
            self.LRDict['conv1.weight'] = 1.0 / ( Conv1Scaling + 0.01).view(ScalingBroadcastShape)

        if self.UseBatchNorm and self.AdaptLR2Scaling :
            ScalingData = self.bn1.weight.data
            # scaling is broadcast to (depth,kerny,kernx)
            ScalingBroadcastShape = ( ScalingData.shape[0], 1, 1, 1 )
            if self.UseExpoScaling :
                Conv1Scaling = ScalingData.exp()
            else :
                Conv1Scaling = ScalingData
            self.LRDict['conv1.weight'] = 1.0 / ( Conv1Scaling + 0.01).view(ScalingBroadcastShape)   #.sqrt()   #.pow(2)

        # CONV1_BIAS adjusted to CONV1_WEIGHT euclidean magnitude

        # STANDARD WEIGHT NORMALIZATION
        if self.UseWeightNorm :
            # scaling is broadcast to (depth,kerny,kernx)
            ScalingBroadcastShape = ( self.conv1.weight_g.shape[0], 1, 1, 1 )
            self.LRDict['conv1.weight_v'] = 1.0 / ( self.conv1.weight_g + 0.001).view(ScalingBroadcastShape)   #.sqrt()   #.pow(2)

        return Layers


def resnet20 ( **kwargs ) :
    return ResNet ( ResBlock, [3, 3, 3], **kwargs )

def resnet32( **kwargs ) :
    return ResNet( ResBlock, [5, 5, 5], **kwargs )

def resnet44( **kwargs ) :
    return ResNet( ResBlock, [7, 7, 7], **kwargs )

def resnet56( **kwargs ) :
    return ResNet( ResBlock, [9, 9, 9], **kwargs )

def resnet110( **kwargs ) :
    return ResNet( ResBlock, [18, 18, 18], **kwargs )

def resnet1202( **kwargs ) :
    return ResNet( ResBlock, [200, 200, 200], **kwargs )

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
