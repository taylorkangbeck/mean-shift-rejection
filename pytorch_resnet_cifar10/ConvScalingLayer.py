import torch
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import numpy as np

import TorchConfig as Config

class ConvScalingLayer ( nn.Module) :
    """
    ConvScalingLayer performs standard 2D convolution but the filter is separated into the kernel followed by a trainable scaling parameter. The reason is to support learning rate adaptation to both the scaling and filter kernel.
    """

    ScalingName = 'Scaling'

    def __init__ (  self,
                    Shape, 
                    Stride              = 1,
                    Padding             = 0,
                    Dilation            = 1,
                    Groups              = 1,
                    UseExpoScaling      = False,
                    ScalingParamName    = 'Scaling',
                    WeightParamName     = 'weight',
                    ScalingInit         = lambda x : x,
                    WeightInit          = init.xavier_normal,
                    ScalingNoise        = 0.0,                       # +ve values modulate the scaling parameter with unit mean random uniform noise of amplitude ScalingNoise
                    SharedScaling       = False,
                    HasScaling          = True,
                    WeightTransform     = None, #lambda x : torch.tanh(x/0.2)*0.2,
                ) :
        
        super ( ConvScalingLayer, self ).__init__()

        self.Shape = Shape
        self.Stride = Stride
        self.Padding = Padding
        self.Dilation = Dilation
        self.Groups = Groups
        self.WeightInit = WeightInit
        self.ScalingInit = ScalingInit
        self.ScalingNoise = ScalingNoise
        self.SharedScaling = SharedScaling
        self.HasScaling = HasScaling

        self.NumFilters  = Shape[0]
        self.Depth       = Shape[1]
        self.KernelSize  = Shape[2:]

        self.WeightTransform = WeightTransform

        #if ndim==3 :
        #    # allocate 1D tensor of length in_planes - this is broadcastable to axes=(X,depth,X,X) by setting dimensions of size 1
        #    ScalingInitOnesBroadcastable = torch.from_numpy ( np.ones ( shape=[1,planes,1,1] ) ).float()
        #    self.Scaling = torch.nn.Parameter ( ScalingInitOnesBroadcastable, requires_grad=True )
        #elif ndim==1 :
        #    # allocate 1D tensor of length in_planes - this is broadcastable to axes=(X,depth,X,X) by setting dimensions of size 1
        #    ScalingInitOnesBroadcastable = torch.from_numpy ( np.ones ( shape=[1,planes] ) ).float()
        #    self.Scaling = torch.nn.Parameter ( ScalingInitOnesBroadcastable, requires_grad=True )
        #else :
        #    assert False, "only ndim of 1 or 3 supported"

        self.UseExpoScaling = UseExpoScaling

        if HasScaling :

            # default scaling is vector of 1's and is optionally altered by the ScalingInit method
            if SharedScaling :
                Scaling = ScalingInit ( torch.ones([],dtype=torch.float32) )
            else :
                Scaling = ScalingInit ( torch.ones([self.NumFilters],dtype=torch.float32) )

            # if scaling is exponentiated then take log of the scaling initialisation
            if UseExpoScaling :
                Scaling = Scaling.log()

            # set the scaling param as a named attribute of the module and as a nn.Parameter : this allows the parameter to be searched for by name
            # ORIG param without a name : self.Scaling = torch.nn.Parameter ( ScalingInitBroadcastable, requires_grad=True )
            setattr ( self, ScalingParamName, torch.nn.Parameter ( Scaling, requires_grad=True ))
            self.ScalingParamName = ScalingParamName
        else :
            setattr ( self, ScalingParamName, None )

        ###########################
        # 4D FILTER BANK PARAMETER
        ###########################
        # allocate the weights tensor as zeros and create a parameter
        NumpyWeightData = np.zeros ( Shape, dtype=Config.DefaultNumpyType)
        WeightTensor = torch.tensor(NumpyWeightData, dtype=Config.DefaultType).to(Config.CurrentDevice)
        WeightParam = torch.nn.Parameter ( WeightTensor, requires_grad=True )

        # call the weight init method on the parameter's data which is NOT differentiable as this is in-place init of the param data tensor
        WeightInit ( WeightParam.data )

        # set the weight parameter as a parameter of the module
        setattr ( self, WeightParamName, WeightParam )
        self.WeightParamName = WeightParamName

        # scaling noise applied to scaling parameter to vary its value channel-wise around its mean value - initialized on first inference and whenever depth changes
        self.ScalingEmptyRandomGPU = None

        pass

    def forward(self,x):  # TODO PORT
        """
        output = conv(x, W*Scaling)
        """

        InputShape = x.shape
        ndim=len(InputShape)
        InputDepth = x.shape[1]

        WeightParam = getattr ( self, self.WeightParamName )

        if self.WeightTransform is not None :
            WeightParam = self.WeightTransform ( WeightParam )

        if self.HasScaling :
            ScalingParam = getattr ( self, self.ScalingParamName )

            # optional exponential scaling projection
            if self.UseExpoScaling :
                Scaling = ScalingParam.exp()
            else :
                Scaling = ScalingParam

            # broadcast scaling to dimensions trailing (NumFilters,Depth)
            if self.SharedScaling :
                ScalingDepth=1
            else :
                ScalingDepth = self.Shape[0]

            if ndim==4 :
                ViewShape = ( ScalingDepth, 1, 1, 1 )
            elif ndim==3 :
                ViewShape = ( ScalingDepth, 1, 1 )
            elif ndim==2 :
                ViewShape = ( ScalingDepth, 1 )
            else :
                assert False, "Input shape " + str(InputShape) + " is not supported"

            # Scale the kernel
            ScalingView = Scaling.view(ViewShape)
            ScaledWeights = WeightParam * ScalingView  # TODO IMPORTANT, reparameterizing the kernel

        else :
            # no scaling so plain kernel weights convolution
            ScaledWeights = WeightParam

        # OPTIONAL - kernel scaling by random draw
        # scaling noise applied to scaling parameter to vary its value channel-wise around its mean value - initialized on first inference and whenever depth changes
        if self.ScalingNoise > 0 and self.training :

            # init the empty gpu tensor if none or depth has changed
            # note - scaling affects the kernel for this step and so is affects all samples in the minibatch equally (similar to scale normalization noise in batch norm)
            if self.ScalingEmptyRandomGPU is None or self.ScalingEmptyRandomGPU.shape[0] != InputDepth :
                self.ScalingEmptyRandomGPU = torch.empty ( self.NumFilters, dtype=Config.DefaultType).to(Config.CurrentDevice)

            # make random draw for the scaling
            ScalingDraw = self.ScalingEmptyRandomGPU.uniform_ ( 1.0 - self.ScalingNoise, 1.0+ self.ScalingNoise )

            if ndim==4 :
                ViewShape = ( self.Shape[0], 1, 1, 1 )
            elif ndim==3 :
                ViewShape = ( self.Shape[0], 1, 1 )
            elif ndim==2 :
                ViewShape = ( self.Shape[0], 1 )
            else :
                assert False, "Input shape " + str(InputShape) + " is not supported"

            ScaledWeights = ScaledWeights * ScalingDraw.view(ViewShape)

        # convolution
        # cheaper than multiplying the activations by scaling
        ret =  torch.nn.functional.conv2d ( x, ScaledWeights, bias=None, stride=self.Stride, padding=self.Padding, dilation=self.Dilation, groups=self.Groups)

        return ret
