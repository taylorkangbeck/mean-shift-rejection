import torch
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import numpy as np
from functools import partial

from Common import *
from TreluLayer import TreluLayer

def atanh ( x ):
    return 0.5 * torch.log ( ( 1 + x ) / ( 1 - x ) )

def Relu ( x, **kwargs ) :
    return F.relu ( x )

def ConveyorRelu ( x, slope=1.0, ripple=1e-6, hip=-0.3, **kwargs ) :

    if hip is None :
        # conveyor is zero for x>0 and is the ripple with speficied slope for x<=0
        RippleSig = slope*torch.remainder ( x, ripple )
        ConveyorX = torch.where ( x<=0.000001, RippleSig, torch.zeros_like(x) )
        ConveyorReluX = F.relu(x) + ConveyorX
    else :
        # - hip variant applies conveyor between hip and zero where hip<0
        # - slope is ignored and the applied slope is linear from 1 at x=0 to 0 at x=-hip
        RippleSig = torch.remainder ( x, ripple ) * ( x - hip )
        ConveyorRangeGate = torch.where ( (x > hip) * (x <= 0.0), RippleSig, torch.zeros_like(x) )
        ConveyorX = RippleSig * ConveyorRangeGate
        ConveyorReluX = F.relu(x) + ConveyorX

    return ConveyorReluX

def Trelu ( x, bias, **kwargs ) :
    return torch.max ( x, bias )

def ChannelEuclidNorm ( x, eps=np.float32(1e-6), **kwargs ) :
    """
    returns x / ( EuclidMag(x) + eps ) over the channel dimensions which are assumed to be the trainling dimensions after (minibatch, channel)
    """
    ChannelDims = tuple ( np.arange ( len(x.shape) )[2:] )
    MagX = x.pow(2).mean(dim=ChannelDims,keepdim=True).sqrt()
    EuclidNormX = x / ( MagX + eps )

    return EuclidNormX

class NonLinType ( EnumBase ) :
    Identity        = "identity"
    Sigmoid         = "sigmoid"
    Relu            = "relu"
    Trelu           = "trelu"
    ConveyorRelu    = 'conveyorrelu'

    @classmethod
    def tolayer ( cls , nonlinenum, **kwargs ) :
        """ convert enum to python nonlinearity method """
        if nonlinenum == NonLinType.Identity :
            return partial ( LambdaLayerNoArgs, func = lambda x : x )
        if nonlinenum == NonLinType.Sigmoid :
            return partial ( LambdaLayerNoArgs, func=F.sigmoid )
        if nonlinenum == NonLinType.Relu :
            return partial ( LambdaLayerNoArgs, func=F.relu )
        if nonlinenum == NonLinType.ConveyorRelu :
            return partial ( LambdaLayerNoArgs, func=partial( ConveyorRelu, **kwargs ) )
        if nonlinenum == NonLinType.Trelu :
            return TreluLayer

        return None

    @classmethod
    def tofunc ( cls , nonlinenum, **kwargs ) :
        """ convert enum to python nonlinearity method """
        if nonlinenum == NonLinType.Relu :
            return ActFunc.Relu
        if nonlinenum == NonLinType.Trelu :
            return ActFunc.Trelu
        if nonlinenum == NonLinType.ConveyorRelu :
            return partial ( ConveyorRelu, **kwargs )

        return None

