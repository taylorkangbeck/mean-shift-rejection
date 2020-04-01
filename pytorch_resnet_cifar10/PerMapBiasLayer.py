import torch
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import numpy as np

class PerMapBiasLayer ( nn.Module) :

    BiasName = 'Bias'

    def __init__ ( self, planes, BiasParamName='Bias' ) :
        super ( PerMapBiasLayer, self ).__init__()

        #if ndim==3 :
        #    # allocate 1D tensor of length in_planes - this is broadcastable to axes=(X,depth,X,X) by setting dimensions of size 1
        #    BiasInitZeroBroadcastable = torch.from_numpy ( np.zeros ( shape=[1,planes,1,1] ) ).float()
        #elif ndim==2 :
        #    # allocate 1D tensor of length in_planes - this is broadcastable to axes=(X,depth,X,X) by setting dimensions of size 1
        #    BiasInitZeroBroadcastable = torch.from_numpy ( np.zeros ( shape=[1,planes,1,1] ) ).float()
        #else :
        #    assert False, "only ndim of 2 or 4 supported"

        self.BiasParamName = BiasParamName

        # allocate 1D tensor of length in_planes - this is broadcastable to axes=(X,depth,X,X) by setting dimensions of size 1
        BiasInitZeroBroadcastable = torch.from_numpy ( np.zeros ( shape=[planes] ) ).float()

        setattr ( self, BiasParamName, torch.nn.Parameter ( BiasInitZeroBroadcastable, requires_grad=True ))

#    @weak_script_method
    def forward ( self, x ) :

        InputShape = x.shape
        ndim=len(InputShape)
        InputDepth = x.shape[1]

        BiasParam = getattr ( self, self.BiasParamName )

        if ndim==4 :
            BiasView = BiasParam.view(1,InputDepth,1,1)
        elif ndim==3 :
            BiasView = BiasParam.view(1,InputDepth,1)
        elif ndim==2 :
            BiasView = BiasParam.view(1,InputDepth)
        else :
            assert False, "Input shape " + str(InputShape) + " is not supported"

        ret = x + BiasView
        #ret = torch.add(x,1.0,self.Bias)

        return ret

