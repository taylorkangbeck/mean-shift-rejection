import torch
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
#from torch._jit_internal import weak_module, weak_script_method

import numpy as np


#@weak_module
class ScalingLayer ( nn.Module) :

    ScalingName = 'Scaling'

    def __init__ (  self,
                    planes,
                    UseExpoScaling      = False,
                    ScalingParamName    = 'Scaling',
                    Init                = 1.0,
                 ) :
        
        super ( ScalingLayer, self ).__init__()

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

        # allocate 1D tensor of length in_planes - this is broadcastable to axes=(X,depth,X,X) by setting dimensions of size 1
        InitNp =  np.ones ( shape=[planes] ) * Init
        if UseExpoScaling :
            # init + epsilon to prevent zero case, and init must be >=0
            assert Init >= 0.0, "exponential scaling cannot be initialized to -ve number"
            ScalingInitBroadcastable = torch.from_numpy ( np.log ( InitNp + 0.0000001 ) ).float()
        else :
            ScalingInitBroadcastable = torch.from_numpy ( InitNp ).float()

        #self.Scaling = torch.nn.Parameter ( ScalingInitBroadcastable, requires_grad=True )
        #self.__dict__[ScalingParamName+'XX'] = torch.nn.Parameter ( ScalingInitBroadcastable, requires_grad=True )
        setattr ( self, ScalingParamName, torch.nn.Parameter ( ScalingInitBroadcastable, requires_grad=True ))
        self.ScalingParamName = ScalingParamName
        pass

#    @weak_script_method
    def forward(self,x):

        InputShape = x.shape
        ndim=len(InputShape)
        InputDepth = x.shape[1]

        ScalingParam = getattr ( self, self.ScalingParamName )

        # optional exponential scaling projection
        if self.UseExpoScaling :
            scaling = ScalingParam.exp()
        else :
            scaling = ScalingParam

        if ndim==4 :
            ScalingView = scaling.view(1,InputDepth,1,1)
        elif ndim==3 :
            ScalingView = scaling.view(1,InputDepth,1)
        elif ndim==2 :
            ScalingView = scaling.view(1,InputDepth)
        else :
            assert False, "Input shape " + str(InputShape) + " is not supported"
        
        ret = x * ScalingView
        #ret = torch.mul ( x, self.Scaling )
        return ret

