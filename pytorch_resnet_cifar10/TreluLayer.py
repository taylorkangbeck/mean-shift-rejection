import torch
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import InitUtil
import ActFunc
import TorchConfig as Config

import numpy as np
from functools import partial


class TreluLayer ( nn.Module ) :

    BiasName = 'Tau'

    def __init__ ( self, planes, BiasParamName='TreluBias', BiasInitValue=None ) :
        """
        Passing BiasInitValue=None causes the bias to be initialized to zero and be trainable. Specifying a scalar value causes the bias to be a non-trainable constant set to that value.
        """

        super ( TreluLayer, self ).__init__()

        if BiasInitValue is None :
            BiasIsTrainable = True
            BiasInitValue = 0.0

        else :
            BiasIsTrainable = False

        BiasInitZeros = np.zeros ( shape=[planes], dtype=Config.DefaultNumpyType )
        BiasInit = InitUtil.TensorConstantInit ( torch.from_numpy ( BiasInitZeros ).float(), BiasInitValue )
        setattr ( self, BiasParamName, torch.nn.Parameter ( BiasInit, requires_grad=BiasIsTrainable ))

        self.BiasParamName = BiasParamName


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

        # elementwise max : trelu(x,bias) = max(x,bias)
        ret = ActFunc.Trelu ( x, BiasView )

        return ret

