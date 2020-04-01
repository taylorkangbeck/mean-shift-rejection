import torch
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
#from torch._jit_internal import weak_module, weak_script_method

import numpy as np


#@weak_module
class FilterEuclideanRescalingLayer ( nn.Module) :

    ScalingName = 'Scaling'

    def __init__ (self, FilterWeightsParam, NoGrad=True ) :
        
        super ( FilterEuclideanRescalingLayer, self ).__init__()

        # store the filter kernel parameter - as this is a parameter then the normalisation is part of the gradient graph
        self.WeightsParam = FilterWeightsParam

        # suppress gradients of the operation
        # - this greatly reduces computational cost during training and such gradients act as an inflation regularizer that is typically corraled using filter magnitude loss/
        # - also without gradients the LR throttle effect is magnified so that the filter magnitude loss has more effect
        self.NoGrad = NoGrad
      
    def forward ( self, x ) :
        """
        output = input / EuclidMag(WeightsParam)

        notes
        [1] if norm is based on the filter weight parameter then it is also a regulariser for the computational graph but if WeightsParam.data is used then this is outside of the computational graph and not a regulariser other than statistically
        [2] if NoGrad=True then the filter inflation effect does not happen and filters on average remain at saling=1 except for the first layer that sees approx 30% inflation
        """

        NumFilters = x.shape[1] # OR self.WeightsParam.data.shape[1]
        FilterShape = self.WeightsParam.shape
        FilterTrailingDims = tuple ( range ( 1, len(FilterShape) ) )

        # compute using weight data so that there is no gradient implication
        # - this greatly simplifies the comptutational graph and speeds up training and the regulariser that would impose is not necessary since the Euclidean unity scale regulariser performs the role of corraling filter inflation
        if self.NoGrad :
            WeightData = self.WeightsParam.data
        else :
            WeightData = self.WeightsParam

        if len(FilterShape)==4 :
            KernelEuclidNormBroadcast = WeightData.pow(2).sum(dim=FilterTrailingDims).sqrt().view(1,NumFilters,1,1)
        elif len(FilterShape)==2 :
            KernelEuclidNormBroadcast = WeightData.pow(2).sum(dim=(1,)).sqrt().view(1,NumFilters)
        else :
            assert False

        ret = x / KernelEuclidNormBroadcast

        return ret


