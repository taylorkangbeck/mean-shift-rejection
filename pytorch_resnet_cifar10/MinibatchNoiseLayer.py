"""
Layer creates minibatch noise similar to Batch Normalization by keeping track of the per map minibatch mean and standard deviation for the previous minibatch and the running means of those and
then applying these to the input by :

    x' = ( x - Prev_MEAN + Running_MEAN ) * Running_VAR/ Prev_VAR
       = x * Running_VAR/ Prev_VAR + ( Prev_MEAN - Running_MEAN ) * Running_VAR / Prev_VAR

This introduces the same noise distribution as batch normalization i.e.
    - the mean offset applied to x is distiburbed by delta minibatch mean which is running_mean - prev_mean where running_mean is the trainable bias term
    - the scale of x is disturbed by running_VAR / prev_VAR where the trainable scaling = running_STD

Note that for validation inference the running mean and VAR replaces the trainable bias and scaling.

The noise is a unit mean scaling of x and additive noise which is the zero mean delta in the per map mean to which is also applied the scaling noise.
It may be important to apply the unit mean scaling noise to the zero mean shift noise as this is what happens with batch norm.

Notes:
    - the momentum term should be set to the equivalent running mean momentum for batch norm which is typically 0.9
    - the noise has no parameters as its purpose is to directly mimic the noise injection arising in batch normalization
    - at inference the layer passes forwards x unaltered
    - when first called the previous and running mean stats do not exist and so these are recorded as the previous and running stats and x is returned unaltered

"""
__authors__   = "Brendan Ruff"
__copyright__ = "(c) 2020, DeepSee AI Ltd"
__license__   = ""
__contact__   = "brendan@deepseeai.co.uk"
__created__   = "21-Jan-2020"
__updated__   = ""

import torch
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

import numpy as np

from Common import EnumBase
import TorchConfig as Config

class MinibatchNoiseLayer ( nn.Module) :
    """
    Per channel noise modulating layer. Two options:
    [1] symmetric zero mean modulating noise so does not shift mean or gain of the filter output
    [2] asymmetric with half +ve Gaussian random scaling so mean of that is is applied as a scaling for inference

    TODO: compute running scaling of the NoiseTensor and apply as a scaling during inference if mode=HalfGauss
    """

    def __init__ (self, planes, Momentum=0.9 ) :
        super ( MinibatchNoiseLayer, self ).__init__()

        self.IsFirstCall = True
        self.RunningMean = None
        self.RunningSTD = None
        self.PrevMean = None
        self.PrevSTD = None
        self.Momentum = torch.tensor ( Momentum, dtype=Config.DefaultType ).to(Config.CurrentDevice)

        self.eps = torch.tensor ( 0.00001, dtype=Config.DefaultType ).to(Config.CurrentDevice)

    def forward(self,x):

        if self.training :

            # compute stats for this minibatch - NB refer to the data not the trainable tensor to avoid extending the history chain
            assert x.ndim == 4, "only 4 dimensional tensors are supported for minibatch noise"
            StatsDims = ( 0, 2, 3 )
            ThisMean = x.data.mean ( dim=StatsDims, keepdim=True )
            ThisSTD = torch.sqrt ( x.data.var ( dim=StatsDims, keepdim=True ) + self.eps )

            # special case that this is the first time called so return x unaltered and initialize the previous and running stats to the current stats
            if self.IsFirstCall :
                self.IsFirstCall = False
                self.RunningMean = ThisMean
                self.RunningSTD = ThisSTD # torch.zeros ( (1,x.data.shape[1], 1, 1), dtype=x.dtype ).to(Config.CurrentDevice)
                self.PrevMean = ThisMean
                self.PrevSTD = ThisSTD

                return x

            else :
                #x_noise = ( x - self.PrevMean + self.RunningMean ) * self.RunningVAR / self.PrevVAR
                # stable version where only the differential component of x is scaled
                scaling = self.RunningSTD / ThisSTD # self.PrevSTD

                if False :
                    # no clip
                    x_noise = ThisMean + ( x - ThisMean ) * scaling #_clip
                elif True :
                    # abs scaling
                    x_noise = x * scaling
                else :
                    # abs scaling with mean noise using current minibatch mean which is therefore identical to batch norm but without the MS gradients as a sparsity regularizer
                    #scaling_clip = torch.where ( scaling < 2.0, scaling, 2.0 * torch.ones_like(scaling) )
                    x_noise = ( x - ThisMean ) * scaling  + self.RunningMean 

                # update stats
                self.RunningMean = self.RunningMean * self.Momentum + ThisMean * ( 1.0 - self.Momentum)
                self.RunningSTD = self.RunningSTD * self.Momentum + ThisSTD * ( 1.0 - self.Momentum)
                self.PrevMean = ThisMean
                self.PrevSTD = ThisSTD

                return x_noise

        else :
            return x



