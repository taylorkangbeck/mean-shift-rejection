import torch
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

import numpy as np

from Common import EnumBase
import TorchConfig as Config

class enumNoiseMode (EnumBase) :
    Half        = 'half'
    Full        = 'full'
    L1Modulate  = 'L1Modulate'

class PerMapRandomScalingLayer ( nn.Module) :
    """
    Per channel noise modulating layer. Two options:
    [1] symmetric zero mean modulating noise so does not shift mean or gain of the filter output
    [2] asymmetric with half +ve Gaussian random scaling so mean of that is is applied as a scaling for inference

    TODO: compute running scaling of the NoiseTensor and apply as a scaling during inference if mode=HalfGauss
    """

    def __init__ (self, planes, delta=0.15, shift=0.15, mode=enumNoiseMode.L1Modulate, DrawBroadcastToMinibatch=False, UseGaussianNoiseDistribution=False  ) :
        super ( PerMapRandomScalingLayer, self ).__init__()

        # allocate 1D tensor of length in_planes - this is broadcastable to axes=(X,depth,X,X) by setting dimensions of size 1 - the value is unimportant as it will be drawn as uniform random on each step
        #RandomTensor = torch.from_numpy ( np.ones ( shape=[1,planes,1,1] ) ).float()
        #self.RandomTensor = torch.nn.Parameter ( RandomTensor, requires_grad=False )
        #self.RandomTensor = torch.nn.Parameter ( torch.tensor([0]*planes).float(), requires_grad=False )
        #self.RandomTensor = torch.tensor([0]*planes).float().cuda()
        # range delta for distribution : 1-delta to 1+delta
        self.Delta = delta 
        self.Shift = shift   
        self.Mode = mode
        self.DrawBroadcastToMinibatch = DrawBroadcastToMinibatch
        self.UseGaussianNoiseDistribution = UseGaussianNoiseDistribution

        assert type(mode)==enumNoiseMode, "Mode must be of type enumMode"
        if  mode==enumNoiseMode.Full :
            self.EvalScaling = 1.0
        elif mode==enumNoiseMode.Half :
            self.EvalScaling = delta * ( 1.0 + np.sqrt(2.0/3.141592653) )
        else :
            self.EvalScaling = 1.0
  
        self.EmptyRandomGPU = None

    def forward(self,x):

        if self.training :
            InputDepth = x.shape[1]
            MinibatchSize = x.shape[0]

            if self.Mode == enumNoiseMode.Full :
                if self.DrawBroadcastToMinibatch :
                    #NoiseTensor = torch.clamp ( torch.cuda.FloatTensor(InputDepth).normal_ ( 1.0, self.Delta).view(1,InputDepth,1,1), 0.0, 2.0 )
                    #ShiftTensor = torch.cuda.FloatTensor(InputDepth).normal_ ( 0.0, self.Delta).view(1,InputDepth,1,1)
                    if self.UseGaussianNoiseDistribution :
                        NoiseTensor = torch.clamp ( torch.empty(InputDepth, dtype=Config.DefaultType).normal_ ( 1.0, self.Delta).view(1,InputDepth,1,1), 0.0, 2.0 ).to(Config.CurrentDevice)
                        ShiftTensor = torch.empty(InputDepth, dtype=Config.DefaultType).normal_ ( 0.0, self.Shift).view(1,InputDepth,1,1).to(Config.CurrentDevice)
                    else :  # uniform noise
                        NoiseTensor = torch.clamp ( torch.empty(InputDepth, dtype=Config.DefaultType).uniform_ ( 1.0 - self.Delta, 1.0 + self.Delta ).view(1,InputDepth,1,1), 0.0, 2.0 ).to(Config.CurrentDevice)
                        ShiftTensor = torch.empty(InputDepth, dtype=Config.DefaultType).uniform_ ( -self.Shift, self.Shift).view(1,InputDepth,1,1).to(Config.CurrentDevice)
                else :
                    # if not yet allocated or not same shape then create an empty GPU tensor for the noise
                    if self.EmptyRandomGPU is None or self.EmptyRandomGPU.shape[0] != MinibatchSize :
                        self.EmptyRandomGPU = torch.empty(MinibatchSize*InputDepth, dtype=Config.DefaultType).to(Config.CurrentDevice)
                    #NoiseTensor = torch.clamp ( torch.cuda.FloatTensor(MinibatchSize*InputDepth).normal_ ( 1.0, self.Delta).view(MinibatchSize,InputDepth,1,1), 0.0, 2.0 )
                    #NoiseTensor = torch.clamp ( torch.cuda.FloatTensor(MinibatchSize*InputDepth).uniform_ ( 1.0 - self.Delta, 1.0 + self.Delta ).view(MinibatchSize,InputDepth,1,1), 0.0, 2.0 )
                    #ShiftTensor = torch.cuda.FloatTensor(MinibatchSize*InputDepth).uniform_ ( -self.Shift, self.Shift).view(MinibatchSize,InputDepth,1,1)
                    NoiseTensor = torch.clamp ( self.EmptyRandomGPU.uniform_ ( 1.0 - self.Delta, 1.0 + self.Delta ).view(MinibatchSize,InputDepth,1,1), 0.0, 2.0 )
                    ShiftTensor = self.EmptyRandomGPU.uniform_ ( -self.Shift, self.Shift).view(MinibatchSize,InputDepth,1,1)

            elif self.Mode==enumNoiseMode.L1Modulate :
                """
                F_b_c = relu ( 1 - L1_b_c / ( L1_b + reg )
                Act_b_c = Act_b_c * ( 1 + UniNoise_b_c(-1,+1) * F_b_c )

                So if the channel L1 is close to the layer L1 then there is no noise but if the channel L1 is small then there is noise with a scale in the range 0..1 and
                overall noise in the range up to -1 to 1 so that the channel gain is in the range 0 .. 2 with mean of 1 hence no scaling bias in the activation per channel on average.

                """
                N_b_c = torch.cuda.empty(MinibatchSize*InputDepth, dtype=Config.DefaultType).uniform_ ( -self.Delta, self.Delta ).view(MinibatchSize,InputDepth,1,1)
                L1_b_c = F.relu (x).mean(dim=(2,3),keepdim=True)
                L1_b = L1_b_c.max(dim=1,keepdim=True)[0]
                F_b_c = F.relu ( 1.0 - 0.5 * L1_b_c / ( L1_b + 0.00000001 ) )    # ratio scaled by 0.5 so even max has 0.5 noise
                Scale_b_c = 1.0 + N_b_c * F_b_c
                NoiseTensor = Scale_b_c

            else :
                assert False, "half mode not yet implemented and needs running mean scaling"
                NoiseTensor = 1.0 + torch.abs ( torch.cuda.Tensor(InputDepth, dtype=Config.DefaultType).normal_ ( 0.0, self.Delta) )

            #self.RandomTensor.uniform ( 1.0-self.Delta, 1.0+self.Delta).float()
            if self.Shift > 0.0 :
                ret = ( x - ShiftTensor ) * NoiseTensor.type(Config.DefaultType)
            else :
                ret = x * NoiseTensor.type(Config.DefaultType)

        else :
            ret = x * self.EvalScaling

        return ret


