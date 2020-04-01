import argparse
import os
import time
import sys
import shutil

import torch
import torch.tensor as T
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import ResnetBiasScaling

import numpy as np
from time import sleep
from itertools import islice

from torch.utils.tensorboard import SummaryWriter
from TensorBoardLogWriter import TensorBoardLogWriter

import itertools
from AttributeDict import AttributeDict
import ParamUtil

from PerMapBiasLayer import PerMapBiasLayer
from ScalingLayer import ScalingLayer
from PerMapRandomScalingLayer import enumNoiseMode
from TreluLayer import TreluLayer

# imports common functions directly into this namespace for readability
from Common import *
import TorchConfig as Config
import ActFunc

#import torch.nn.functional as F

model_names = sorted(name for name in ResnetBiasScaling.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(ResnetBiasScaling.__dict__[name]))

# globals
class Globs () :
    UseZeroMeanGrads    = False
    best_prec1          = 0.0

def SetupCommandLineArgs () :
    parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')

    parser.add_argument('-debugprintparams', dest='debugprintparams', action='store_true', help='switch to print out param info')

    parser.add_argument('logdir', default=None, type=str, help='path to log loss and error and validation and training stats groups : this is a required variable and may not be overwritten for safety so manually delete previous runs')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet32',
                        choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                        ' (default: resnet32)')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers (default: 4 for 6 core processor with 2 threads per core and 3xGPU i.e 12 threads and 4 allocated to feed each CPU)')
    parser.add_argument('-epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-validateprewarmsteps', default=0, type=int, help='number of fwd pass steps to perform on the validation set so that the batch norm runnings stats catch up with the model')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    #parser.add_argument('-lr', '--learning-rate', default=0.1, type=float,
    #                    metavar='LR', help='initial learning rate')
    parser.add_argument('--print-freq', '-p', default=50, type=int,
                        metavar='N', help='print frequency (default: 20)')
    parser.add_argument('-model', default=None, type=str, help='model to load')
    parser.add_argument('-resume', dest='resume', action='store_true', help='resume training of the input model in a directory named loagdir_RESUME')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('-half', dest='half', action='store_true', help='use half-precision(16-bit) ')
    parser.add_argument('-preval', dest='preval', action='store_true', help='validate the initial model before training')
    parser.add_argument('-perchannel', dest='perchannel', action='store_true', help='selects per channel zero mean i.e. over dim=(2,3) of the grads tensor for conv filter')
    parser.add_argument('-l2actloss', dest='l2actloss', type=float, default=0.0, help='L2 activation loss scaling applied to the ouput of convolutional layers regardless of their kernel shape')
    parser.add_argument('-l2negactloss', type=float, default=0.0, help='L2 negative signal activation loss scaling applied to the ouput of convolutional layers regardless of their kernel shape - this encourages +ve direction of filters without using a non-linearity')
    parser.add_argument('-filterscaleloss', dest='filterscaleloss', type=float, default=0.0, help='convolution kernel magnitude difference from anchor loss scaling applied to the ouput of convolutional layers regardless of their kernel shape')
    parser.add_argument('-filterl2scaleloss', dest='filterl2scaleloss', type=float, default=0.0, help='convolution kernel L2 magnitude difference from anchor loss scaling applied to the ouput of convolutional layers regardless of their kernel shape')
    parser.add_argument('-filterscalelossanchor', dest='filterscalelossanchor', type=float, default=0.0, help='convolution kernel magnitude difference from unity loss scaling applied to the ouput of convolutional layers regardless of their kernel shape')
    parser.add_argument('-infoloss', type=float, default=0.0, help='')
    parser.add_argument('-l2actmagloss', type=float, default=0.0, help='')

    parser.add_argument('-prewarm', default=0, type=int, help='number of steps in pre-warm : default=0')
    parser.add_argument('-ope', default=0, type=int, help='forces console output on the defined number of steps with a default of 0 which indicates 16 outputs per minibatch')
    parser.add_argument('-overwrite', dest='overwrite', action='store_true', help='CAUTION : setting this overwrites any existing log directory files')
    parser.add_argument('-tvmodel', dest='tvmodel', action='store_true', help='Use torchvision model where available')
    parser.add_argument('-noaug', dest='noaug', action='store_true', help='do not augment training using random flip and crop : default is to augment')

    parser.add_argument('-batchnorm', dest='batchnorm', action='store_true', help='')
    parser.add_argument('-bias', dest='bias', action='store_true', help='')
    parser.add_argument('-inputconvbias', action='store_true', help='')
    parser.add_argument('-prebias', dest='prebias', action='store_true', help='')
    parser.add_argument('-scaling', dest='scaling', action='store_true', help='')
    parser.add_argument('-noconv1scaling', action='store_true', help='')
    parser.add_argument('-noconv2scaling', action='store_true', help='')
    parser.add_argument('-inputconvscaling', action='store_true', help='')
    parser.add_argument('-sharedscaling', dest='sharedscaling', action='store_true', help='')
    parser.add_argument('-exposcaling', dest='exposcaling', action='store_true', help='')

    parser.add_argument('-conv2zeroinit', action='store_true', help='')

    parser.add_argument('-zmg', dest='zmg', type=float, default=0.0, help='mean gradient kernel stabilisation fraction : default is zero which is no stabilisation')
    parser.add_argument('-zmi', dest='zmi', action='store_true', help='')
    parser.add_argument('-zmw', default=0.0, type=float, help='if specified then per per channel L1 zero mean weight decay is applied')
    parser.add_argument('-l2zmd', default=0.0, type=float, help='if specified then per per channel L2 zero mean weight decay is applied')
    parser.add_argument('-l2zsd', default=0.0, type=float, help='if specified then per per channel L2 zero sum weight decay is applied')

    parser.add_argument('-perfiltmagbalance', dest='perfiltmagbalance', action='store_true', help='each filter in a filter bank is rescaled to the average euclidean magnitude of the filters in the filter bank')
    parser.add_argument('-inputhasbias', dest='inputhasbias', action='store_true', help='')
    parser.add_argument('-classhasbias', dest='classhasbias', action='store_true', help='')
    parser.add_argument('-classhasscaling', dest='classhasscaling', action='store_true', help='')

    parser.add_argument('-weightnoise', default=0.0, type=float, help='noise injection scale for weights based on the channel-wise gradient mean : default is 0 (no noise) and a typical value is ?')
    parser.add_argument('-weightnoiseperweight', action='store_true', help='if true then noise injection into weights is independent for each weight, otherwise it is common to all weights within each filter')
    parser.add_argument('-weightnoiseweightscaling', action='store_true', help='if true then noise injection into weights is scaled by absolute weight size independently for each weight')
    parser.add_argument('-weightnoisermsperfilter', action='store_true', help='select weight noise estimate of grad scale per filter, otherwise it is computed per layer')
    parser.add_argument('-scalingnoise', default=0.0, type=float, help='Random unit mean uniform modulation of the scaling within the convscaling layer if applicable : default is 0 (no noise) and a typical value is 0.1')
    parser.add_argument('-noise', default=0.0, type=float, help='Random uniform modulation of the output of each convolutional layer at the raw conv output after any random noise shift : default is 0 (no noise) and a typical value is 0.1')
    parser.add_argument('-noiseshift', default=0.0, type=float, help='Random uniform shift in of the output of each convolutional layer at the raw conv output : default is 0 (no noise) and a typical value is 0.1')
    parser.add_argument('-noisemode', default=enumNoiseMode.Full.value, type=str, help='Mode for noise is half or full Gaussian')
    parser.add_argument('-inputnoise', action='store_true', help='noise injection is at resblock input')
    parser.add_argument('-conv1noise', action='store_true', help='minibatch noise layer at resblock conv1 output')
    parser.add_argument('-conv2noise', action='store_true', help='minibatch noise layer at resblock conv1 output')
    parser.add_argument('-biaslr', default=None, type=float, help='if specified then the bias has this lr separately to all other params lr')
    parser.add_argument('-scalinglr', default=None, type=float, help='if specified then the scaling has this lr separately to all other params lr')
    parser.add_argument('-classlr', default=None, type=float, help='if specified then the classifier has this lr separately to all other params lr')
    parser.add_argument('-classbiaslr', default=None, type=float, help='if specified then the classifier bias has this lr separately to all other params lr')
    parser.add_argument('-lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('-lrgamma', default=0.1, type=float, help='learning rate decay in optimiser schedule')
    parser.add_argument('-annealdecay', dest='annealdecay', action='store_true', help='L2weight decay is annealed to match the reduction in LR during the LR annealing schedule')
    parser.add_argument('-milestones', action='store', nargs='+', type=int, default=(100,150), help='LR annealing milestones' )
    parser.add_argument('-momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('-l2decay', default=0.0, type=float, help='weight decay (default:0, typical is 5e-4)')
    parser.add_argument('-l1decay', default=0.0, type=float, help='weight decay (default:0, typical is 1e-5)')

    parser.add_argument('-silentstats', dest='silentstats', action='store_true', help='stops training stats progress bars so only epoch validation is shown')
    parser.add_argument('-showtiming', dest='showtiming', action='store_true', help='output timing of each fwd-bwd pass to console')
    parser.add_argument('-schedpoints', nargs='+', type=int, default=[50,75], help='Learning rate annealing schedule : default = '+str('[50,75]') )
    parser.add_argument('-schedrate', nargs='+', type=float, default=[0.1,0.1], help='Learning rate annealing schedule : default = '+str('[0.1,0.1]') )
    parser.add_argument('-validate', dest='validate', action='store_true', help='perform a validation without training')

    parser.add_argument('-euclidnormxfer', dest='euclidnormxfer', action='store_true', help='')
    parser.add_argument('-adaptconvlr', dest='adaptconvlr', action='store_true', help='')
    parser.add_argument('-weightnorm', dest='weightnorm', action='store_true', help='applies PyTorch inbuilt weight norm to 3x3 conv layers so that separate scaling is not needed, and if LRAdapt is used then this is applied using the weight now scaling')
    parser.add_argument('-filterl2rescale', dest='filterl2rescale', action='store_true', help='conv filters output are rescaled by the Euclidean norm of the filter kernels i.e. y = conv(x,Wi) / EuclidNorm(Wi)')
    parser.add_argument('-adaptbiaslr', dest='adaptbiaslr', action='store_true', help='bias gradients are adjusted to 1/mag(filter)')
    parser.add_argument('-scaleconvgradsbymag', dest='scaleconvgradsbymag', action='store_true', help='gradients of conv filters are rescaled by inverse Euclidean norm of the filter kernels i.e grad(W) *= EuclidNorm(W)' )
    parser.add_argument('-resbalanceloss', type=float, default=0.0, help='')

    parser.add_argument('-conv1scalingunitydecay', type=float, default=0.0, help='conv1 scaling unity decay loss scaling : default=0 so not active')
    parser.add_argument('-conv1unitydecay', type=float, default=0.0, help='')
    parser.add_argument('-filterresetthresh', dest='filterresetthresh', type=float, default=0.0, help='Filter Euclidean magnitude below which the filter will be reset to unity magnitude : default is zero i.e. innactive')
    parser.add_argument('-classunitymag', dest='classunitymag', action='store_true', help='if enabled then the classifier is Euclidean magnitude normalised to unity after each training step outside of the graph which prevents local gradient adaptation and loss is unity scaled')
    parser.add_argument('-classunitydecay', dest='classunitydecay', type=float, default=0.0, help='Classifier Euclidean magnitude unity decay loss scaling : default=0 so not active')
    parser.add_argument('-classl2decay', dest='classl2decay', type=float, default=0.0, help='Classifier L2 decay loss scaling : default=0 so not active')
    parser.add_argument('-projunitydecay', dest='projunitydecay', type=float, default=0.0, help='Conv1x1 residual projection Euclidean magnitude unity decay loss scaling : default=0 so not active')
    parser.add_argument('-randomisedreset', dest='randomisedreset', action='store_true', help='reset filters are randomised and per channel zero mean rather than just magnitude rescaled')
    parser.add_argument('-resetthreshisclamp', dest='resetthreshisclamp', action='store_true', help='reset filters are clamped to the magnitude of the threshold which prevents excessive local LR multiplication i.e. > 1 / ClampThresh')
    parser.add_argument('-classinitmag', dest='classinitmag', type=float, default=None, help='Classifier init Euclidean magnitude : default=None so not active')
    parser.add_argument('-convinitmag', dest='convinitmag', type=float, default=None, help='conv layer init Euclidean magnitude : default=None so not active')
    parser.add_argument('-scalinginitmag', dest='scalinginitmag', type=float, default=None, help='scaling init Euclidean magnitude : default=None so not active')
    parser.add_argument('-resprojinitmag', dest='resprojinitmag', type=float, default=None, help='residual projection conv1x1 layer init Euclidean magnitude : default=None so not active')
    parser.add_argument('-treluinitmag', type=float, default=None, help='trelu bias initial value, and if None then is trainable')
    parser.add_argument('-classl2actloss', type=float, default=0.0, help='')
    parser.add_argument('-scalingbalanceloss', type=float, default=0.0, help='')
    parser.add_argument('-sparsityloss', type=float, default=0.0, help='')
    parser.add_argument('-elrloss', type=float, default=0.0, help='')
    parser.add_argument('-scalingl2unitydecay', dest='scalingl2unitydecay', type=float, default=0.0, help='')
    parser.add_argument('-scalingl2decay', type=float, default=0.0, help='')
    parser.add_argument('-scalingfloatingmeandecay', dest='scalingfloatingmeandecay', type=float, default=0.0, help='')
    parser.add_argument('-reluconv1x1', action='store_true', help='residual blocks have conv2 relu and conv3 1x1 random init projection')
    parser.add_argument('-resprojunitymag', dest='resprojunitymag', action='store_true', help='res projection conv1x1 has its magnitude reset after every training step so that it mixes but does not scale')
    parser.add_argument('-convunitymag', dest='convunitymag', action='store_true', help='conv3x3 (1st conv3x3 1ayer and all residual conv3c3 has its magnitude reset after every training step so that it mixes but does not scale')
    parser.add_argument('-flipnegscaling', dest='flipnegscaling', action='store_true', help='if scaling goes -ve then at end of epoch the sign of both the scaling and filter weights is flipped')
    parser.add_argument('-scalinglowerbound', dest='scalinglowerbound', type=float, default=None, help='lower bound applied to scaling params : default=None so not active')
    parser.add_argument('-normgrad2mag', dest='normgrad2mag', action='store_true', help='normalize the gradients of conv kernels to the Euclidean magnitude of the kernel : grad(W)/=mag(W)')
    parser.add_argument('-convgrad2elr', action='store_true', help='adapt the gradients of conv kernels to their ELR by grad * abs(W) / scaling')

    parser.add_argument('-convscaling', dest='convscaling', action='store_true', help='ConvScalingLayer is used for efficient scaling of the conv weights tensor rather than scaling the conv output')
    parser.add_argument('-adaptscalinglr', action='store_true', help='scaling param gradients are adjusted by multiplying by their magnitude so that ELRM is unity')
    parser.add_argument('-mipscaling', type=float, default=0.0, help='if not zero then is the adaptive filter gradient modification to the filter inflation pressure')

    parser.add_argument('-scalinginitlow', dest='scalinginitlow', type=float, default=1.0, help='lower value for bn scaling init : default=1.0')
    parser.add_argument('-scalinginithigh', dest='scalinginithigh', type=float, default=1.0, help='upper value for bn scaling init : default=1.0')
    parser.add_argument('-nl', default='relu', type=str, help='nonlinearity for the entire net : default is relu : alternatives (relu,trelu)')
    parser.add_argument('-actnorm', dest='actnorm', action='store_true', help='Euclidean normalization layer is placed after the convolution layers')
    parser.add_argument('-gradmaxnorm', action='store_true', help='normalizes the weight update stage by normalizing the gradient to |Wi|/max(|Wi|). An espison is also applied by max(|Wi|,eps) and max(max(|Wi|), eps) to probide a lower bound for the gradient scaling and for numerical stability')
    parser.add_argument('-skipscaling', action='store_true', help='skip has exponential scaling init 1 so that the skip may be reduced')

    parser.add_argument('-classresetmag', default=None, type=float, help='')
    parser.add_argument('-resetclass2scaling', action='store_true', help='')
    parser.add_argument('-resetfiltermag', action='store_true', help='')
    parser.add_argument('-resetfilt2scaling', action='store_true', help='')
    parser.add_argument('-xferscaling2weights', action='store_true', help='')
    parser.add_argument('-dualminibatch', action='store_true', help='')

    # NOT YET USED
    parser.add_argument('-weightscaling', default=1.0, type=float, help='')

    parser.add_argument('-save', dest='save_dir', help='The directory used to save the trained models', default='save_temp', type=str)
    parser.add_argument('--save-every', dest='save_every', help='Saves checkpoints at every specified number of epochs', type=int, default=10)

    return parser

def ProcessCommandLineArgs ( Globs, args ) :

    Globs.DebugPrintParams      = args.debugprintparams
    Globs.Net                   = args.arch
    Globs.NumEpochs             = args.epochs
    Globs.Resume                = args.resume
    Globs.ValidateOnly          = args.validate
    Globs.Model                 = args.model
    Globs.ZeroMeanGrad          = args.zmg
    Globs.PreValidate           = args.preval
    Globs.ValidationPreWarmNumSteps = args.validateprewarmsteps
    Globs.PerChannel            = args.perchannel
    Globs.PreWarm               = args.prewarm
    Globs.SchedPointsLR         = args.schedpoints
    Globs.SchedRateLR           = args.schedrate
    Globs.GammaLR               = args.lrgamma
    Globs.LR                    = args.lr
    Globs.BiasLR                = args.biaslr
    Globs.ScalingLR             = args.scalinglr
    Globs.ClassLR               = args.classlr
    Globs.ClassBiasLR           = args.classbiaslr
    Globs.OutputEvery           = args.ope
    Globs.UsePyTorchModel       = args.tvmodel
    Globs.AugmentTraining       = not args.noaug
    Globs.UseBatchNorm          = args.batchnorm
    Globs.HasBias               = args.bias
    Globs.HasPreBias            = args.prebias
    Globs.HasScaling            = args.scaling
    Globs.NoConv1Scaling        = args.noconv1scaling
    Globs.NoConv2Scaling        = args.noconv2scaling
    Globs.InputConvHasScaling   = args.inputconvscaling
    Globs.HasScaling            = args.scaling
    Globs.UseExpoScaling        = args.exposcaling
    Globs.SharedScaling         = args.sharedscaling
    Globs.UsePerFilterMagBalancing = args.perfiltmagbalance
    Globs.UseZMI                = args.zmi
    Globs.InputHasBias          = args.inputhasbias
    Globs.ClassHasBias          = args.classhasbias
    Globs.ClassHasScaling       = args.classhasscaling
    Globs.HasReluConv1x1        = args.reluconv1x1

    Globs.Conv2ZeroInit         = args.conv2zeroinit

    Globs.ConvNoiseScale                = args.noise
    Globs.ConvNoiseShift                = args.noiseshift
    Globs.ConvNoiseMode                 = enumNoiseMode.fromstring ( args.noisemode )
    Globs.ScalingNoise                  = args.scalingnoise
    Globs.UseResBlockInpuNoise          = args.inputnoise
    Globs.Conv1Noise                    = args.conv1noise
    Globs.Conv2Noise                    = args.conv2noise

    Globs.SilentTrainingStats           = args.silentstats
    Globs.ShowFwdBwdTiming              = args.showtiming
    Globs.ZMW                           = args.zmw
    Globs.L2ActLossScaling              = args.l2actloss
    Globs.L2NegActLossScaling           = args.l2negactloss
    Globs.L2InformationLossScaling      = args.infoloss
    Globs.L2MeanActMagLossScaling       = args.l2actmagloss
    Globs.UseFilterEuclidNormXferToScaling = args.euclidnormxfer
    Globs.AdaptLR2Scaling               = args.adaptconvlr
    Globs.ResBlockScalingBalanceLoss    = args.resbalanceloss
    Globs.ScalingLowerBound             = args.scalinglowerbound
    Globs.UseWeightNorm                 = args.weightnorm
    Globs.UseFilterL2Rescale            = args.filterl2rescale
    Globs.ScaleConvGradientsByMag       = args.scaleconvgradsbymag
    Globs.NormConvGrad2KernelMag        = args.normgrad2mag
    Globs.NormaliseConvGrad2ELR         = args.convgrad2elr
    Globs.AdaptScalingLR                = args.adaptscalinglr
    Globs.MIPScaling                    = args.mipscaling
    Globs.FilterScaleLoss               = args.filterscaleloss
    Globs.FilterL2ScaleLoss             = args.filterl2scaleloss
    Globs.FilterScaleLossAnchor         = args.filterscalelossanchor
    Globs.ZeroMeanL2Decay               = args.l2zmd
    Globs.ZeroSumL2Decay                = args.l2zsd

    Globs.ConvScalingBalanceLossScaling = args.scalingbalanceloss
    Globs.SparsityLossScaling           = args.sparsityloss
    Globs.ELRLossScaling                = args.elrloss

    Globs.ClassifierL2ActLossScaling    = args.classl2actloss
    Globs.ScalingL2UnityDecay           = args.scalingl2unitydecay
    Globs.ScalingL2Decay                = args.scalingl2decay
    Globs.ScalingFloatingMeanDecay      = args.scalingfloatingmeandecay
    Globs.L2Decay                       = args.l2decay
    Globs.L1Decay                       = args.l1decay
    Globs.ProjUnityDecayLoss            = args.projunitydecay
    Globs.ClassifierUnityDecayLoss      = args.classunitydecay
    Globs.ClassifierL2Decay             = args.classl2decay
    Globs.Conv1ScalingUnityAnchorLoss   = args.conv1scalingunitydecay
    Globs.Conv1UnityDecayLoss           = args.conv1unitydecay

    Globs.AdaptBiasLR2FilterMag         = args.adaptbiaslr

    Globs.FilterEuclidMagResetThresh    = args.filterresetthresh
    Globs.ResetThreshIsClamp            = args.resetthreshisclamp
    Globs.RandomiseFilterReset          = args.randomisedreset

    Globs.TransferScalingtoWeights      = args.xferscaling2weights
    Globs.ResetClassifierToScaling      = args.resetclass2scaling
    Globs.ResetFilterToScaling          = args.resetfilt2scaling
    Globs.ResetFilterMagnitude          = args.resetfiltermag
    Globs.ClassScaleResetMag            = args.classresetmag
    Globs.ClassifierUnityMag            = args.classunitymag
    Globs.ClassifierInitMagnitudeScale  = args.classinitmag
    Globs.ConvInitMagnitudeScale        = args.convinitmag

    Globs.WeightNoiseUseWeightScaling   = args.weightnoiseweightscaling
    Globs.WeightNoiseInjectionScale     = args.weightnoise
    Globs.WeightNoisePerWeight          = args.weightnoiseperweight
    Globs.WeightNoiseRMSPerFilter       = args.weightnoisermsperfilter
    Globs.ScalingInitMagnitudeScale     = args.scalinginitmag

    if args.scalinginitmag is None :
        Globs.ScalingInitLow            = args.scalinginitlow
        Globs.ScalingInitHigh           = args.scalinginithigh
    else :
        # scalinginitmag overrides the low high method
        Globs.ScalingInitLow            = args.scalinginitmag
        Globs.ScalingInitHigh           = args.scalinginitmag

    Globs.Conv1x1InitMagnitudeScale     = args.resprojinitmag
    Globs.ResProjectionUnityMagReset    = args.resprojunitymag
    Globs.ConvUnityMagReset             = args.convunitymag
    Globs.FlipNegFilterAndScalingSign   = args.flipnegscaling
    Globs.Milestones                    = args.milestones
    Globs.AnnealL2Decay                 = args.annealdecay
    Globs.UseConvScaling                = args.convscaling
    Globs.UseGradMaxNorm                = args.gradmaxnorm
    Globs.TreluInitMag                  = args.treluinitmag
    Globs.HasSkipScaling                = args.skipscaling

    # default nonlinearity layer
    Globs.NonLinLay                     = ActFunc.NonLinType.tolayer (  ActFunc.NonLinType.fromstring(args.nl) )
    
    # activation euclidean norm layer as used by Filter Response Normalization
    Globs.HasActivationEuclidNorm       = args.actnorm

    # training is based on min abs grads of current and previous minibatch
    Globs.DualMinibatch                 = args.dualminibatch

def FlipFiltersWithNegScaling ( ConvBN ) :
#    for (filter,scaling) in zip ( ConvWeightParams, RawScalingParams ) :
    for ThisConvBN in ConvBN :
        FilterData = ThisConvBN[0].weight.data
        FilterShape = FilterData.shape
        ScalingData = ThisConvBN[1].weight.data
        # per filter switch to create a +/-1 broadcastable scaling of the filters
        SignSwitch = ScalingData.gt(0.0)
        SignSwitchBroadcast = SignSwitch.reshape(ScalingData.shape[0],1,1,1)
        SignSwitchRepeat = SignSwitchBroadcast.repeat (1,FilterShape[1],FilterShape[2],FilterShape[3])
        ScalingData[...] = torch.where ( SignSwitch, ScalingData, -ScalingData ).to(Globs.Device)
        FilterData[...] = torch.where ( SignSwitchRepeat, FilterData, -FilterData ).to(Globs.Device)

def WriteToTB ( StreamName, Value ) :
    
    Iter = int(Globs.Iter[StreamName])
    Value = float(Value)
    StreamName = str(StreamName)
    Globs.Log.TB.add_scalar ( StreamName, Value, Iter )
    Globs.Iter[StreamName] += 1
    #sleep(0.001)

def WriteToTBX () :
    for i in range(100) :
        Globs.Log.TB.add_scalar ( 'ValLoss', np.random.random(), i )
        sleep(0.001)

def MakeWorkAroundIterCountersDict() :
    # create (workaround) dictionary for iteration counters for each scalar stream for TensorBoard
    Globs.Iter = AttributeDict()
    Globs.Iter['TrainStepLoss'] = 0
    Globs.Iter['TrainStepPrec'] = 0
    Globs.Iter['TrainProgMeanLoss'] = 0
    Globs.Iter['TrainProgMeanPrec'] = 0
    Globs.Iter['TrainProgMaxLoss'] = 0
    Globs.Iter['TrainProgMaxPrec'] = 0
    Globs.Iter['TrainProgMinLoss'] = 0
    Globs.Iter['TrainProgMinPrec'] = 0
    Globs.Iter['ValLoss'] = 0
    Globs.Iter['ValPrec'] = 0

def TestTensorBoard() :
    if False :
        writer = Globs.Log.TB
        for n_iter in range(100):
            writer.add_scalar('Data1', np.random.random(), n_iter)
            sleep(0.001)
            writer.add_scalar('Data2', np.random.random(), n_iter)
            sleep(0.001)
            writer.add_scalar('Data3', np.random.random(), n_iter)
            sleep(0.001)
            writer.add_scalar('Data4', np.random.random(), n_iter)
            sleep(0.001)

        Globs.Log.Close()

    elif True :
        for n_iter in range(100):
            WriteToTB ('Data1', np.random.random(), n_iter)
            WriteToTB('Data2', np.random.random(), n_iter)
            WriteToTB('Data3', np.random.random(), n_iter)
            WriteToTB('Data4', np.random.random(), n_iter)

        Globs.Log.Close()

    else :
        # THIS CODE CREATES an event file but TB claims there is no scalar data in it. Crazy as the code is identical to above but just wrapped in a class call.
        Globs.Log.AddStream ('Loss/train')
        Globs.Log.AddStream ('Accuracy/train')
        Globs.Log.AddStream ('Loss/test')
        Globs.Log.AddStream ('Accuracy/test')
        for n_iter in range(100):
            Globs.Log.AddToLogX('Loss/train', np.random.random(), n_iter)
            Globs.Log.AddToLogX('Loss/test', np.random.random(), n_iter)
            Globs.Log.AddToLogX('Accuracy/train', np.random.random(), n_iter)
            Globs.Log.AddToLogX('Accuracy/test', np.random.random(), n_iter)

        Globs.Log.Close()

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    # - note that in validate mode the minibatch stats can also be used since the minibatch is 128 so replicating the situation of the training model and allowing for more realistic validation accuracy
    model.eval()

    NumSamplesInEpoch = len(val_loader.dataset.data)
    MinibatchSize = val_loader.batch_size
    NumMinibatchesInDataset = NumSamplesInEpoch//MinibatchSize
    StatsGroupProgressBar = ConsoleProgressBar ( NumMinibatchesInDataset, InfoStr="VALIDATING" )

    end = time.time()
    with torch.no_grad() :
        for i, (input, target) in enumerate(val_loader):
            target = target.to(Globs.Device,non_blocking=True)

            input_var = torch.autograd.Variable(input).to(Globs.Device)
            target_var = torch.autograd.Variable(target)    #.to(Globs.Device)

            if Globs.Half :
                input_var = input_var.half()

            # compute output
            output = model(input_var)[-1].Out
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            MinibatchSize =  input.size()[0]

            # measure accuracy and record loss
            prec1 = ComputePrecision ( output.data, target )
            losses.update ( loss.data, MinibatchSize )
            top1.update ( prec1, MinibatchSize )

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # output progress and on last step remove
            StatsGroupProgressBar.Update()
            if StatsGroupProgressBar.Complete :
                # remove and reset the progress
                StatsGroupProgressBar.Hide()

            if i % args.print_freq == 0 and False :
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

            # clean up CUDA mem
            del input_var
            del target_var
            del output
            del loss

            torch.cuda.empty_cache()

    # clean up CUDA mem
    torch.cuda.empty_cache()

    return ( top1.avg, losses.avg )

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class TrainingStats (object) :

    TrainAccIdx = 0
    TrainErrMin = 0.0
    TrainErrMax = 0.0
    TrainLossMin = 0.0
    TrainLossMax = 0.0
    TrainErrAcc = 0.0
    TrainLossAcc = 0.0

    def __init__ ( self, AuxOutputsDesc=[] ) :

        # create an array of floats to hold the auxiliary outputs
        self.AuxOutputsAcc = [0.0] * len(AuxOutputsDesc)
        self.AuxOutputsDesc = AuxOutputsDesc

    def Reset ( self ) :
        self.TrainAccIdx = 0
        self.TrainErrMin = 0.0
        self.TrainErrMax = 0.0
        self.TrainLossMin = 0.0
        self.TrainLossMax = 0.0
        self.TrainErrAcc = 0.0
        self.TrainLossAcc = 0.0

        # aux outputs
        for idx in range(len(self.AuxOutputsAcc)) :
            self.AuxOutputsAcc[idx] = 0.0

    def Update ( self, Loss, Error, AuxOutputs=[] ) :

        # convert params to standard python otherwise they will cause var allocation problems due to incompatibility and lack of type checking in Python
        Loss = float (Loss)
        Error = float (Error)

        if ( self.TrainAccIdx == 0 ) :
            self.TrainErrMin = Error
            self.TrainErrMax = Error
            self.TrainLossMin = Loss
            self.TrainLossMax = Loss
            self.TrainErrAcc = Error
            self.TrainLossAcc = Loss

            # aux outputs
            for idx in range(len(self.AuxOutputsAcc)) :
                self.AuxOutputsAcc[idx] = AuxOutputs[idx]

        else :
            if Error < self.TrainErrMin: self.TrainErrMin = Error
            if Error > self.TrainErrMax: self.TrainErrMax = Error
            if Loss < self.TrainLossMin: self.TrainLossMin = Loss
            if Loss > self.TrainLossMax: self.TrainLossMax = Loss

            self.TrainErrAcc += Error
            self.TrainLossAcc += Loss

            # aux outputs
            for idx in range(len(self.AuxOutputsAcc)) :
                self.AuxOutputsAcc[idx] += AuxOutputs[idx]

        # increment the sample index
        self.TrainAccIdx += 1

    @property
    def MeanErr(self) :
        return self.TrainErrAcc / float(self.TrainAccIdx)
    @property
    def MeanLoss(self) :
        return self.TrainLossAcc / float(self.TrainAccIdx)

class ConsoleProgressBar () :
    """
    Simple class that displays a console progress over a specified max starting at zero and updated with a given step in the Update() method
    """
    _MaxVal = 0
    _ProgressCount = 0

    def __init__ (self, MaxVal, InitProgress=0, InfoStr="" ) :
        self._MaxVal = MaxVal
        self._ProgressCount = InitProgress
        self._InfoStr = InfoStr
        self.Visible = False
        #print( "\n" + self._InfoStr + " : Progress: {:>3} %".format( self._ProgressCount * 100 / self._MaxVal ), end='\r')

    def Update ( self, Step=1 ) :
        self._ProgressCount += Step
        #print ( self._InfoStr + " : Progress [{:>4}] : {:>3} %".format( self._ProgressCount, self._ProgressCount * 100 / self._MaxVal ) + " : " + str(self._MaxVal), end='\r' )
        self.Show()
        #print ( self._InfoStr + " : Progress: {:>3} %".format( self._ProgressCount * 100 / self._MaxVal ) + " : " + str(self._MaxVal), end='\r' )
        #sys.stdout.flush()

    def Remove ( self ) :
        #sys.stdout.write("\r                                                                                                               \r")
        print ( " "*60, end='\r' )
        #sys.stdout.flush()
        self.Visible = False

    def Hide ( self ) :
        self.Remove()

    def Show ( self ) :
        print ( self._InfoStr + " : Progress [{:>4}] : {:>3} %".format( self._ProgressCount, self._ProgressCount * 100 // self._MaxVal ) + " : " + str(self._MaxVal) + " "*5, end='\r' )
        self.Visible = True

    def Reset ( self ) :
        # idea is that the bar is cleared and the count cleared
        self.Remove()
        self._ProgressCount=0

    @property
    def Complete ( self ) :
        return self._ProgressCount >= self._MaxVal

def OutputTrainingStatsGroup ( TrainingStatsGroup, MinibatchIdx, TotalTrainingSteps, TotalSamplesTrained, EpochIdx ) :
    """ Ouput formatted results of loss and error to console """
    TS = TrainingStatsGroup

    PopSize = float(TS.TrainAccIdx)

    # format string for fixed part of the output
    #fs = "[%4d,%7d,%8d] : loss[%3.3f,%3.3f,%3.3f], prec[%3.3f,%3.3f,%3.3f]"
    fs = "[{:3d},{:6d},{:10d}] : loss[{:6.3f},{:6.3f},{:6.3f}, prec[{:7.3f},{:7.3f},{:7.3f}]"
    #fsouts = ( MinibatchIdx+1, TotalTrainingSteps+1, TotalSamplesTrained, TS.TrainLossAcc/TS.TrainAccIdx, TS.TrainLossMin, TS.TrainLossMax, TS.TrainErrAcc/TS.TrainAccIdx, TS.TrainErrMin, TS.TrainErrMax )

    # print the outputs
    print ( fs.format ( MinibatchIdx+1, TotalTrainingSteps+1, TotalSamplesTrained, TS.TrainLossAcc/TS.TrainAccIdx, TS.TrainLossMin, TS.TrainLossMax, TS.TrainErrAcc/TS.TrainAccIdx, TS.TrainErrMin, TS.TrainErrMax ) )

def ComputePrecision(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    # if only the top1 result is requested then return as a scalar
    if len(topk)==1 :
        return res[0].item()
        #return np.asscalar( res[0] )
    else :
        return res

def OuputValidationToConsole ( Loss, Prec, BestPrec, ElapsedTime, Epoch, Iteration, MinibatchIdx, NumMinibatches, MinibatchSize, TrainingPrecision=None ) :
        """ Output validation loss and error to console"""

        if TrainingPrecision is None :
            print('epoch %i, test acc %4.3f[%4.3f], val loss %4.3f, took %3.2f min, %4.1f ms per minbatch, and %4.3f ms per sample' % (
                                    Epoch+1, Prec, BestPrec, Loss,
                                    ElapsedTime / 60.0,
                                    ElapsedTime * 1000.0 / NumMinibatches,
                                    ElapsedTime * 1000.0 / MinibatchSize / NumMinibatches ) )
        else :
            print('epoch %i, test acc %4.3f[%4.3f], val loss %4.3f, gap[%3.2f], epoch[%3.2f], minbatch[%4.1f], sample[%4.3f]' % (
                                    Epoch+1, Prec, BestPrec, Loss,
                                    TrainingPrecision-Prec,
                                    ElapsedTime / 60.0,
                                    ElapsedTime * 1000.0 / NumMinibatches,
                                    ElapsedTime * 1000.0 / MinibatchSize / NumMinibatches )
                 )


        if Prec > BestPrec :
            print ('    -> new best validation Prec!')

def OuputValidationToLogFile ( ValLogFile, Loss, Prec, BestPrec, ElapsedTime, Epoch, Iteration, MinibatchIdx, NumMinibatches, MinibatchSize ) :
        """ Log validation loss and Prec file """

        # File output - has optional class IoU and mean
        FileOutStr = str(Epoch) + " " + str(Iteration) + " " + str(Loss) + " " + str(Prec)

        ValLogFile.write ( FileOutStr + "\n")
        ValLogFile.flush()

def GetConvParams ( Params ) :
    """ return list of parameters that have 4 axes as a guess that these are convolutional """
    ConvParams = [p for p in Params if len(p.shape)==4 and p.shape[2:]!=[1,1] ]
    ConvParamsShapes = [p.shape for p in ConvParams if len(p.shape)==4 and p.shape[2:]!=[1,1] ]
    return ConvParams

def LearningRateAdjustGradients ( NamedParamList, GradLRDict ) :

    for p in NamedParamList :
        ParamName = p[0]
        Param = p[1]
        if ParamName in GradLRDict :
            Param.grad.data *= GradLRDict[ParamName]

def ClampScalingToLowerBound ( Params, LowerBound ) :

    for p in Params :
        # lower bound tensor of same shape as Params
        LowerBoundLikeP = torch.ones( p.shape, dtype=Config.DefaultType ).to(Globs.Device) * LowerBound
        # clamp the param data directly (i.e. no grads) where the values are < LowerBound
        p.data[...] = torch.where ( p.data.gt(LowerBound), p.data, LowerBoundLikeP)

def ScaleConvGradientsByMag ( ConvParams, SquareLaw=False ) :
    """ scale the gradient of each conv filter by mag(filter) """

    for p in ConvParams :
        ParamName = p[0]
        Param = p[1]
        KernelMag2 = Param.data.pow(2).sum(dim=(1,2,3))
        if SquareLaw :
            Param.grad.data *= KernelMag2.view ( (Param.shape[0],1,1,1) )
        else :
            Param.grad.data *= KernelMag2.sqrt().view ( (Param.shape[0],1,1,1) )

def FilterEuclidNormXferToScaling ( WeightsParams, ScalingParams, ExpoScaling=False ) :
    LayerCount = 0
    for (w,s) in zip(WeightsParams, ScalingParams) :
        w_name=w[0]
        s_name=s[0]
        w_val=w[1].data
        s_val=s[1].data

        # compute Euclidean magnitude of w per filter
        LayerShape = w[1].shape

        if len(LayerShape)==4 :
            w_norm = w_val.pow(2).sum(dim=(1,2,3)).sqrt()
            w_val /= w_norm.view(w_val.shape[0],1,1,1)
        elif len(LayerShape)==2 :
            w_norm = w_val.pow(2).sum(dim=(1,)).sqrt()
            w_val /= w_norm.view(w_val.shape[0],1)
        else :
            assert False

        if ExpoScaling :
            s_val += w_norm.log()
        else :
            s_val *= w_norm

        if False and LayerCount==0 :
            ValList = w_norm.cpu().numpy().tolist()
            ListStr = ""
            for val in ValList :
                ListStr += " {:2.3f}".format( val )
            print(ListStr)

        LayerCount += 1

def ZeroMeanNormaliseGrads ( Params, PerChannel=False, MeanMult=0.99 ) :
        """
        TODO: conv1x1 and linear has cross channel zmg
        """

        if PerChannel :
            MeanDims = (2,3)
        else:
            MeanDims = (1,2,3)

        # only gradient normalise conv weight params if they have (h,w) spatial dimensions
        for p in Params :
            ThisParam = p[1]
            if len(ThisParam.shape)==4 :
                if ThisParam.shape[2:]!=[1,1] :
                    ParamMeanGrad = torch.mean ( ThisParam.grad, dim=MeanDims, keepdim=True )
                    # CLEAR VERSION
                    # ThisParam.grad = ThisParam.grad + ( ThisParam.grad - ParamMeanGrad ) * MeanMult
                    # EFFICIENTLY REARRANGED VERSION
                    ThisParam.grad.data = ThisParam.grad.data * ( 1 + MeanMult ) - ParamMeanGrad * MeanMult
            else :
                debug11 = 0

def ZeroMeanNormaliseGrads_ORIG ( Params, PerChannel=False, MeanMult=0.99 ) :
        """
        TODO: conv1x1 and linear has cross channel zmg
        """

        if PerChannel :
            MeanDims = (2,3)
        else:
            MeanDims = (1,2,3)

        # only gradient normalise conv weight params if they have (h,w) spatial dimensions
        for p in Params :
            ThisParam = p[1]
            if len(ThisParam.shape)==4 :
                if ThisParam.shape[2:]!=[1,1] :
                    ParamMeanGrad = torch.mean ( ThisParam.grad.data, dim=MeanDims, keepdim=True )
                    ThisParam.grad.data -= ParamMeanGrad * MeanMult
            else :
                debug11 = 0


def ComputeGradNoise ( Params, Scale=1.0, UseWeightScaling=False ) :
        """
        Injects channel-wise zero mean noise modulated by channel-wise rms gradients        
        Notes :

        To try:
        - estimate gradient scale on entire filter rather than 2D slice : this is gradient gated at the filter level rather than at the per input map level.
            The forgettery is stronger vs per input map and gated only by the backprop error and state of the nonlinearity fwd gate. The question is whether filter level gradient flow
            is relevant to a given input map and its corresponing 2D kernel slice. Any gradient flow in the filter will cause 2D slices to randomize if they have no gradient flow of their own.
       
        """
        GradNoiseDict = {}

        for p in Params :
            Param = p[1]
            ParamData = Param.data
            ParamShape = ParamData.shape

            # draw channel-wise so per filter per channel i.e. omitting the spatial dimensions so shape (filter,depth)
            FilterDraw = torch.empty( ParamShape, dtype=Config.DefaultType).uniform_ ( -Scale, Scale ).to(Config.CurrentDevice)
            # subtract mean per channel
            SpatialDims = (2,3)
            FilterDrawCZM = FilterDraw - FilterDraw.mean ( dim=SpatialDims, keepdim=True)

            # compute channel-wise rms gradient
            ChannelDims = (2,3)
            GradScale = Param.grad.data.pow(2).mean(dim=ChannelDims, keepdim=True).sqrt()

            # optionally scale by weights which has the effect of reducing the noise for smaller weights so that they are less disturbed proportionally to their size
            # note - multiplication order matters here and the noise broadcast to spatial dims is modulated by the weights to acquire the same shape as the parameter
            if UseWeightScaling :
                GradScale = ParamData.abs() * GradScale

            # scale the noise draw
            ScaledDraw = FilterDraw * GradScale

            # mean subtract so CZMN condition is re-established after the weighting
            ScaledDrawCZM = ScaledDraw - ScaledDraw.mean ( dim=SpatialDims, keepdim=True)

            GradNoiseDict[Param] = ScaledDrawCZM

        return GradNoiseDict

def WeightNoiseInjection ( Params, Scale=1.0, UseWeightScaling=False ) :
        """
        Injects channel-wise zero mean noise modulated by channel-wise rms gradients        
        Notes :
       
        """
        for p in Params :
            Param = p[1]
            ParamData = Param.data
            ParamShape = ParamData.shape

            # draw channel-wise so per filter per channel i.e. omitting the spatial dimensions so shape (filter,depth)
            FilterDraw = torch.empty( ParamShape, dtype=Config.DefaultType).uniform_ ( -Scale, Scale ).to(Config.CurrentDevice)
            # subtract mean per channel
            SpatialDims = (2,3)
            FilterDrawCZM = FilterDraw - FilterDraw.mean ( dim=SpatialDims, keepdim=True)

            # compute channel-wise rms gradient
            ChannelDims = (2,3)
            FilterwiseGradScale = Param.grad.data.pow(2).mean(dim=ChannelDims, keepdim=True).sqrt()

            # optionally scale by weights which has the effect of reducing the noise for smaller weights so that they are less disturbed proportionally to their size
            # note - multiplication order matters here and the noise broadcast to spatial dims is modulated by the weights to acquire the same shape as the parameter
            if UseWeightScaling :
                FilterwiseGradScale = ParamData.abs() * FilterwiseGradScale
                # mean subtract so CZMN condition is re-established after the weighting
                FilterwiseGradScale -= FilterwiseGradScale.mean ( dim=SpatialDims, keepdim=True)

            # add scaled noise to gradients
            Param.grad.data += FilterDrawCZM * FilterwiseGradScale

        pass

def WeightNoiseInjection_Naive ( Params, Scale=1.0, PerWeight=True, RMSPerFilter=False ) :
        """
        Injects noise into each filter according to the passed Scale and to the rms size of the gradients of each filter. The noise may be uniform (sigma=0) or Gaussian if sigma>0.
        The noise may be per weight or per filter in which case the noise shifts the entire filter systematically.
        
        Notes :

        To Try :
        - broadcast same weight-wise draw to each filter in layer : this emulates the operation of map-wise noise scaling in the previous layer (or resblock) as each noise scaled map has the same noise
            for the corresponding filter in the next layer. Hence this is a sysmtematic effect on gradients corresponding to the wieghts that connect to that map. Such a systematic effect may have more
            benefit since the map then has a sysmtematically adjusted set of corresponding downstream paths. Also by injecting noise into the weights then the activations going thru the residual skup path
            are unaffected and so do not change the ouput of the net, so likely more noise may be injected this way.
        - modify the weights directly according to the current LR (get from the optimizer) to avoid momentum smoothing out the weight noise
        - scale noise by weight size so that smaller weights get smaller noise and so as most weights are small then the majority of weights receive little noise allowing larger weights to be given more noise
        - inject noise once every k steps rather than per step
        - inject noise per filter or weight with random frequency using a binary mask (Bernoulli distribution)
        - Gaussian noise with various sigma : large outliers persist thru momentum to have a systematic effect that uniform distribution lacks, hence likely more effective
       
        """

        # for each parameter :
        #   [1] compute the rms gradient for the each filter in the layer to estimate the gradient scale that will modulate the noise signal
        #   [2] draw a random uniform distribution in the range +- scale, U
        #   [3] modulate the distribution filter-wise by the gradient scale computed in [1] i.e. U*GradSize
        #   [4] modulate the gradients for each filter i by grad_i += U*GradSize_i
        for p in Params :
            Param = p[1]
            ParamData = Param.data
            ParamShape = ParamData.shape

            if PerWeight :
                FilterDraw = torch.empty( ParamShape, dtype=Config.DefaultType).uniform_ ( -Scale, Scale ).to(Config.CurrentDevice)
            else :
                FilterDraw = torch.empty( ParamShape[0], dtype=Config.DefaultType).uniform_ ( -Scale, Scale ).to(Config.CurrentDevice).view ( (ParamShape[0],)+(1,)*(len(ParamShape)-1) )

            if RMSPerFilter :
                KernelDims = tuple(range(1,len(ParamShape)))
                FilterwiseGradScale = Param.grad.data.pow(2).mean(dim=KernelDims, keepdim=True).sqrt()
            else :
                FilterwiseGradScale = Param.grad.data.pow(2).mean().sqrt()

            # add scaled noise to gradients
            Param.grad.data += FilterDraw * FilterwiseGradScale

        pass

def FilterwiseMaxNormGrads ( Params, eps=0.01, Filterwise=True, NormToMax=False ) :
        """
        Notes :
        - epsilon is the lower bound for weight size so any weight smaller than this is clipped to eps when computing its max norm
        """

        TensorEps = torch.tensor(eps).to(Globs.Device)
        if Globs.Half :
            TensorEps = TensorEps.half()

        # apply relative weight magnitude gradient normalization to all params - the max is across all fitlers and kernel dimensions so filters should have similar magnitudes using unity decay
        for p in Params :
            Param = p[1]
            ParamData = Param.data
            ParamShape = ParamData.shape
            if len(ParamShape)==4 :
                AbsParamData = ParamData.abs()

                if NormToMax :
                    if Filterwise :
                        # max over dim=(1,2,3)
                        MaxWeightMag = torch.max(AbsParamData,dim=1)[0]
                        MaxWeightMag = torch.max(MaxWeightMag,dim=1)[0]
                        MaxWeightMag = torch.max(MaxWeightMag,dim=1)[0]
                        MaxWeightMag = MaxWeightMag.reshape ( (-1,1,1,1) )
                    else :
                        # max over entire filterbank
                        MaxWeightMag = torch.max(AbsParamData)

                    MaxNormFactor = torch.max ( TensorEps, AbsParamData ) / torch.max ( TensorEps, MaxWeightMag )

                else :
                    MaxNormFactor = torch.max ( TensorEps, AbsParamData )

                # modify gradients
                Param.grad.data[...] = Param.grad.data * MaxNormFactor

def PreWarm ( train_loader, model, criterion, optimizer, PreWarmLR, NumEpochs ) :

    print("PRE-WARM before main training : epochs=" + str(NumEpochs) + " at LR=" +str(PreWarmLR) )

    # save the initial LRs for all param groups and set the lr to PreWarmLR
    #ParamGroupOrigLR = {}
    #for param_group in optimizer.param_groups:
        #ParamGroupOrigLR[param_group] = param_group['lr']
        #param_group['lr'] = PreWarmLR
    ParamGroupOrigLR = optimizer.param_groups[0]['lr']
    optimizer.param_groups[0]['lr'] = PreWarmLR

    NumSamplesInEpoch = len(train_loader.dataset.data)
    MinibatchSize = train_loader.batch_size
    NumStepsInEpoch = NumSamplesInEpoch//MinibatchSize

    # SETUP A PROGRESS BAR FOR SHOWING STATS AND PROGREES OF GROUPS OF MINIBATCHES
    TrainStatsGroupProgressBar = ConsoleProgressBar ( NumEpochs*NumStepsInEpoch, InfoStr="" )
    TrainStatsGroupProgressBar.Show()

    for epoch in range(NumEpochs) :
        # take a draw from the dataset from which NumSteps will be used to pre-warm
        MinibatchDraw = list ( enumerate(train_loader) )

        # pre-warm with NumSteps minibatches
        for MinibatchIdx, (input, target) in MinibatchDraw :

            target = target.to(Globs.Device,non_blocking=True)
            input_var = torch.autograd.Variable(input).to(Globs.Device)
            target_var = torch.autograd.Variable(target)
            if Globs.Half :
                input_var = input_var.half()

            # compute output
            forward_start = time.time()
            output = model(input_var)
            loss = criterion(output, target_var)
            forward_duration = time.time() - forward_start

            # compute gradient and do SGD step
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            backward_duration = time.time() - backward_start

            # DEBUG - print duration of fwd and bwd passes
            if True :
                print ( "fwd time=%4.3f    bwd time=%4.3f" % (forward_duration, backward_duration) )

            # optionally modify the gradients to be zero mean for convolutional params
            if Globs.ZeroMeanGrad>0.0 :
                ZeroMeanNormaliseGrads ( Globs.ConvWeightParams, PerChannel=Globs.PerChannel, MeanMult=Globs.ZeroMeanGrad )

            # perform parameter update step using parameter gradients
            optimizer.step()

            TrainStatsGroupProgressBar.Update()

    # restore the learning rate noting that 1 epoch has been lost from the optimiser
    optimizer.param_groups[0]['lr'] = ParamGroupOrigLR

def NormaliseConvGrad2KernelMag ( ConvParams, WeightScalingDict, IncludeScaling=True, SquaredMag=False, Invert=False ) :
    """
    Adapts the gradients of the weight params according to 1/(mag(W).scaling) . The scaling corresponding to each conv weight parameter is available in the WeightScalingDict
    """
    for p in ConvParams :
        ParamName = p[0]
        Param = p[1]

        if SquaredMag :
            KernelMag = Param.data.pow(2).sum(dim=(1,2,3))
        else :
            KernelMag = Param.data.pow(2).sum(dim=(1,2,3)).sqrt()

        if IncludeScaling :
            ThisTup = WeightScalingDict[ParamName]
            UseExpoScaling = ThisTup.UseExpoScaling
            WeightNameCheck = ThisTup.WeightName
            ScalingParam = ThisTup.Scaling
            WeightParam = ThisTup.Weight

            # xfer original filter weight to the scaling - the mask is 1 if not reset so no xfer and is ResetMag/PerFilterMag if reset so need to invert so scaling' = scaling * PerFilterMag/ResetMag
            if UseExpoScaling :
                ThisScalingData = ScalingParam.data.exp()
            else :
                ThisScalingData = ScalingParam.data

            Param.grad.data *= ( KernelMag / ( ThisScalingData + 0.0001) ) .view ( (Param.shape[0],1,1,1) )

        else :
            if Invert :
                Param.grad.data /= ( KernelMag + 0.001 ).view ( (Param.shape[0],1,1,1) )
            else :
                Param.grad.data *= ( KernelMag + 0.001 ).view ( (Param.shape[0],1,1,1) )

def NormaliseConvGrad2ELR ( ConvParams, WeightScalingDict, Adapt2Weight=True, Adapt2Scaling=True, WeightMaxNorm=False ) :
    """
    Adapts the gradients of the weight params according to abs(W)/scaling which undoes the ELR of the scaling gradient gain for each filter and the individual weight sizes for relative gradient.
    """
    for p in ConvParams :
        ParamName = p[0]
        Param = p[1]
        absdata = Param.data.abs()

        ThisTup = WeightScalingDict[ParamName]
        UseExpoScaling = ThisTup.UseExpoScaling
        WeightNameCheck = ThisTup.WeightName
        ScalingParam = ThisTup.Scaling
        WeightParam = ThisTup.Weight

        # xfer original filter weight to the scaling - the mask is 1 if not reset so no xfer and is ResetMag/PerFilterMag if reset so need to invert so scaling' = scaling * PerFilterMag/ResetMag
        if UseExpoScaling :
            ThisScalingData = ScalingParam.data.exp()
        else :
            ThisScalingData = ScalingParam.data

        WeightFact = torch.where ( absdata>0.005, absdata, torch.ones_like(absdata)*0.005 )
        ScalingFact = torch.where ( ThisScalingData>0.01, ThisScalingData, torch.ones_like(ThisScalingData)*0.01 )

        if WeightMaxNorm :
            PerFiltMax = absdata.max(dim=1,keepdim=True)[0].max(dim=2,keepdim=True)[0].max(dim=3,keepdim=True)[0]
            WeightFact /= PerFiltMax

        if Adapt2Weight :
            Param.grad.data *= WeightFact 
        if Adapt2Scaling :
            Param.grad.data /= ScalingFact.view ( (Param.shape[0],1,1,1) )

def TransferScalingtoWeights ( FilterBankList, ScalingList, UseExpoScaling=False, ScalingMag=1.0 ) :
    """
    
    """

    assert len(FilterBankList)==len(ScalingList)

    # for each filter bank
    for idx in range (len(FilterBankList)) :

        # get shortcut to the data tensor
        ThisFilterBankData = FilterBankList[idx][1].data
        ThisScaling = ScalingList[idx][1].data

        if UseExpoScaling :
            ThisScaling = ThisScaling.exp()

        assert ThisFilterBankData.shape[0]==ThisFilterBankData.shape[0]

        # compute scaling broadcasting
        NumDims = len(ThisFilterBankData.shape)
        if NumDims==1 :
            ViewShape=(-1,)
        elif NumDims==2 :
            ViewShape=(-1,1)
        elif NumDims==4 :
            ViewShape=(-1,1,1,1)
        else :
            assert False

        # transfer excess scaling to weights
        ThisFilterBankData[...] = ThisFilterBankData * ThisScaling.view ( ViewShape ) / ScalingMag

        # reset scaling to unity
        ScalingList[idx][1].data[...] = torch.ones_like ( ThisScaling ) * ScalingMag
        if UseExpoScaling :
            ScalingList[idx][1].data[...] = ScalingList[idx][1].data[...].log()

def ResetFilterToScaling ( FilterBankList, ScalingList, UseExpoScaling=False ) :
    """
    Resets the magnitude of each filter to its corresponding scaling in ScalingList so these list must be compatible in length and depth.
    """

    assert len(FilterBankList)==len(ScalingList)

    # for each filter bank
    for idx in range (len(FilterBankList)) :

        # get shortcut to the data tensor
        ThisFilterBankData = FilterBankList[idx][1].data
        ThisScaling = ScalingList[idx][1].data

        if UseExpoScaling :
            ThisScaling = ThisScaling.exp()

        assert ThisFilterBankData.shape[0]==ThisFilterBankData.shape[0]

        NumDims = len(ThisFilterBankData.shape)

        if NumDims==1 :
            # magnitude over 1st and only dimension
            MagDims = (0,)
            ViewShape=(-1,)
        elif NumDims==2 :
            # magnitude over 2nd dimension only
            MagDims = (1,)
            ViewShape=(-1,1)
        elif NumDims==4 :
            # magnitude over all trailing dims
            MagDims = (1,) + tuple(range(2,len(ThisFilterBankData.shape)))
            ViewShape=(-1,1,1,1)
        else :
            assert False

        # compute the magnitude for each filter in the filterbank
        if NumDims ==1 :
            # scalar per filter so already a magnitude
            PerFilterMag = ThisFilterBankData
        else :
            PerFilterMag = ThisFilterBankData.pow(2).sum(dim=MagDims, keepdim=True).sqrt()

        # per filter magnitude
        Normalizer = ThisScaling.view(ViewShape) / PerFilterMag

        ThisFilterBankData[...] = ThisFilterBankData * Normalizer

def ResetFilterMagnitude ( FilterBankList, NormMode='unity', Scale=1.0, MagDims=None ) :
    """
    Resets any filter whose Euclidean magnitude is <=Thresh to unit magnitude or to the thresh if ResetThreshIsClamp with optional randomise if reset to unity.
    In the case that reset is to unity and not randomise then if a corresponing scaling parameter is available then the magnitude of the filter is transferred to the scaling.
    """

    # for each filter bank
    for idx in range (len(FilterBankList)) :

        # get shortcut to the data tensor
        ThisFilterBankData = FilterBankList[idx][1].data

        NumDims = len(ThisFilterBankData.shape)

        if MagDims is None :
            if NumDims==1 :
                # magnitude over 1st and only dimension
                MagDims = (0,)
            elif NumDims==2 :
                # magnitude over 2nd dimension only
                MagDims = (1,)
            else :
                # magnitude over all trailing dims
                MagDims = (1,) + tuple(range(2,len(ThisFilterBankData.shape)))

        # compute the magnitude for each filter in the filterbank
        if NumDims ==1 :
            # scalar per filter so already a magnitude
            PerFilterMag = ThisFilterBankData
        else :
            PerFilterMag = ThisFilterBankData.pow(2).sum(dim=MagDims, keepdim=True).sqrt()

        # normalizer is optionally the max magnitude
        if NormMode=='max' :
            # max magnitude broadcast to filter dim
            Normalizer = PerFilterMag.max ( dim=0, keepdim=True )[0]
        elif NormMode=='mean' :
            Normalizer = PerFilterMag.mean ( dim=0, keepdim=True )[0]
        elif NormMode=='unity' :
            # per filter magnitude
            Normalizer = PerFilterMag

        ThisFilterBankData[...] = ThisFilterBankData * Scale / Normalizer

def ResetLowMagitudeFilters ( FilterBankList, Thresh, ResetMag=1.0, Randomise=False, ResetThreshIsClamp=False, WeightScalingDict={}, TransferMagToScaling=True ) :
    """
    Resets any filter whose Euclidean magnitude is <=Thresh to unit magnitude or to the thresh if ResetThreshIsClamp with optional randomise if reset to unity.
    In the case that reset is to unity and not randomise then if a corresponing scaling parameter is available then the magnitude of the filter is transferred to the scaling.
    """

    # for each filter bank
    for idx in range (len(FilterBankList)) :
        # get shortcut to the data tensor
        ThisFilterBankData = FilterBankList[idx][1].data

        # compute the magnitude for each filter in the filterbank
        PerFilterMag = ThisFilterBankData.pow(2).sum(dim=(1,2,3)).sqrt()

        # optional reset threshold clamp
        if ResetThreshIsClamp :
            ResetMag = Thresh

        # create binary mask for filters that need to be reset
        if not Randomise :

            OnesMask = torch.ones ( ThisFilterBankData.shape[0], dtype=Config.DefaultType).to(Globs.Device,non_blocking=True)
            # mask set the scaling mask to 1/magnitude
            ResetMask = torch.where ( PerFilterMag.gt(Thresh), OnesMask, ResetMag / PerFilterMag )
            # scale the per filter kernels according to the mask
            ResetMaskView = ResetMask.view ( ( ResetMask.shape[0], 1, 1, 1) )
            ThisFilterBankData *= ResetMaskView

            # if reset to unity then if a scaling is available then xfer the magnitude to the scaling for the reset filters
            if not ResetThreshIsClamp and TransferMagToScaling :
                FilterBankName = FilterBankList[idx][0]
                if FilterBankName in WeightScalingDict :
                    UseExpoScaling = WeightScalingDict[FilterBankName].UseExpoScaling
                    WeightNameCheck = WeightScalingDict[FilterBankName].WeightName
                    ScalingParam = WeightScalingDict[FilterBankName].Scaling
                    WeightParam = WeightScalingDict[FilterBankName].Weight
                    # xfer original filter weight to the scaling - the mask is 1 if not reset so no xfer and is ResetMag/PerFilterMag if reset so need to invert so scaling' = scaling * PerFilterMag/ResetMag
                    if UseExpoScaling :
                        ScalingParam.data -= ResetMask.log()
                    else :
                        ScalingParam.data /= ResetMask

        else :
            FilterBankShape = ThisFilterBankData.shape
            # create uniform distribution filter weights

            RandUniformWeights = np.random.uniform ( -1.0, 1.0, size=FilterBankShape ).astype('float32')

            # zero mean shift each channel in each filter for common mode signal rejection aka model mean shift
            # TEST THIS
            #RandUniformWeights -= np.mean ( RandUniformWeights, axis=(2,3), keepdims=True )

            # normalise the Euclidean magniude of each filter to unity
            RandWeightsMag = np.sqrt(np.square(RandUniformWeights).sum(axis=(1,2,3)))
            UnitMagRandUniformWeights = RandUniformWeights / ( RandWeightsMag.reshape (RandWeightsMag.shape[0],1,1,1) + 0.000001 )

            GtSwitch = PerFilterMag.gt(Thresh).reshape (RandWeightsMag.shape[0],1,1,1)
            GtSwitch = GtSwitch.repeat (1,FilterBankShape[1],FilterBankShape[2],FilterBankShape[3])
            UnitMagRandUniformWeightsGPU = torch.from_numpy(UnitMagRandUniformWeights).to(Globs.Device,non_blocking=True)

            # create reset mask of sime size as weights tensor with 
            ThisFilterBankData[...] = torch.where ( GtSwitch, ThisFilterBankData, UnitMagRandUniformWeightsGPU )

def ResetLowMagitudeFiltersConvBN ( FilterBankList, Thresh, ResetMag=1.0, Randomise=False, ResetThreshIsClamp=False ) :

    # for each filter bank
    for idx in range (len(FilterBankList)) :
        # get shortcut to the data tensor
        ThisFilterBankData = FilterBankList[idx][0].weight.data
        ThisScalingData = FilterBankList[idx][1].weight.data
        FilterBankShape = ThisFilterBankData.shape

        # compute the magnitude for each filter in the filterbank
        PerFilterMag = ThisFilterBankData.pow(2).sum(dim=(1,2,3)).sqrt()

        # optional reset threshold clamp
        if ResetThreshIsClamp :
            ResetMag = Thresh

        # create binary mask for filters that need to be reset
        if not Randomise :
            OnesMask = torch.ones ( ThisFilterBankData.shape[0], dtype=Config.DefaultType).to(Globs.Device,non_blocking=True)
            # mask set the scaling mask to 1/magnitude
            ResetMask = torch.where ( PerFilterMag.gt(Thresh), OnesMask, ResetMag / PerFilterMag )

            # scale the per filter kernels according to the mask
            ThisFilterBankData *= ResetMask.view ( ( ResetMask.shape[0], 1, 1, 1) )

        else :
            # create uniform distribution filter weights
            RandUniformWeights = np.random.uniform ( -1.0, 1.0, size=FilterBankShape ).astype('float32')

            # zero mean shift each channel in each filter for common mode signal rejection aka model mean shift
            # TEST THIS
            #RandUniformWeights -= np.mean ( RandUniformWeights, axis=(2,3), keepdims=True )

            # normalise the Euclidean magniude of each filter to unity
            RandWeightsMag = np.sqrt(np.square(RandUniformWeights).sum(axis=(1,2,3)))
            UnitMagRandUniformWeights = RandUniformWeights / ( RandWeightsMag.reshape (RandWeightsMag.shape[0],1,1,1) + 0.000001 )

            GtSwitch = PerFilterMag.gt(Thresh)
            GtSwitchRepeat = GtSwitch.reshape(RandWeightsMag.shape[0],1,1,1).repeat(1,FilterBankShape[1],FilterBankShape[2],FilterBankShape[3])
            UnitMagRandUniformWeightsGPU = torch.from_numpy(UnitMagRandUniformWeights).to(Globs.Device,non_blocking=True)
            # create reset mask of sime size as weights tensor with 
            ThisFilterBankData[...] = torch.where ( GtSwitchRepeat, ThisFilterBankData, UnitMagRandUniformWeightsGPU )

            # scalings also need to be reset to +unity using the same reset mask
            OnesLikeScaling = torch.ones ( ThisScalingData.shape, dtype=Config.DefaultType ).to(Globs.Device,non_blocking=True)
            ThisScalingData[...] = torch.where ( GtSwitch, ThisScalingData, OnesLikeScaling * ResetMag)

def AddAllAuxiliaryLosses ( loss ) :

    ############################
    # L1 weight decay applied only to conv kernels
    ############################
    if Globs.L1Decay > 0 :
        # L1WeightLoss = None
        # if not Globs.Conv1UnityDecayLoss :
        Conv1Weights = Globs.Conv1WeightParams[0][1]
        L1WeightLoss = Conv1Weights.abs().sum()

        for weight in Globs.ResConvWeightParams :
            if L1WeightLoss is None :
                L1WeightLoss = weight[1].abs().sum()
            else :
                L1WeightLoss += weight[1].abs().sum()

        loss += L1WeightLoss * Globs.L1Decay

    ############################
    # informatation difference from mean loss - all points in each block output are regularized to have the same embedding vector magnitude i.e. common pointwise information size
    ############################
    if Globs.L2InformationLossScaling > 0.0 :

        AllBlockReluActivations = GetActivationsByTags ( AllOutputs, Tags=['nl'])
            
        # build computational graph sum_over_all_maps(per_map_mean(abs(activation)))
        L2InformationLoss = None
        L2MeanActMagLoss = None
        for act in AllBlockReluActivations :
            # pointwise information Euclidean magnitude
            PerPointMag2 = act.pow(2).sum ( dim=1, keepdims=True )
            PerPointMag = PerPointMag2.sqrt()

            # per sample mean which is used as the mean reference for the information magnitude difference
            MeanMag = PerPointMag.mean ( dim=(2,3), keepdims=True )

            # loss is the mean over the minibatch of the per sample squared mean diff over spatial locations
            ThisLoss = ( PerPointMag - MeanMag ).pow(2).mean()

            if L2InformationLoss is None :
                L2InformationLoss = ThisLoss
            else :
                L2InformationLoss += ThisLoss

        # add weighted ActLoss to the total loss
        loss += L2InformationLoss * Globs.L2InformationLossScaling

    if Globs.L2MeanActMagLossScaling > 0.0 :

        EncOut = GetActivationsByTags ( AllOutputs, Tags=['enc_out'])[0]
        PerPointMag2 = EncOut.pow(2).sum ( dim=1, keepdim=True )
        PerPointMag = PerPointMag2.sqrt()
        MeanMag = PerPointMag.mean()
        EncOutMeanMagL2Loss = MeanMag.pow(2)

        loss += EncOutMeanMagL2Loss * Globs.L2MeanActMagLossScaling

    ############################
    # L2 -ve activation loss - this encourages filters to have +ve response in the absence of a non-linearity [experimental] and also to avoid filters going too -ve which could impact training adversely
    ############################
    if Globs.L2NegActLossScaling > 0.0 :
        AllConvActivations = GetActivationsByTags ( AllOutputs, Tags=['conv'] )
            
        # build computational graph sum_over_all_maps(per_map_mean(abs(activation)))
        L2NegActLoss = None
        for act in AllConvActivations :
            # - per layer sum over minibatch and filters of per filter mean of abs activation
            # - act loss masked to -ve activations ignoring all zero or -ve activations
            # - negates the activation and applies relu as the min and max functions take only a single dimension
            # TOTRY: mask to -ve only points for normalisation i.e. sum(sqr(pos points))/num_pos_points
            ThisLayActLoss = F.relu(-act).pow(2).mean(dim=(0,2,3)).sum()

            if L2NegActLoss is None :
                L2NegActLoss = ThisLayActLoss
            else :
                L2NegActLoss += ThisLayActLoss

        # add weighted ActLoss to the total loss
        loss += L2NegActLoss * Globs.L2NegActLossScaling

    ############################
    # L2 activation loss - this encourages sparse activation and prevents model inflation
    ############################
    if Globs.L2ActLossScaling > 0.0 :
        # 4 options:
        # [1] loss is the sum of the mean per map conv activations including the -ve and +ve
        # [2] loss is the sum of the abs mean per map conv activations including the -ve and +ve
        # [3] loss is the sum of the mean per map +ve conv activations i.e. excluding -ve activations
        # [4] loss is the sum of the mean per map +ve conv activations excluding all -ve or zero activations in the mean computation i.e. on +ve activations are considered with adaptive population size
        #   so activations are encouraged to be small normalised to the +ve population size which means the loss is equally strong no matter the population size vs smaller loss for sparser populations

        # [opt 3 : +ve clipped activations per map L1 norm] for each conv activation map
        # note :
        # - all activation maps are stored in a dictionary indexed by the layer name
        # note on method : by taking mean(abs(conv(x))) then activations need to be as small and sparse as possible and not become -ve biased i.e. they must stick around zero. This has a strong
        #   regularising effect to prevent activation growth in magnitude and therefore prevents weights inflation. At the same time the filter output is regularised to be close to zero which
        #   is a strong anchor for reducing mean shift at all.
        AllConvActivations = GetActivationsByTags ( AllOutputs, Tags=['conv'])
            
        StdL2ActLoss = False
        PosOnlyL2ActLoss = True

        # build computational graph sum_over_all_maps(per_map_mean(abs(activation)))
        L2ActLoss = None
        for act in AllConvActivations :
            # per layer sum over minibatch and filters of per filter mean of abs activation
            if StdL2ActLoss :
                ThisLayActLoss = act.pow(2).mean(dim=(0,2,3)).sum()
                #ThisLayActLoss = act.mean(dim=(0,2,3)).pow(2).sum()
            elif PosOnlyL2ActLoss:
                # this induces greater activation sparsity and pushes activations more -ve since there is no cost of -ve activation
                # Since activations are induced to be sparser then learning should be slower initially but ultimately a sparser representation may generalise more.
                    
                # act loss masked to +ve activations ignoring all zero or -ve activations
                # TODO: mask to +ve only points for normalisation i.e. sum(sqr(pos points))/num_pos_points
                ThisLayActLoss = F.relu(act).pow(2).mean(dim=(0,2,3)).sum()

            if L2ActLoss is None :
                L2ActLoss = ThisLayActLoss
            else :
                L2ActLoss += ThisLayActLoss

        # add weighted ActLoss to the total loss
        loss += L2ActLoss * Globs.L2ActLossScaling

    ############################
    # CLASSIFIER L2 activation loss - this encourages lateral inhibition and prevents model inflation or deflation
    ############################
    if Globs.ClassifierL2ActLossScaling > 0.0 :
        ClassActivation = GetActivationsByTags ( AllOutputs, Tags=['classifier'])[0]

        if False :
            # unity variant
            ClassL2ActLoss = ( 1.0 - ClassActivation).pow(2).sum(dim=(1,)).mean()
        else :
            # L2 variant
            ClassL2ActLoss = ClassActivation.pow(2).sum(dim=(1,)).mean()

        # add weighted ActLoss to the total loss
        loss += ClassL2ActLoss * Globs.ClassifierL2ActLossScaling

    ############################
    # CONVSCALING BALANCE LOSS
    # - scaling provides the target anchor for the decay of the kernel magnitude and critically the target is decoupled from gradients so scaling has no pressure applied i.e. inflation
    ############################
    if Globs.ConvScalingBalanceLossScaling > 0.0 and Globs.UseConvScaling :

        # note - conv and scaling parameters which must be in order
        PerFiltConv1Mag = Globs.Conv1WeightParams[0][1].pow(2).sum(dim=(1,2,3)).sqrt()

        ConvScalingBalanceLoss = None

        if True :  # sqrt(|W|.scaling) anchor

            # input conv special case
            if len(Globs.RawConv1ScalingParams)>0 :
                scaling =  Globs.RawConv1ScalingParams[0]
                if Globs.UseExpoScaling :
                    scaling =  Globs.RawConv1ScalingParams[0].exp()
                AnchorData = ( scaling.data * PerFiltConv1Mag.data ).sqrt()
                ConvScalingBalanceLoss = ( PerFiltConv1Mag - AnchorData ).pow(2).sum()

            for (weight, scaling) in zip ( Globs.ResConvWeightParams, Globs.RawScalingParams ) :
                PerFiltMag = weight[1].pow(2).sum(dim=(1,2,3)).sqrt()
                if Globs.UseExpoScaling :
                    scaling =  scaling.exp()
                # anchor is mid point between kernel and scaling (actually it should be the balance point ratio above and below
                AnchorData = ( scaling.data * PerFiltMag.data ).sqrt()
                ThisLoss = ( PerFiltMag - AnchorData ).pow(2).sum()

                if ConvScalingBalanceLoss is None :
                    ConvScalingBalanceLoss = ThisLoss
                else :
                    ConvScalingBalanceLoss += ThisLoss

        else :  # (|W|-scaling)^2 loss with gradients applied to scaling as well as W

            # input conv special case
            if len(Globs.RawConv1ScalingParams)>0 :
                scaling =  Globs.RawConv1ScalingParams[0]
                if Globs.UseExpoScaling :
                    scaling =  Globs.RawConv1ScalingParams[0].exp()
                ConvScalingBalanceLoss = ( PerFiltConv1Mag - scaling.data ).pow(2).sum()

            for (weight, scaling) in zip ( Globs.ResConvWeightParams, Globs.RawScalingParams ) :
                PerFiltMag = weight[1].pow(2).sum(dim=(1,2,3)).sqrt()
                if Globs.UseExpoScaling :
                    scaling =  scaling.exp()
                # anchor is mid point between kernel and scaling (actually it should be the balance point ratio above and below
                ThisLoss = ( PerFiltMag - scaling.data ).pow(2).sum()

                if ConvScalingBalanceLoss is None :
                    ConvScalingBalanceLoss = ThisLoss
                else :
                    ConvScalingBalanceLoss += ThisLoss

        loss += ConvScalingBalanceLoss * Globs.ConvScalingBalanceLossScaling

    ############################
    # ELR LOSS
    # - ELR = scaling/|W| so applying a loss directly to the ELR of something like (but not) 1/ELR will linearly cost the reduction in ELR with increase in its inverse.
    # - to prevent affecting the scaling then its gradient is detached so only |W| gradients are active
    # - ideally with LR very small then the very low inflation pressure should cause the ELR to approach 1 so |W|/scaling is not ideal for this as its gradient is 1 at ELR=1, whereas it needs to be zero.
    ############################
    if Globs.ELRLossScaling > 0.0 and Globs.UseConvScaling :

        ELRLoss = None

        for (weight, scaling, idx) in zip ( Globs.ResConvWeightParams, Globs.RawScalingParams, range(len(Globs.ResConvWeightParams)) ) :

            PerFiltMag = weight[1].pow(2).sum(dim=(1,2,3)).sqrt()
            if Globs.UseExpoScaling :
                scaling =  scaling.exp()

            # update running characteristic scale
            if True or Globs.RunningFilterScale[idx] is None :
            # if Globs.RunningFilterScale[idx] is None :
                Globs.RunningFilterScale[idx] = ( PerFiltMag.data * scaling.data ).sqrt()
            else :
                # update the running mean
                momentum = 0.99
                m = ( PerFiltMag.data * scaling.data ).sqrt()
                Globs.RunningFilterScale[idx] = Globs.RunningFilterScale[idx] * momentum + ( 1.0 - momentum ) * m

            # loss = ( |W| - m )^2 / m^2 
            #ThisLoss = ( ( PerFiltMag - Globs.RunningFilterScale[idx] ).pow(2) / Globs.RunningFilterScale[idx].pow(2) ).sum()
            # loss = ( |W| - m )^2
            ThisLoss = ( PerFiltMag - Globs.RunningFilterScale[idx] ).pow(2).sum()
            if ELRLoss is None :
                ELRLoss = ThisLoss
            else :
                ELRLoss += ThisLoss

        # append to the training loss
        loss += ( ELRLoss * Globs.ELRLossScaling )

    ############################
    # DIRECT SPARSITY LOSS
    # 
    ############################
    if Globs.SparsityLossScaling > 0.0 :

        SparsityLoss = None
        SparsityLimit = 2.0

        for weight in Globs.ResConvWeightParams :
            param = weight[1]
            if True :
                AbsWeights = param.abs()
                # max over kernel dimensions in 3 steps as torch.max does not support tuple dimensions
                PerFiltMax = AbsWeights.max (dim=1)[0]
                PerFiltMax = PerFiltMax.max (dim=1)[0]
                PerFiltMax = PerFiltMax.max (dim=1)[0]
                # per filter max norm by broadcasting per filter max to the kernel dimensions
                PerFiltMaxNorm = AbsWeights / PerFiltMax.view((-1,1,1,1))
                PerFiltMaxNormSum = PerFiltMaxNorm.sum(dim=(1,2,3))
                # loss is the excess above the sparsity target
                ThisLoss = ( PerFiltMaxNormSum - SparsityLimit ).pow(2).sum()
            elif False :
                AbsWeights = param.abs()
                # max over kernel dimensions in 3 steps as torch.max does not support tuple dimensions
                PerFiltMax = AbsWeights.max (dim=1)[0]
                PerFiltMax = PerFiltMax.max (dim=1)[0]
                PerFiltMax = PerFiltMax.max (dim=1)[0]
                # per filter max norm by broadcasting per filter max to the kernel dimensions
                PerFiltMaxNorm = AbsWeights / PerFiltMax.view((-1,1,1,1))
                ThisLoss = PerFiltMaxNorm.pow(2).sum()
            else :
                ThisLoss = param.pow(2).sum()

            if SparsityLoss is None :
                SparsityLoss = ThisLoss
            else :
                SparsityLoss += ThisLoss

        loss += SparsityLoss * Globs.SparsityLossScaling

    # res block conv1+2 mean scaling balance loss to encourage the same learning rate in both
    if Globs.ResBlockScalingBalanceLoss :
        Scaling1 = Globs.RawScalingParams[::2]
        Scaling2 = Globs.RawScalingParams[1::2]
        ScalingBalanceLoss = None
        for (s1,s2) in zip(Scaling1,Scaling2) :
            if Globs.UseExpoScaling :
                ThisLoss = ( s1.data.exp().mean() - s2.data.exp().mean() ).pow(2)
            else :
                ThisLoss = ( s1.data.mean() - s2.data.mean ).pow(2)

            if ScalingBalanceLoss is None :
                ScalingBalanceLoss = ThisLoss
            else :
                ScalingBalanceLoss += ThisLoss

        loss += ScalingBalanceLoss * Globs.ResBlockScalingBalanceLoss

    # L2 scaling decay
    if Globs.ScalingL2Decay > 0.0 and len(Globs.AllScalingParams)>0 :

        SumL2Scaling = None

        for i in range(len(Globs.AllScalingParams)) :

            # if exponential scaling then decay to zero else to 1
            if Globs.UseExpoScaling :
                ThisParam = Globs.AllScalingParams[i].exp()
            else :
                ThisParam = Globs.AllScalingParams[i]

            ThisSum =  ThisParam.pow(2).sum()

            if SumL2Scaling is None :
                SumL2Scaling = ThisSum
            else :
                SumL2Scaling += ThisSum

        loss += SumL2Scaling * Globs.ScalingL2Decay

    # L2 unity scaling decay - inflation bias to prevent filters staying off
    if Globs.ScalingL2UnityDecay > 0.0 and len(Globs.AllScalingParams)>0 :
        assert False, "this is not correctly coded as exposcaling requires exponentiation"
        SumL2Scaling = None
        for i in range(len(Globs.AllScalingParams)) :
            ThisParam = Globs.AllScalingParams[i]
            # if exponential scaling then decay to zero else to 1
            if Globs.UseExpoScaling :
                ThisSum =  ThisParam.pow(2).sum()
            else :
                ThisSum =  (1.0 - ThisParam).pow(2).sum()

            if SumL2Scaling is None :
                SumL2Scaling = ThisSum
            else :
                SumL2Scaling += ThisSum

        loss += SumL2Scaling * Globs.ScalingL2UnityDecay

    if Globs.ScalingFloatingMeanDecay > 0.0 and len(Globs.RawScalingParams)>0 :

        if Globs.UseExpoScaling :
            ScalingParams = [p.exp() for p in Globs.RawScalingParams]
        else :
            ScalingParams = Globs.RawScalingParams

        # compute mean scaling using numpy's mean function for convenience but this remains as a pytorch tensor
        ScalingSum = None
        NumScalings = 0
        for ThisScaling in ScalingParams :
            if len(ThisScaling.shape)==0 :
                NumScalings += 1
            else :
                NumScalings += np.prod(ThisScaling.shape)

            if ScalingSum is None :
                ScalingSum = ThisScaling.sum()
            else :
                ScalingSum = ScalingSum + ThisScaling.sum()

        ScalingMean = ScalingSum / np.float32(NumScalings)

        # compute L2 loss from mean
        SumL2ScalingLoss = None
        for ThisScaling in ScalingParams :

            # note - sum is taken even if scalar
            ThisLoss = ( ThisScaling - ScalingMean ).pow(2).sum()

            if SumL2ScalingLoss is None :
                SumL2ScalingLoss = ThisLoss
            else :
                SumL2ScalingLoss = SumL2ScalingLoss + ThisLoss

        loss += SumL2ScalingLoss * Globs.ScalingFloatingMeanDecay

    ##########################
    # Conv3x3 scale loss decay - anchors filters at unity using L1 Euclidean magnitude difference from unity
    ##########################
    # set anchor=1 for unity decay and anchor=0 for L2 decay equivalent
    if Globs.FilterL2ScaleLoss > 0.0 :
        FilterScaleSum = None
        for i in range(len(Globs.ConvWeightParams)) :
            Param = Globs.ConvWeightParams[i][1]
            ParamShape = Param.shape
            SumDims = tuple ( range(1,len(ParamShape)) )
            ThisPerFilterScale =  Param.pow(2).sum(dim=SumDims).sqrt()
            ThisPerFilterScaleUnityDiff = (ThisPerFilterScale-Globs.FilterScaleLossAnchor).pow(2)

            if FilterScaleSum is None :
                FilterScaleSum = ThisPerFilterScaleUnityDiff.sum()
            else :
                FilterScaleSum += ThisPerFilterScaleUnityDiff.sum()

        loss += FilterScaleSum * Globs.FilterL2ScaleLoss

    ################################
    # filter L2 zero mean decay loss
    ################################
    # channelwise sqr(mean(W))
    if Globs.ZeroMeanL2Decay > 0 :
        ZML2Sum = None
        for i in range(len(Globs.ConvWeightParams)) :
            Param = Globs.ConvWeightParams[i][1]
            ParamShape = Param.shape
            ChanDims = (2,3)
            ThisPerFilterL2Mean = Param.mean(dim=ChanDims).pow(2)

            if ZML2Sum is None :
                ZML2Sum = ThisPerFilterL2Mean.sum()
            else :
                ZML2Sum += ThisPerFilterL2Mean.sum()

        loss += ZML2Sum * Globs.ZeroMeanL2Decay

    ################################
    # filter L2 zero mean decay loss
    ################################
    # channelwise sqr(mean(W))
    if Globs.ZeroSumL2Decay > 0 :
        ZSL2Sum = None
        for i in range(len(Globs.ConvWeightParams)) :
            Param = Globs.ConvWeightParams[i][1]
            ParamShape = Param.shape
            ChanDims = (1,2,3)
            BankPerFilterL2Sum = Param.sum(dim=ChanDims).pow(2)

            if ZSL2Sum is None :
                ZSL2Sum = BankPerFilterL2Sum.sum()
            else :
                ZSL2Sum += BankPerFilterL2Sum.sum()

        loss += ZSL2Sum * Globs.ZeroSumL2Decay

    # per channel zero mean weight decay for conv layers - this promotes per filter per channel zero mean kernel slice which is stable for rejecting common mode noise
    if Globs.ZMW > 0.0 :
        ZML1Sum = None
        for i in range(len(Globs.ConvWeightParams)) :
            Param = Globs.ConvWeightParams[i][1]
            ParamShape = Param.shape
            ChanDims = (2,3)
            ThisPerFilterL1Mean = Param.mean(dim=ChanDims).abs()

            if ZML1Sum is None :
                ZML1Sum = ThisPerFilterL1Mean.sum()
            else :
                ZML1Sum += ThisPerFilterL1Mean.sum()

        loss += ZML1Sum * Globs.ZMW

    # PROJECTION UNITY LOSS
    if Globs.HasReluConv1x1 and Globs.ProjUnityDecayLoss > 0.0 :
        LossSum = None
        for i in range(len(Globs.ResProjWeightParams)) :
            Param = Globs.ResProjWeightParams[i][1]
            ParamShape = Param.shape
            SumDims = tuple ( range(1,len(ParamShape)) )
            ThisPerFilterScale =  Param.pow(2).sum(dim=SumDims).sqrt()
            ThisPerFilterScaleUnityDiff = (ThisPerFilterScale-1.0).pow(2)

            if LossSum is None :
                LossSum = ThisPerFilterScaleUnityDiff.sum()
            else :
                LossSum += ThisPerFilterScaleUnityDiff.sum()

        loss += LossSum * Globs.ProjUnityDecayLoss

    if Globs.ClassifierUnityDecayLoss > 0.0 :
        Param = Globs.ClassifierWeightParams[0][1]
        ParamShape = Param.shape
        SumDims = tuple ( range(1,len(ParamShape)) )
        ClassifierMagPerChannel =  Param.pow(2).sum(dim=SumDims).sqrt()
        ClassifierL2UnityPerChannel = (ClassifierMagPerChannel-1.0).pow(2)
        ClassifierL2UnitySum = ClassifierL2UnityPerChannel.sum()

        loss += ClassifierL2UnitySum * Globs.ClassifierUnityDecayLoss

    if Globs.ClassifierL2Decay > 0.0 :
        Param = Globs.ClassifierWeightParams[0][1]
        ClassifierL2DecayLoss = Param.pow(2).sum()
        loss += ClassifierL2DecayLoss * Globs.ClassifierL2Decay

    if Globs.Conv1UnityDecayLoss > 0.0 :
        Param = Globs.Conv1WeightParams[0][1]
        ParamShape = Param.shape
        SumDims = tuple ( range(1,len(ParamShape)) )
        Conv1MagPerChannel =  Param.pow(2).sum(dim=SumDims).sqrt()
        Conv1L2UnityPerChannel = (Conv1MagPerChannel-1.0).pow(2)
        Conv1L2UnitySum = Conv1L2UnityPerChannel.sum()

        loss += Conv1L2UnitySum * Globs.Conv1UnityDecayLoss

    if Globs.Conv1ScalingUnityAnchorLoss > 0.0 :
        Conv1ScalingData = Globs.Conv1ScalingParams[0][1].data
        Conv1ScalingDataExp = Conv1ScalingData.exp()
        Conv1ScalingL2UnityPerChannel = ( Conv1ScalingDataExp - 1.0 ).pow(2)
        Conv1ScalingL2UnitySum = Conv1ScalingL2UnityPerChannel.sum()

        loss += Conv1ScalingL2UnitySum * Globs.Conv1ScalingUnityAnchorLoss

def ApplyAllGradientModifications ( ) :

    # optionally normalize gradient magnitudes according to filter-wise relative weight scale to max weight - this prevents small weights having large updates to curb mean shift at source
    # - this reduces the nuisance gradients of small weights but does not prevent them learning at a reduced rate so allowing weights to change sign by moving across zero
    # - set NormToMax to avoid LR being affected as otherwise the scale of filter changes the LR
    # NB gradmaxnorm is applied before other gradient techniques here so be careful to note that the gradients are modified and if raw gradients are needed then place that code BEFORE here
    if Globs.UseGradMaxNorm :
        FilterwiseMaxNormGrads ( Globs.ConvWeightParams, eps=0.01, NormToMax=True, Filterwise=True )

    # optionally modify the gradients to be zero mean for convolutional params
    if Globs.ZeroMeanGrad > 0.0 :
        ZeroMeanNormaliseGrads ( Globs.ConvWeightParams, PerChannel=Globs.PerChannel, MeanMult=Globs.ZeroMeanGrad )

    # optionally inject noise into the gradients using the rms size of the gradients across the whole filter
    if Globs.WeightNoiseInjectionScale > 0.0 :
        #WeightNoiseInjection ( Globs.ConvWeightParams, Scale=Globs.WeightNoiseInjectionScale, UseWeightScaling=Globs.WeightNoiseUseWeightScaling )
        GradNoiseDict = ComputeGradNoise ( Globs.ConvWeightParams, Scale=Globs.WeightNoiseInjectionScale, UseWeightScaling=Globs.WeightNoiseUseWeightScaling )
    else :
        GradNoiseDict = None

    # apply gradient noise that was pre-computed before optional max norm of gradients so that the noise retains its CZM condition
    # APPLY PER PARAMETER LEARNING RATE ADJUSTMENT TO GRADIENTS
    if GradNoiseDict is not None :
        for p in Globs.ConvWeightParams :
            param = p[1]
            param.grad.data += GradNoiseDict[param]

    # notes:
    # - GradLRDict is indexed by a parameter and if a param is in the dictionary then its grad is multiplied by the tensor it indexes

    # firstly create tuple of (weight,grad) to associate the param's tensor with its gradient for the conv param whose tensor is referred to by p.weight
    if False and Globs.AdaptLR2Scaling and ( Globs.HasScaling or Globs.UseConvScaling) :
        # apply using scaling LR dict
        Globs.NamedConvParams = [(p[0],p[1]) for p in Globs.ConvWeightParams]
        LearningRateAdjustGradients ( Globs.NamedConvParams, Globs.GradLRDict )

    # optional adjust gradients of BN conv kernels by grad *= mag(W) which reduces the effect of the local LR increase from 1/mag(W)^2 to 1/mag(w)
    if Globs.ScaleConvGradientsByMag and Globs.UseBatchNorm :
        ScaleConvGradientsByMag ( Globs.ConvWeightParams, SquareLaw=True )

    if Globs.AdaptBiasLR2FilterMag :
        # for each convolutional layer (conv and linear filters) adjust the bias gradients according to the euclidean magnitude of the filter.
        # note - this is setup in the GradLRDict and is outside of the computational graph as this is not a regulariser and also otherwise circular gradients
        LearningRateAdjustGradients ( Globs.NamedBiasParams, Globs.GradLRDict )

    # optional normalise conv gradients to the Euclidean magnitude of the filter - this supports the same dynamic inflation VS weight decay that WeightNorm and BatchNorm perform but without the normalisation.
    if Globs.NormConvGrad2KernelMag :
        NormaliseConvGrad2KernelMag ( Globs.ConvWeightParams, WeightScalingDict=Globs.WeightScalingDict, IncludeScaling=False )

    if Globs.NormaliseConvGrad2ELR or Globs.AdaptLR2Scaling :
        NormaliseConvGrad2ELR ( Globs.ConvWeightParams, WeightScalingDict=Globs.WeightScalingDict, Adapt2Weight=Globs.NormaliseConvGrad2ELR, Adapt2Scaling=Globs.AdaptLR2Scaling )
        
    # adapt scaling LR to itself so that its ELRM is unity
    # [10-FEB-2020]
    if Globs.AdaptScalingLR and len(Globs.RawScalingParams) > 0 :
        for param in Globs.RawScalingParams :
            if Globs.UseExpoScaling :
                param.grad.data *= param.data.exp()
            else :
                param.grad.data *= param.data.abs()

    # apply ELRM to kernels according to the model inflation pressure
    # notes :
    # - the model inflation pressure (MIP) is measured using the very long term running mean MIP' = MIP*m + DelInf*(m-1)
    #   where m is a momentum term e.g. 0.98, and DelInf is the signed kernel magnitude difference between the current step and the fixed previous step magnitude which is the previous scaling value typically using resets
    if Globs.MIPScaling > 0 :
        # apply gradient mod according to g' = g / ( 1 + MIP * R )  where R is the rate modifier to be calibrated
        for ( namedparam, mip ) in zip ( Globs.ConvWeightParams, Globs.RunningMIP ) :
            param = namedparam[1]
            if True :
                if len(param.data.shape)==2 :
                    param.grad.data /= ( 1.0 + mip * Globs.MIPScaling ).view( (-1,)+(len(param.data.shape)-1)*(1,) )
                else :
                    param.grad.data /= ( 1.0 + mip * Globs.MIPScaling )
            else :
                LRM = torch.exp ( ( mip - mip.mean()) * Globs.MIPScaling )
                # note - numerical stability is applied so that the LR multiplier is < 1/0.1 i.e. <10
                param.grad.data = param.grad.data / torch.where ( LRM > 0.1, LRM, torch.ones_like(LRM) ).view( (-1,)+(len(param.data.shape)-1)*(1,) )

        # record the filter magnitude before the update
        PrevFiltMag = []
        for namedparam in Globs.ConvWeightParams :
            pdata = namedparam[1].data
            KernelDims = tuple ( range( 1, len(pdata.shape) ) )
            ThisFiltMag = pdata.pow(2).sum(dim=KernelDims).sqrt()
            PrevFiltMag.append ( ThisFiltMag )

def ApplyAllModelPostProcessing () :

    # transfer scaling to weights
    # notes :
    # - this keeps the scaling at unity and avoids dealing with kernel Euclidean magnitude which is then free to adopt any value rather than being anchored while countering model 
    # - after this operation the scaling values are unity taking into account exponentiation (which is not needed)
    if Globs.TransferScalingtoWeights :
        # input conv is special case
        if Globs.InputConvHasScaling :
            TransferScalingtoWeights ( Globs.Conv1WeightParams, Globs.Conv1ScalingParams, UseExpoScaling=Globs.UseExpoScaling, ScalingMag=1.0 )

        TransferScalingtoWeights ( Globs.ConvWeightParams, Globs.ScalingParams, UseExpoScaling=Globs.UseExpoScaling, ScalingMag=Globs.ScalingInitMagnitudeScale)

    # optionally each filter is reset to its scaling (if it has one)
    if Globs.ResetFilterToScaling :
        # input conv layer
        if Globs.InputConvHasScaling and not Globs.ResetFilterMagnitude :
            ResetFilterToScaling ( Globs.Conv1WeightParams, Globs.Conv1ScalingParams, UseExpoScaling=Globs.UseExpoScaling )
        # all residual conv layers
        ResetFilterToScaling ( Globs.ConvWeightParams, Globs.ScalingParams, UseExpoScaling=Globs.UseExpoScaling)

    if Globs.ResetClassifierToScaling and Globs.ClassHasScaling :
        # optionally reset classifier and its scaling if there is a scaling
        ResetFilterToScaling ( Globs.ClassifierWeightParams, Globs.ClassifierScalingParams, UseExpoScaling=Globs.UseExpoScaling )

    if Globs.ClassScaleResetMag is not None :
        ResetFilterMagnitude ( Globs.ClassifierWeightParams, NormMode='mean', Scale=Globs.ClassScaleResetMag )

    # optionally normalize all convolutional filters to unit magnitude
    # - conv1 is special case where each filter is separately renormalized to unit magnitude and has indivual scaling
    if Globs.ResetFilterMagnitude :
        #ResetFilterMagnitude ( Globs.Conv1WeightParams, NormMode='unity', Scale=1.0 )
        ResetFilterMagnitude ( Globs.ConvWeightParams, NormMode='unity', Scale=Globs.FilterScaleLossAnchor )

        # reset conv1scaling normalizing to mean - hard coded
        # - if used in conjunction with unity decay then this balances the mean to unity
        if False :
            Conv1ScalingData = Globs.Conv1ScalingParams[0][1].data
            Conv1ScalingDataExp = Conv1ScalingData.exp()
            Conv1ScalingDataExpMean = torch.mean ( Conv1ScalingDataExp )
            Conv1ScalingDataExpMeanNorm = Conv1ScalingDataExp / Conv1ScalingDataExpMean
            Globs.Conv1ScalingParams[0][1].data = Conv1ScalingDataExpMeanNorm.log()
            
    # optionally clamp scaling magnitude to a lower bound (e.g. 0.1) to prevent excessive LR which leads to instability and excessive filter inflation
    if ( Globs.HasScaling or Globs.UseConvScaling) and Globs.ScalingLowerBound is not None :
        if Globs.UseExpoScaling :
            ScalingLowerBound = np.log(Globs.ScalingLowerBound+0.00001)
        else :
            ScalingLowerBound = Globs.ScalingLowerBound

        ClampScalingToLowerBound ( Globs.AllScalingParams, ScalingLowerBound )

    # optional Euclidean filter norm xfer to scaling xfer(weights,scaling)
    if ( Globs.HasScaling or Globs.UseConvScaling) and Globs.UseFilterEuclidNormXferToScaling :
        # xfer weights magnitude to scaling - note both weights and scalings are in same order due to construction order and each 3x3 conv layer must have a scaling
        FilterEuclidNormXferToScaling ( Globs.ConvWeightParams, Globs.ScalingParams, ExpoScaling=Globs.UseExpoScaling )

    # optionally the classifier is normalised to unity Euclidean magnitude so that the deep net is balanced both for output magnitude and for LR through each class filter
    if Globs.ClassifierUnityMag :
        # classifier
        LocalClassifierWeightParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), "weight", False, ndim=2, WithName=False )
        # for each filter bank
        # get shortcut to the data tensor
        ClassFilterBankData = LocalClassifierWeightParams[0].data
        # compute the magnitude for each filter in the filterbank
        PerFilterMag = ClassFilterBankData.pow(2).sum(dim=(1)).sqrt()
        # scale the per filter kernels according to the mask
        ClassFilterBankData /= PerFilterMag.view ( ( ClassFilterBankData.shape[0], 1) )

    # optionally reset all conv3x3 filters to unity magnitude - note the fraction is not transferred to the scaling as the std normalisation statistically removes the filter scale anyway - this just resets the local LR scaling to unity
    if Globs.ConvUnityMagReset :
        ParamData = [p[1].data for p in Globs.ConvWeightParams ]
        # for each filter bank
        for pdata in ParamData :
            # compute the magnitude for each filter in the filterbank
            PerFilterMag = pdata.pow(2).sum(dim=(1,2,3)).sqrt()
            # scale the per filter kernels according to the mask
            pdata /= PerFilterMag.view ( (pdata.shape[0],1,1,1) )

    if Globs.ResProjectionUnityMagReset :
        # classifier
        #AllConvWeightParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), "weight", False, ndim=4, WithName=False )
        # very quick hack to get conv 1x1 projection using the 2D size - note PyTorch has no way to tag parameters so otherwise need to go through entire layer list using the separate tags for each layer which is a correct approach but cumbersome without a well thought out framework
        #ProjParamData = [p.data for p in AllConvWeightParams if p.shape[2:]==(1,1) ]
        ProjParamData = [ p[1].data for p in Globs.ResProjWeightParams ]
        # for each filter bank
        for pdata in ProjParamData :
            # compute the magnitude for each filter in the filterbank
            PerFilterMag = pdata.pow(2).sum(dim=(1,2,3)).sqrt()
            # scale the per filter kernels according to the mask
            pdata /= PerFilterMag.view ( (pdata.shape[0],1,1,1) )

def TrainOneEpoch ( train_loader, model, criterion, optimizer, EpochIdx, StatsOutputEvery, IsFirstEpoch=False, DualMinibatch=False ) :
    """
    Train all the minibatches in an epoch using the dual minibatch method.

    DUAL MINIBATCH METHOD :
        The usual minibatch training method computes the gradients that reduce the losses on the model to the data by moving the model towards the data. This causes the model to learn all structure
        of the data whether general or just detail of a particular sample. However detail is of no use as this is just memorization of the training samples. To avoid single sample mmemorization a dual
        minibatch is used which is simply the minibatch split in half. The model is moved towards min(mb1_grad, mb2_grad). This has the effect of ignoring the gradients that derive from sample details
        since if mb1 has a particular sample I_k then this is not present in mb2 and so the gradients in mb1 that arise from the details loss in I_k are not present in mb2 i.e. they are smaller. So taking
        min(mb1_grad, mb2_grad) ensures that the gradients arising from particular samples are not considered.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # running sum of precision over the epoch for presenting the mean precision which is useful later in training to compare the validation and training precision
    Globs.EpochRunningPrecisionSum = 0.0

    # switch to train mode
    model.train()

    NumSamplesInEpoch = len(train_loader.dataset.data)
    MinibatchSize = train_loader.batch_size
    NumStepsInEpoch = NumSamplesInEpoch//MinibatchSize

    # SETUP A PROGRESS BAR FOR SHOWING STATS AND PROGREES OF GROUPS OF MINIBATCHES
    TrainStatsGroupProgressBar = ConsoleProgressBar ( StatsOutputEvery, InfoStr="" )
    # and hide it as it is not in use yet
    TrainStatsGroupProgressBar.Hide()
    # stats group
    TrainStatsGroup = TrainingStats()

    # draw the samples for this epoch
    DrawStartTime = time.time()
    TrainLoadDraw = enumerate(train_loader)
    data_time.update ( time.time() - DrawStartTime )

    # record time at start of epoch
    EpochStartTime = time.time()

    TrainStatsGroupProgressBar.Show()

    GradsBuf = None

    # loop over all minibatches in dataset
    for MinibatchIdx, (input_cpu, target_cpu) in TrainLoadDraw :

        # set input and target to non-blocking xfer to GPU
        input = input_cpu.to(Globs.Device,non_blocking=True)
        target = target_cpu.to(Globs.Device,non_blocking=True)
        input_var = torch.autograd.Variable(input).to(Globs.Device)
        target_var = torch.autograd.Variable(target).to(Globs.Device)      # CHECK ME

        if Globs.Half:
            input_var = input_var.half().to(Globs.Device)      # CHECK ME

        AllOutputs = model(input_var)
        output = AllOutputs[-1].Out
        loss = criterion(output, target_var)
        target_loss = loss.data.clone()
        AddAllAuxiliaryLosses ( loss )
        optimizer.zero_grad()
        loss.backward()
        ApplyAllGradientModifications ( )

        if DualMinibatch :

            AllConvParams = Globs.ConvWeightParams #+ Globs.ScalingParams 
            if Globs.PrevGrads is None :
                # just copy current grads ready for next training step
                Globs.PrevGrads = ParamUtil.DeepCopyGrads ( AllConvParams )
            else :
                # first deepcopy the current grads as these are needed for the next training step
                if GradsBuf is None :
                    GradsBuf = ParamUtil.DeepCopyGrads ( AllConvParams )
                else :
                    for ( namedp, buf_g ) in zip ( AllConvParams, GradsBuf ) :
                        p = namedp[1]
                        buf_g[...] = p.grad.data

                # update the gradients to the abs min of the dual minibatchs 
                for ( namedp, prev_g ) in zip ( AllConvParams, Globs.PrevGrads ) :
                    p = namedp[1]
                    p.grad.data[...] = torch.where ( p.grad.data.abs() < prev_g.abs(), p.grad.data, prev_g )

                # deepcopy the unmodified (i.e. buffered) grads to the existing previous grad storage ( just avoids constant memory management )
                for ( buf_g, prev_g ) in zip ( GradsBuf, Globs.PrevGrads ) :
                    prev_g[...] = buf_g

        # perform parameter update step using parameter gradients
        optimizer.step()

        # apply all post processing steps to the model e.g. resets and scale transfers
        ApplyAllModelPostProcessing ()

        # convert graph outputs to Python floats for recording and display
        #OutputFloat = output.float()

        # measure and record accuracy
        ThisMinibatchPrec = ComputePrecision ( output, target )
        #ThisMinibatchLoss = np.asscalar(target_loss.data)
        ThisMinibatchLoss = target_loss.data.item()
        losses.update ( ThisMinibatchLoss, MinibatchSize )
        top1.update(ThisMinibatchPrec, MinibatchSize )

        # log these to TensorBoard
        Globs.Log.AddToLog ( 'TrainStepLoss', ThisMinibatchLoss )
        Globs.Log.AddToLog ( 'TrainStepPrec', ThisMinibatchPrec )

        # running total of precision
        Globs.EpochRunningPrecisionSum += ThisMinibatchPrec

        # measure elapsed time
        batch_time.update(time.time() - EpochStartTime )

        # update training stats by recording loss and error
        TrainStatsGroup.Update ( ThisMinibatchLoss, ThisMinibatchPrec )

        # output progress and on last step remove
        if not Globs.SilentTrainingStats :
            TrainStatsGroupProgressBar.Update()

        if TrainStatsGroupProgressBar.Complete or MinibatchIdx==NumStepsInEpoch :
            if MinibatchIdx==NumStepsInEpoch :
                pass

            # remove and reset the progress
            if not Globs.SilentTrainingStats :
                TrainStatsGroupProgressBar.Hide()
            TrainStatsGroupProgressBar.Reset()

            # output group stats on end of group
            TotalTrainingSteps = MinibatchIdx + EpochIdx*NumStepsInEpoch
            TotalSamplesTrained = TotalTrainingSteps * MinibatchSize
            if not Globs.SilentTrainingStats :
                OutputTrainingStatsGroup ( TrainStatsGroup, MinibatchIdx, TotalTrainingSteps, TotalSamplesTrained, EpochIdx )

            # log these to TensorBoard
            Globs.Log.AddToLog ( 'TrainProgMeanLoss', TrainStatsGroup.MeanLoss )
            Globs.Log.AddToLog ( 'TrainProgMaxLoss', TrainStatsGroup.TrainLossMax )
            Globs.Log.AddToLog ( 'TrainProgMinLoss', TrainStatsGroup.TrainLossMin )
            Globs.Log.AddToLog ( 'TrainProgMeanPrec', TrainStatsGroup.MeanErr )
            Globs.Log.AddToLog ( 'TrainProgMaxPrec', TrainStatsGroup.TrainErrMax )
            Globs.Log.AddToLog ( 'TrainProgMinPrec', TrainStatsGroup.TrainErrMin )

            # reset stats for next group
            TrainStatsGroup.Reset()

        # any special conditions for first epoch
        if IsFirstEpoch :

            # conv layers with BN may adopt -ve scalings which affects anything but zero anchor decays so in case of -ve scaling then the signs of the filter kernel and scaling are flipped
            if Globs.FlipNegFilterAndScalingSign and Globs.UseBatchNorm :
                FlipFiltersWithNegScaling ( model.ConvBN )

        # TORCH CLEANUP to prevent ever slowing training
        #for out in AllOutputs :
        #    del out
        del output
        del input_var
        del target_var
        del input
        del target
        del loss
        del target_loss
        torch.cuda.empty_cache()

    # end of epoch - remove training stats
    if TrainStatsGroupProgressBar.Visible and not Globs.SilentTrainingStats :
        TrainStatsGroupProgressBar.Hide()

    # TORCH CLEANUP to prevent ever slowing training
    for (out,idx) in zip(AllOutputs,range(len(AllOutputs))) :
        AllOutputs[idx] = None
        del out
    del AllOutputs
    del TrainLoadDraw
    torch.cuda.empty_cache()

def TrainOneEpoch_DualMinibatch ( train_loader, model, criterion, optimizer, EpochIdx, StatsOutputEvery, IsFirstEpoch=False, DualMinibatch=False ) :
    """
    Train all the minibatches in an epoch using the dual minibatch method.

    DUAL MINIBATCH METHOD :
        The usual minibatch training method computes the gradients that reduce the losses on the model to the data by moving the model towards the data. This causes the model to learn all structure
        of the data whether general or just detail of a particular sample. However detail is of no use as this is just memorization of the training samples. To avoid single sample mmemorization a dual
        minibatch is used which is simply the minibatch split in half. The model is moved towards min(mb1_grad, mb2_grad). This has the effect of ignoring the gradients that derive from sample details
        since if mb1 has a particular sample I_k then this is not present in mb2 and so the gradients in mb1 that arise from the details loss in I_k are not present in mb2 i.e. they are smaller. So taking
        min(mb1_grad, mb2_grad) ensures that the gradients arising from particular samples are not considered.
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # running sum of precision over the epoch for presenting the mean precision which is useful later in training to compare the validation and training precision
    Globs.EpochRunningPrecisionSum = 0.0

    # switch to train mode
    model.train()

    NumSamplesInEpoch = len(train_loader.dataset.data)
    MinibatchSize = train_loader.batch_size
    NumStepsInEpoch = NumSamplesInEpoch//MinibatchSize

    # SETUP A PROGRESS BAR FOR SHOWING STATS AND PROGREES OF GROUPS OF MINIBATCHES
    TrainStatsGroupProgressBar = ConsoleProgressBar ( StatsOutputEvery, InfoStr="" )
    # and hide it as it is not in use yet
    TrainStatsGroupProgressBar.Hide()
    # stats group
    TrainStatsGroup = TrainingStats()

    # draw the samples for this epoch
    DrawStartTime = time.time()
    TrainLoadDraw = enumerate(train_loader)
    data_time.update ( time.time() - DrawStartTime )

    # record time at start of epoch
    EpochStartTime = time.time()

    TrainStatsGroupProgressBar.Show()

    # loop over all minibatches in dataset
    for MinibatchIdx, (input_cpu, target_cpu) in TrainLoadDraw :

        # set input and target to non-blocking xfer to GPU
        input = input_cpu.to(Globs.Device,non_blocking=True)
        target = target_cpu.to(Globs.Device,non_blocking=True)
        input_var = torch.autograd.Variable(input).to(Globs.Device)
        target_var = torch.autograd.Variable(target).to(Globs.Device)      # CHECK ME

        if Globs.Half:
            input_var = input_var.half().to(Globs.Device)      # CHECK ME

        # split the minibatch into two halves for the dual training
        HalfMBSize = input_cpu.shape[0]//2
        input_var1 = input_var[:HalfMBSize]
        input_var2 = input_var[HalfMBSize:]
        target_var1 = target_var[:HalfMBSize]
        target_var2 = target_var[HalfMBSize:]

        # MINIBATCH 1
        AllOutputs = model(input_var1)
        output1 = AllOutputs[-1].Out
        OutputData1 = output1.data.clone()
        loss = criterion(output1, target_var1)
        target_loss1 = loss.data.clone()
        AddAllAuxiliaryLosses ( loss )
        optimizer.zero_grad()
        loss.backward()
        ApplyAllGradientModifications ( )
        # copy the gradients of all parameters
        Grads1 = ParamUtil.DeepCopyAllGrads ( model )

        # MINIBATCH 2
        AllOutputs = model(input_var2)
        output2 = AllOutputs[-1].Out
        OutputData2 = output2.data.clone()
        loss = criterion(output2, target_var2)
        target_loss2 = loss.data.clone()
        AddAllAuxiliaryLosses ( loss )
        optimizer.zero_grad()
        loss.backward()
        ApplyAllGradientModifications ( )
        # copy the gradients of all parameters
        Grads2 = ParamUtil.DeepCopyAllGrads ( model )

        Params = ParamUtil.TrainableParams ( model )

        # update the gradients to the abs min of the dual minibatchs 
        for ( p, g1, g2 ) in zip ( Params, Grads1, Grads2 ) :
            p.grad.data[...] = torch.where ( g1.abs() < g2.abs(), g1, g2 )

        # perform parameter update step using parameter gradients
        optimizer.step()

        # apply all post processing steps to the model
        ApplyAllModelPostProcessing ()

        # convert graph outputs to Python floats for recording and display
        CombinedOutput = torch.cat ( [OutputData1, OutputData2] )
        output = CombinedOutput.float()
        target_loss = (( target_loss1 + target_loss2)/2.0).float()

        # measure and record accuracy
        ThisMinibatchPrec = ComputePrecision ( output, target )
        ThisMinibatchLoss = np.asscalar(target_loss.data)
        losses.update ( ThisMinibatchLoss, MinibatchSize )
        top1.update(ThisMinibatchPrec, MinibatchSize )

        # log these to TensorBoard
        Globs.Log.AddToLog ( 'TrainStepLoss', ThisMinibatchLoss )
        Globs.Log.AddToLog ( 'TrainStepPrec', ThisMinibatchPrec )

        # running total of precision
        Globs.EpochRunningPrecisionSum += ThisMinibatchPrec

        # measure elapsed time
        batch_time.update(time.time() - EpochStartTime )

        # update training stats by recording loss and error
        TrainStatsGroup.Update ( ThisMinibatchLoss, ThisMinibatchPrec )

        # output progress and on last step remove
        if not Globs.SilentTrainingStats :
            TrainStatsGroupProgressBar.Update()

        if TrainStatsGroupProgressBar.Complete or MinibatchIdx==NumStepsInEpoch :
            if MinibatchIdx==NumStepsInEpoch :
                pass

            # remove and reset the progress
            if not Globs.SilentTrainingStats :
                TrainStatsGroupProgressBar.Hide()
            TrainStatsGroupProgressBar.Reset()

            # output group stats on end of group
            TotalTrainingSteps = MinibatchIdx + EpochIdx*NumStepsInEpoch
            TotalSamplesTrained = TotalTrainingSteps * MinibatchSize
            if not Globs.SilentTrainingStats :
                OutputTrainingStatsGroup ( TrainStatsGroup, MinibatchIdx, TotalTrainingSteps, TotalSamplesTrained, EpochIdx )

            # log these to TensorBoard
            Globs.Log.AddToLog ( 'TrainProgMeanLoss', TrainStatsGroup.MeanLoss )
            Globs.Log.AddToLog ( 'TrainProgMaxLoss', TrainStatsGroup.TrainLossMax )
            Globs.Log.AddToLog ( 'TrainProgMinLoss', TrainStatsGroup.TrainLossMin )
            Globs.Log.AddToLog ( 'TrainProgMeanPrec', TrainStatsGroup.MeanErr )
            Globs.Log.AddToLog ( 'TrainProgMaxPrec', TrainStatsGroup.TrainErrMax )
            Globs.Log.AddToLog ( 'TrainProgMinPrec', TrainStatsGroup.TrainErrMin )

            # reset stats for next group
            TrainStatsGroup.Reset()

        # any special conditions for first epoch
        if IsFirstEpoch :

            # conv layers with BN may adopt -ve scalings which affects anything but zero anchor decays so in case of -ve scaling then the signs of the filter kernel and scaling are flipped
            if Globs.FlipNegFilterAndScalingSign and Globs.UseBatchNorm :
                FlipFiltersWithNegScaling ( model.ConvBN )

        # TORCH CLEANUP to prevent ever slowing training
        #for out in AllOutputs :
        #    del out
        del output
        del input_var
        del target_var
        del input
        del target
        del loss
        del target_loss
        torch.cuda.empty_cache()

    # end of epoch - remove training stats
    if TrainStatsGroupProgressBar.Visible and not Globs.SilentTrainingStats :
        TrainStatsGroupProgressBar.Hide()

    # TORCH CLEANUP to prevent ever slowing training
    for (out,idx) in zip(AllOutputs,range(len(AllOutputs))) :
        AllOutputs[idx] = None
        del out
    del AllOutputs
    del TrainLoadDraw
    torch.cuda.empty_cache()

def ApplyPreValidationProtocol ( Model, NumSteps, Data ) :

    DataDraw = islice ( enumerate(Data), 0, NumSteps )

    StatsGroupProgressBar = ConsoleProgressBar ( NumSteps, InfoStr="VALIDATION BN STATS PRE-WARM" )

    for MinibatchIdx, (input_cpu, target_cpu) in DataDraw :
        input = input_cpu.to ( Globs.Device, non_blocking=True )
        input_var = torch.autograd.Variable(input).to(Globs.Device)

        if Globs.Half:
            input_var = input_var.half().to(Globs.Device)      # CHECK ME

        AllOutputs = Model ( input_var )
        StatsGroupProgressBar.Update()

        # clean up CUDA mem
        del input
        del input_var
        torch.cuda.empty_cache()

    # remove the progress bar
    StatsGroupProgressBar.Hide()

    # clean up CUDA mem
    torch.cuda.empty_cache()

def AnnealL2DecayToLR ( ParamGroups, BaseL2Decay ) :
    for group in ParamGroups :
        lr = group['lr']
        initial_lr = group['initial_lr']
        group['weight_decay'] = BaseL2Decay * lr / initial_lr
        pass

def CreateWeightScalingDict ( WeightScalingList ) :
    ret = {}
    for weight_scaling in WeightScalingList :
        ret[weight_scaling.WeightName] = weight_scaling
        #ret[named_weight_param[0]] = (scaling_param,named_weight_param[1])

    return ret

""" MAIN """

def main():

    UseMultiGPU = False

    # make only the specified GPU visible to pytorch
    CudaDeviceStr = os.environ['CUDA'].lower()
    if CudaDeviceStr[:4] == 'cuda' :
        CudaDeviceID = int(CudaDeviceStr[-1])
    else :
        CudaDeviceID = -1

    #os.environ['CUDA_VISIBLE_DEVICES'] = str(CudaDeviceID)
    torch.cuda.set_device(CudaDeviceID) 

    # record cuDNN enabled state
    Globs.cuDNN = torch.backends.cudnn.enabled
    # cudnn.benchmark must be enabled for cuDNN to run optimisations (apparently)
    torch.backends.cudnn.benchmark = True
    Globs.cuDnnBenchmark = torch.backends.cudnn.benchmark
    # console output of CUDA availability
    print ( "CUDA availability : cuDNN[%s] cuDnnBenchMark[%s]" % (Globs.cuDNN,Globs.cuDnnBenchmark) )

    # create empty LR dictionary which is used in the model to setup any per parameter LR adjustment
    Globs.GradLRDict = {}

    # record the device in global as there is no way find out the current device (at least not in the documentation though no doubt it is there somewhere)
    Globs.Device = torch.device(CudaDeviceStr)
    Config.CurrentDevice = Globs.Device

    Globs.Half = args.half and CudaDeviceStr != 'cpu'

    # choose net
    if Globs.UsePyTorchModel :
        model = models.resnet101()

    else :
        # create the CNN according to the function whose name corresponds to the model name
        ModelFunc = ResnetBiasScaling.__dict__[Globs.Net]
        model = ModelFunc ( UseBatchNorm            = Globs.UseBatchNorm,
                            ScalingInitLow          = Globs.ScalingInitLow,
                            ScalingInitHigh         = Globs.ScalingInitHigh,
                            HasBias                 = Globs.HasBias,
                            HasPreBias              = Globs.HasPreBias,
                            HasScaling              = Globs.HasScaling,
                            NoConv1Scaling          = Globs.NoConv1Scaling,
                            NoConv2Scaling          = Globs.NoConv2Scaling,
                            InputConvHasScaling     = Globs.InputConvHasScaling,
                            UseExpoScaling          = Globs.UseExpoScaling,
                            SharedScaling           = Globs.SharedScaling,
                            HasReluConv1x1          = Globs.HasReluConv1x1,
                            UsePerFilterMagBalancing = Globs.UsePerFilterMagBalancing,
                            UseZMI                  = Globs.UseZMI,
                            InputHasBias            = Globs.InputHasBias,
                            ClassHasBias            = Globs.ClassHasBias,
                            ClassHasScaling         = Globs.ClassHasScaling,
                            ConvNoiseScale          = Globs.ConvNoiseScale,
                            ConvNoiseShift          = Globs.ConvNoiseShift,
                            ConvNoiseMode           = Globs.ConvNoiseMode,
                            UseResBlockInpuNoise    = Globs.UseResBlockInpuNoise,
                            Conv1Noise              = Globs.Conv1Noise,
                            Conv2Noise              = Globs.Conv2Noise,
                            ScalingNoise            = Globs.ScalingNoise,
                            LRDict                  = Globs.GradLRDict,
                            UseWeightNorm           = Globs.UseWeightNorm,
                            ClassifierInitMagnitudeScale = Globs.ClassifierInitMagnitudeScale,
                            ConvInitMagnitudeScale  = Globs.ConvInitMagnitudeScale,
                            ScalingInitMagnitudeScale = Globs.ScalingInitMagnitudeScale,
                            Conv1x1InitMagnitudeScale = Globs.Conv1x1InitMagnitudeScale,
                            AdaptLR2Scaling         = Globs.AdaptLR2Scaling,
                            UseConvScaling          = Globs.UseConvScaling,
                            UseFilterL2Rescale      = Globs.UseFilterL2Rescale,
                            Conv2ZeroInit           = Globs.Conv2ZeroInit,
                            NonLinLay               = Globs.NonLinLay,
                            HasActivationEuclidNorm = Globs.HasActivationEuclidNorm,
                            )

    # alter config to half if selected AFTER creating the model as the model is constructed on cpu that does not support half and then changed to GPU which supports half
    if Globs.Half :
        Globs.DefaultNumpyType = 'float16'
        Globs.DefaultType = torch.float16
    else :
        Globs.DefaultNumpyType = 'float32'
        Globs.DefaultType = torch.float32
    Config.DefaultNumpyType = Globs.DefaultNumpyType
    Config.DefaultType = Globs.DefaultType

    # optional float16 model
    if Globs.Half :
        model.half()

    # create dictionary of scaling params (if any) indexed by its corresponding weight
    Globs.WeightScalingDict = CreateWeightScalingDict (model.WeightScaling)

    if UseMultiGPU :
        model = torch.nn.DataParallel(model)
    else :
        print ( 'Using GPU : ' + CudaDeviceStr )

        # construct model using weird direct access to class dictionary and the functions for different resnets that return a constructed class
        print ( 'Architecture is : ' + Globs.Net )

        # set model to the gpu
        model.to(Globs.Device)

    # create a TensorBoard writer for logging
    #Globs.TB = SummaryWriter()
    # add model to TensorBoard
    #Globs.TB.add_graph(model)

    MinibatchSize = args.batch_size

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if Globs.AugmentTraining :
        # original pad=4
        TrainLoadTransform = transforms.Compose ( [ transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize ] )
        #TrainLoadTransform = transforms.Compose ( [ transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, (14,0)), transforms.ToTensor(), normalize ] )
        #TrainLoadTransform = transforms.Compose ( [ transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize ] )
    else :
        TrainLoadTransform = transforms.Compose ( [ transforms.ToTensor(), normalize ] )

    Cifar10Path = os.environ['CIFAR10']
    TrainDataset = datasets.CIFAR10 ( root=Cifar10Path, train=True, transform=TrainLoadTransform, download=False )

    train_loader = torch.utils.data.DataLoader (    TrainDataset,
                                                    batch_size=MinibatchSize, shuffle=True,
                                                    #num_workers=args.workers, pin_memory=True
                                                    num_workers=0, pin_memory=True
                                               )
    TestLoadTransform = transforms.Compose ( [ transforms.ToTensor(), normalize ] )
    ValDataset = datasets.CIFAR10 ( root=Cifar10Path, train=False, transform=TestLoadTransform )

    val_loader = torch.utils.data.DataLoader(
        ValDataset,
        batch_size=MinibatchSize, shuffle=False,
        num_workers=0, pin_memory=True)
        #num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and put it on the GPU
    criterion = nn.CrossEntropyLoss().to(Globs.Device)

    if Globs.Half :
        model.half()
        criterion.half()

    # params with specified LR
    AllNamedParams = list ( model.named_parameters() )
    ParamsWithLR = []
    ParamsWithLRDictList = []

    if Globs.ScalingLR is None :
        Globs.ScalingLR = Globs.LR

    if Globs.BiasLR is None :
        Globs.BiasLR = Globs.LR 

    if Globs.ClassLR is None :
        Globs.ClassLR = Globs.LR

    if Globs.ClassBiasLR is None :
        Globs.ClassBiasLR = Globs.BiasLR

    if Globs.HasScaling or Globs.UseBatchNorm or Globs.UseConvScaling :
        Globs.ScalingParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), ScalingLayer.ScalingName, False, WithName=True )
        # conv1 is special case scaling param
        Globs.Conv1ScalingParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), 'Conv1Scaling', False, WithName=True )
        Globs.BNScalingParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), 'weight', False, ndim=1, WithName=True )
        Globs.ClassifierScalingParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), 'ClassifierScaling', False, WithName=True )
        Globs.RawScalingParams = [p[1] for p in Globs.ScalingParams]
        Globs.RawBNScalingParams = [p[1] for p in Globs.BNScalingParams]
        Globs.RawClassifierScalingParams = [p[1] for p in Globs.ClassifierScalingParams]
        Globs.RawConv1ScalingParams = [p[1] for p in Globs.Conv1ScalingParams]
        Globs.AllScalingParams = []

        if len(Globs.RawScalingParams) > 0 :
            if True :
                ScalingParamsDictList = [{ 'params' : Globs.RawScalingParams, 'lr':Globs.ScalingLR}]
            else :
                NetDepth = len(Globs.RawScalingParams)
                LRM1 = 0.4
                LRM2 = 1.0
                ScalingParamsDictList = []
                for (param,depth) in zip(Globs.RawScalingParams,range(len(Globs.RawScalingParams))) :
                    DepthFract = depth / NetDepth
                    ScalingParamsDictList += [{ 'params' : [param], 'lr' : Globs.ScalingLR*(LRM1 + DepthFract*(LRM2-LRM1)) }]

            ParamsWithLRDictList += ScalingParamsDictList
            Globs.AllScalingParams += Globs.RawScalingParams

        if len(Globs.RawBNScalingParams) > 0 :
            BNScalingParamsDictList = [{ 'params' : Globs.RawBNScalingParams, 'lr':Globs.ScalingLR}]
            ParamsWithLRDictList += BNScalingParamsDictList
            Globs.AllScalingParams += Globs.RawBNScalingParams
        if len(Globs.RawClassifierScalingParams) > 0 :
            # classifier scaling LR is adapted to any relative classifier LR vs net LR
            ClassifierScalingParamsDictList = [{ 'params' : Globs.RawClassifierScalingParams, 'lr':Globs.ClassLR }]
            ParamsWithLRDictList += ClassifierScalingParamsDictList
            Globs.AllScalingParams += Globs.RawClassifierScalingParams
        if len(Globs.RawConv1ScalingParams) > 0 :
            # classifier scaling LR is adapted to any relative classifier LR vs net LR
            Conv1ScalingParamsDictList = [{ 'params' : Globs.RawConv1ScalingParams, 'lr':Globs.ScalingLR }]
            ParamsWithLRDictList += Conv1ScalingParamsDictList
            Globs.AllScalingParams += Globs.RawConv1ScalingParams

    elif Globs.UseWeightNorm :
        Globs.ScalingParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), 'weight_g', False, WithName=True )
        Globs.RawScalingParams = [p[1] for p in Globs.ScalingParams]
        #ScalingParamsDictList = [ {'params' : p[1], 'lr':Globs.ScalingLR} for p in ScalingParams ]
        ScalingParamsDictList = [{ 'params' : Globs.RawScalingParams, 'lr':Globs.ScalingLR}]
        ParamsWithLRDictList += ScalingParamsDictList

    # gather any bias param including batch norm biases that are also called 'bias'
    if ( Globs.HasBias or Globs.InputHasBias or Globs.ClassHasBias or Globs.UseBatchNorm ) :

        BiasParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), 'bias', False, WithName=False )
        if len(BiasParams) > 0 :
            BiasParamsDictList = {'params' : BiasParams, 'lr':Globs.BiasLR}
            ParamsWithLRDictList += [BiasParamsDictList]

        ClassifierBiasParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), 'ClassifierBias', False, WithName=False )
        if len(ClassifierBiasParams) > 0 :
            ClassifierBiasParamsDictList = {'params' : ClassifierBiasParams, 'lr':Globs.ClassBiasLR }
            ParamsWithLRDictList += [ClassifierBiasParamsDictList]

    # classifier layer - param name is 'weight' and ndim=2
    ClassParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), 'weight', False, WithName=False, ndim=2 )
    #ClassParamsDictList = [ {'params' : p[1], 'lr':Globs.ClassLR} for p in ClassParams ]
    ClassParamsDictList = { 'params' : ClassParams, 'lr':Globs.ClassLR }
    ParamsWithLRDictList += [ClassParamsDictList]

    # conv parameters with default lr
    ConvParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), 'Conv1Weight', Exclude=False, WithName=False, ndim=4 )
    ConvParams += ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), 'weight', Exclude=False, WithName=False, ndim=4 )
    #ConvParams += ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), 'conv1_weight', Exclude=False, WithName=False, ndim=4 )
    #ConvParams += ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), 'conv2_weight', Exclude=False, WithName=False, ndim=4 )

    #ConvParamsDictList = [ {'params' : p, 'lr':Globs.LR } for p in ConvParams ]
    if True :  # all conv params have same LR
        ConvParamsDictList = [{ 'params' : ConvParams, 'lr':Globs.LR  }]
    else :
        # LR set according to depth increasing to 1 at deepest noting that conv params are ordered by layer depth
        NetDepth = len(ConvParams)
        LRM1 = 0.4
        LRM2 = 1.0
        ConvParamsDictList = []
        for (param,depth) in zip(ConvParams,range(len(ConvParams))) :
            DepthFract = depth / NetDepth
            ConvParamsDictList += [{ 'params' : [param], 'lr' : Globs.LR*(LRM1 + DepthFract*(LRM2-LRM1)) }]

    # OPTIONAL SKIP SCALING
    if ( Globs.HasSkipScaling ) :

        SkipScalingParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), 'skipscaling', False, WithName=False )
        if len(SkipScalingParams) > 0 :
            SkipScalingParamsDictList = { 'params' : SkipScalingParams, 'lr':0.01 }
            ParamsWithLRDictList += [SkipScalingParamsDictList]

    # optimiser with per paramete lr
    optimizer = torch.optim.SGD ( ConvParamsDictList + ParamsWithLRDictList, lr=args.lr, momentum=args.momentum, weight_decay=Globs.L2Decay )
    # BATCH NORM OPTION - all parameters have same learning rate in one param group
    #optimizer = torch.optim.SGD ( model.parameters(), args.lr,  momentum=args.momentum, weight_decay=args.l2decay )

    # collect conv weights params
    if not Globs.UseWeightNorm :
        pnamestr= "weight"
    else :
        pnamestr= "weight_v"

    Globs.AllConvWeightParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), pnamestr, False, ndim=4, WithName=True )
    Globs.ConvWeightParams = [ p for p in Globs.AllConvWeightParams if p[1].shape[2:]==(3,3) ]
    Globs.ResProjWeightParams = [ p for p in Globs.AllConvWeightParams if p[1].shape[2:]==(1,1) ]

    # first conv layer special
    Globs.Conv1WeightParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), 'Conv1Weight', False, ndim=4, WithName=True )
    Globs.ResConvWeightParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), 'weight', Exclude=False, WithName=True, ndim=4 )
    Globs.ClassifierWeightParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), "weight", False, ndim=2, WithName=True )

    # MODEL LOAD OPTION : resume takes precedent
    if Globs.Model is not None :
        ModelPath = Globs.ExpRoot + '/' + Globs.Model #+ '/CheckPoint.pt'
        if os.path.isfile ( ModelPath ):

            print("=> loading checkpoint '{}'".format(ModelPath))
            # note that the loaded file may be either a model file or a checkpoint file containing a model
            checkpoint = torch.load(ModelPath)
            args.last_epoch = 0
            if 'model_state_dict' in checkpoint :
                model.load_state_dict(checkpoint['model_state_dict'])
                print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
            else :
                model.load_state_dict(checkpoint)

            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            Globs.best_prec1 = 0.0  # checkpoint['best_prec1']

            # learning rate schedule for training - note last epoch is always set to -1 otherwise there is a daft error and nobody knows what this parameter is actually for
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR ( optimizer, milestones=Globs.Milestones, last_epoch=-1, gamma=Globs.GammaLR )

        else :
            print ('model file' + ModelPath + ' does not exist')
            sys.exit()

    # RESUME OPTION
    elif Globs.Resume :
        if os.path.isfile ( Globs.CheckpointPath ):

            print("=> loading checkpoint '{}'".format(Globs.CheckpointPath))
            checkpoint = torch.load(Globs.CheckpointPath)

            # learning rate schedule for training - note last epoch is always set to -1 otherwise there is a daft error and nobody knows what this parameter is actually for
            # NB : last_epoch means the last epoch that happened, i.e. the epoch in the previous training when it stopped, so the scheduler needs to start from previous_run_last_epoch+1, hence the daftly named last_epoch param
            # - also note that if you set the last_epoch correctly to the last one in the resumed run then a bug in PyTorch claims initi_lr not set. So the workaround is to change this last_epoch as part of load_state_dict
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR ( optimizer, milestones=Globs.Milestones, last_epoch=-1, gamma=Globs.GammaLR )

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            # set the last_epoch which is the first epoch of the next run
            args.last_epoch = checkpoint['epoch'] + 1

            Globs.best_prec1 = 0.0  # checkpoint['best_prec1']

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}'".format(ModelPath))
    else :
    # TRAIN MODEL FROM SCRATCH OPTION
        # learning rate schedule for training - note last epoch is always set to -1 otherwise there is a daft error and nobody knows what this parameter is actually for
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR ( optimizer, milestones=Globs.Milestones, last_epoch=-1, gamma=Globs.GammaLR )

    # optionally validate and exit
    if Globs.ValidateOnly :
        (ValPrec,ValLoss) = validate ( val_loader, model, criterion )
        print('val precision=%4.3f, val loss=%4.3f' % (ValPrec, ValLoss) )
        sys.exit()

    ##########################
    # DEBUG PRINT OUT SCALING
    ##########################
    if Globs.DebugPrintParams :
        if Globs.HasScaling and False :
            # extract parameter weight tensor from list of named params
            ScalingParams = [p[1].data for p in Globs.ScalingParams]

            if Globs.UseExpoScaling :
                ScalingParams = [ p.exp() for p in ScalingParams]

            print ( ScalingParams[2] )

        #print ( Globs.ConvWeightParams[60])
        #print ( Globs.ScalingParams[81])
        #x = torch.min ( Globs.ConvWeightParams[1] ).item()
        #print(x)

        #################
        # FILTER WEIGHTS
        #################
        # extract parameter weight tensor from list of named params
        LocalConvWeightParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), "weight", False, ndim=4, WithName=True )
        LocalConvWeightParams += ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), "conv1_weight", False, ndim=4, WithName=True )
        LocalConvWeightParams += ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), "conv2_weight", False, ndim=4, WithName=True )
        Globs.RawConvWeightParams = [p[1].data for p in LocalConvWeightParams]
        ConvMinsParamString = ''
        ConvMaxsParamString = ''
        ConvMeansParamString = ''
        ConvEuclidMeanParamString = ''
        ConvEuclidMaxParamString = ''
        ConvEuclidMinParamString = ''
        ConvLiveFractString = ''
        ConvLiveFract = []
        ConvL1Means = []
        ConvL1Mean = 0.0
        # loop over all conv layers parameters (filter,depth,k_y,k_x)
        for idx in range (len(Globs.RawConvWeightParams)) :
            #WeightTransform = lambda x : torch.tanh(x/0.4)*0.4
            #WeightTransform = lambda x : torch.tanh(x/0.25)*0.25
            #WeightTransform = lambda x : torch.tanh(x/0.2)*0.2
            #WeightTransform = lambda x : torch.tanh(x)
            WeightTransform = lambda x : x

            ThisWeight = WeightTransform ( Globs.RawConvWeightParams[idx] )
            ConvL1Means.append (ThisWeight.abs().mean())
            ConvL1Mean += ConvL1Means[-1]
            ConvMinsParamString += ' '+'{:1.2f}'.format ( torch.min(ThisWeight.abs()) )
            ConvMaxsParamString += ' '+'{:1.2f}'.format ( torch.max(ThisWeight.abs()) )
            ConvMeansParamString += ' '+'{:1.2f}'.format ( torch.mean(ThisWeight.abs()) )
            ConvEuclidMeanParamString += ' '+'{:1.2f}'.format ( torch.mean(ThisWeight.pow(2).sum(dim=(1,2,3)).sqrt()) )
            ConvEuclidMinParamString += ' '+'{:1.2f}'.format ( torch.min(ThisWeight.pow(2).sum(dim=(1,2,3)).sqrt()) )
            ConvEuclidMaxParamString += ' '+'{:1.2f}'.format ( torch.max(ThisWeight.pow(2).sum(dim=(1,2,3)).sqrt()) )

            # fractional live filters by layer defined by fract of filters in the layer with Euclide magnitude > 0.2
            LiveFractThresh=0.15
            PerFilterMag = ThisWeight.pow(2).sum(dim=(1,2,3)).sqrt()
            LivePerFilterMagMask = PerFilterMag.gt(LiveFractThresh).float()
            FractLivePerFilter = LivePerFilterMagMask.mean()
            ConvLiveFractString += ' '+'{:1.2f}'.format(FractLivePerFilter)
            ConvLiveFract.append ( FractLivePerFilter )

        ConvL1Mean /= float(len(ConvL1Means))
        print('CONV MIN PARAMS')
        print ( ConvMinsParamString )
        print('CONV MAX PARAMS')
        print ( ConvMaxsParamString )
        print('CONV MEAN PARAMS')
        print ( ConvMeansParamString )
        print('CONV EUCLID MIN')
        print ( ConvEuclidMinParamString )
        print('CONV EUCLID MAX')
        print ( ConvEuclidMaxParamString )
        print('CONV EUCLID MEAN')
        print ( ConvEuclidMeanParamString )
        print('FILTER LIVE FRACT MEAN')
        print ( ConvLiveFractString )
        # print ( 'Mean live fract = ' + str(np.mean([t.numpy() for t in ConvLiveFract])))
        ConvLiveFractSum=0.0
        for idx in range (len(ConvLiveFract)) :
            ConvLiveFractSum += ConvLiveFract[idx]
        print ( 'Mean live fract = ' + str ( ConvLiveFractSum/float(len(ConvLiveFract)) ) )

        # BIAS
        BiasParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), "Bias", False, WithName=False )
        if len(BiasParams)>0 :
            BiasMinsParamString = ''
            BiasMaxsParamString = ''
            BiasMeansParamString = ''
            BiasL1Means = []
            BiasL1Mean = 0.0
            for idx in range (len(BiasParams)) :
                ThisBias = BiasParams[idx]

                BiasL1Means.append (torch.mean(ThisBias))
                BiasL1Mean += BiasL1Means[-1]
                BiasMinsParamString += ' '+'{:1.2f}'.format ( torch.min(ThisBias).item() )
                BiasMaxsParamString += ' '+'{:1.2f}'.format ( torch.max(ThisBias).item() )
                BiasMeansParamString += ' '+'{:1.2f}'.format ( torch.mean(ThisBias).item() )
            BiasL1Mean /= float(len(BiasL1Means))
            print('BIAS MINS PARAMS')
            print ( BiasMinsParamString )
            print('BIAS MAXS PARAMS')
            print ( BiasMaxsParamString )
            print('BIAS MEANS PARAMS')
            print ( BiasMeansParamString )

        # TRELU BIAS
        if False :
            TreluBiasParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), "TreluBias", False, WithName=False )
            if len(TreluBiasParams)>0 :
                TreluBiasMinsParamString = ''
                TreluBiasMaxsParamString = ''
                TreluBiasMeansParamString = ''
                TreluBiasL1Means = []
                TreluBiasL1Mean = 0.0
                for idx in range (len(TreluBiasParams)) :
                    ThisTreluBias = TreluBiasParams[idx]

                    TreluBiasL1Means.append (torch.mean(ThisTreluBias))
                    TreluBiasL1Mean += TreluBiasL1Means[-1]
                    TreluBiasMinsParamString += ' '+'{:1.2f}'.format ( torch.min(ThisTreluBias).item() )
                    TreluBiasMaxsParamString += ' '+'{:1.2f}'.format ( torch.max(ThisTreluBias).item() )
                    TreluBiasMeansParamString += ' '+'{:1.2f}'.format ( torch.mean(ThisTreluBias).item() )
                TreluBiasL1Mean /= float(len(TreluBiasL1Means))
                print('TRELU BIAS MINS PARAMS')
                print ( TreluBiasMinsParamString )
                print('TRELU BIAS MAXS PARAMS')
                print ( TreluBiasMaxsParamString )
                print('TRELU BIAS MEANS PARAMS')
                print ( TreluBiasMeansParamString )

        # CLASSIFIER
        ClassifierWeightParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), "weight", False, ndim=2, WithName=True )
        # output euclidean magnitude for each class
        ClassEuclidMagParamString = ''
        ClassMag = ClassifierWeightParams[0][1].detach().pow(2).sum(dim=(1)).sqrt().cpu().numpy()
        for idx in range (len(ClassMag)) :
            ClassEuclidMagParamString += ' '+'{:1.2f}'.format ( ClassMag[idx] )
        print('CLASSIFIER EUCLID MAG')
        print ( ClassEuclidMagParamString )

        # CLASSIFIER SCALING
        if Globs.ClassHasScaling :
            ClassifierScalingParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), "ClassifierScaling", False, WithName=True )
            ClassifierScalingNP = ClassifierScalingParams[0][1].detach().cpu().numpy()
            if Globs.UseExpoScaling :
                ClassifierScalingNP = np.exp(ClassifierScalingNP)

            ClassifierScalingString = ''
            for idx in range (len(ClassifierScalingNP)) :
                ClassifierScalingString += ' '+'{:1.2f}'.format ( ClassifierScalingNP[idx] )

            print('CLASSIFIER SCALING')
            print ( ClassifierScalingString )

        # CLASSIFIER BIAS
        if Globs.ClassHasBias :
            ClassifierBiasParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), "ClassifierBias", False, WithName=True )
            ClassifierBiasNP = ClassifierBiasParams[0][1].detach().cpu().numpy()

            ClassifierBiasString = ''
            for idx in range (len(ClassifierBiasNP)) :
                ClassifierBiasString += ' '+'{:1.2f}'.format ( ClassifierBiasNP[idx] )

            print('CLASSIFIER BIAS')
            print ( ClassifierBiasString )

        # SCALING
        ScalingLayerParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), "scaling", False, WithName=False )
        ScalingBNParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), "weight", False, ndim=1, WithName=False )
        ScalingParams = ScalingLayerParams + ScalingBNParams
        if len(ScalingParams) > 0 :
            ScalingMinsParamString = ''
            ScalingMaxsParamString = ''
            ScalingMeansParamString = ''
            ScalingL1Means = []
            ScalingL1Mean = 0.0
            for idx in range (len(ScalingParams)) :
                if Globs.UseExpoScaling :
                    ThisScaling = ScalingParams[idx].exp()
                else :
                    ThisScaling = ScalingParams[idx]

                ScalingL1Means.append (torch.mean(ThisScaling))
                ScalingL1Mean += ScalingL1Means[-1]
                ScalingMinsParamString += ' '+'{:1.2f}'.format ( torch.min(ThisScaling).item() )
                ScalingMaxsParamString += ' '+'{:1.2f}'.format ( torch.max(ThisScaling).item() )
                ScalingMeansParamString += ' '+'{:1.2f}'.format ( torch.mean(ThisScaling).item() )
            ScalingL1Mean /= float(len(ScalingL1Means))
            print('SCALING MINS PARAMS')
            print ( ScalingMinsParamString )
            print('SCALING MAXS PARAMS')
            print ( ScalingMaxsParamString )
            print('SCALING MEANS PARAMS')
            print ( ScalingMeansParamString )

        # EFFECTIVE LEARNING RATE - residual branch conv layers
        if len(ScalingLayerParams) > 0 and len(LocalConvWeightParams) > 0 :
            assert len(ScalingLayerParams) == len(LocalConvWeightParams)

            ELRMinsParamString = ''
            ELRMaxsParamString = ''
            ELRMeansParamString = ''
            for idx in range (len(ScalingLayerParams)) :
                if Globs.UseExpoScaling :
                    ThisScaling = ScalingParams[idx].exp()
                else :
                    ThisScaling = ScalingParams[idx]

                #WeightTransform = lambda x : torch.tanh(x/0.2)*0.2
                #WeightTransform = lambda x : torch.tanh(x)
                WeightTransform = lambda x : x
                ThisWeight = WeightTransform ( LocalConvWeightParams[idx][1].data )
                
                ThisConvWeightMag = ThisWeight.pow(2).sum(dim=(1,2,3)).sqrt()

                ThisELR = ThisScaling / ThisConvWeightMag
                ELRMinsParamString += ' '+'{:1.2f}'.format ( torch.min(ThisELR).item() )
                ELRMaxsParamString += ' '+'{:1.2f}'.format ( torch.max(ThisELR).item() )
                ELRMeansParamString += ' '+'{:1.2f}'.format ( torch.mean(ThisELR).item() )
            print('ELR MINS PARAMS')
            print ( ELRMinsParamString )
            print('ELR MAXS PARAMS')
            print ( ELRMaxsParamString )
            print('ELR MEANS PARAMS')
            print ( ELRMeansParamString )

        # SKIP SCALING
        SkipScalingParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), "skipscaling", False, WithName=False )
        if len(SkipScalingParams) > 0 :
            SkipScalingMinsParamString = ''
            SkipScalingMaxsParamString = ''
            SkipScalingMeansParamString = ''
            SkipScalingL1Means = []
            SkipScalingL1Mean = 0.0
            for idx in range (len(SkipScalingParams)) :
                ThisScaling = SkipScalingParams[idx].exp()

                SkipScalingL1Means.append (torch.mean(ThisScaling))
                SkipScalingL1Mean += SkipScalingL1Means[-1]
                SkipScalingMinsParamString += ' '+'{:1.2f}'.format ( torch.min(ThisScaling).item() )
                SkipScalingMaxsParamString += ' '+'{:1.2f}'.format ( torch.max(ThisScaling).item() )
                SkipScalingMeansParamString += ' '+'{:1.2f}'.format ( torch.mean(ThisScaling).item() )
            SkipScalingL1Mean /= float(len(SkipScalingL1Means))
            print('SKIP SCALING MINS PARAMS')
            print ( SkipScalingMinsParamString )
            print('SKIP SCALING MAXS PARAMS')
            print ( SkipScalingMaxsParamString )
            print('SKIP SCALING MEANS PARAMS')
            print ( SkipScalingMeansParamString )

        # conv and scaling statistics
        if Globs.HasScaling and False :
            print('MODEL STATS')
            print ( "ConvL1Mean [" + '{:1.3f}'.format ( ConvL1Mean ) + '] : ScalingL1Mean [' + '{:1.3f}'.format(ScalingL1Mean) + '] : Conv*Scaling [' + '{:1.3f}'.format(ConvL1Mean*ScalingL1Mean) + ']' )
            # per layer
            ProdParamString = ''
            for idx in range (len(ConvL1Means)) :
                ProdParamString += ' '+'{:5.3}'.format ( ConvL1Means[idx]*ScalingL1Means[idx] )
                #RatioParamString += ' '+'{:>6}'.format ( ConvL1Means[idx]*ScalingL1Means[idx] )
            print ( ProdParamString )

            # per layer effective LR
            LRString = ''
            print('LR STATS - not normalised')
            for idx in range (len(ConvL1Means)) :
                LRString += ' '+'{:3.1f}'.format ( ScalingL1Means[idx] / ConvL1Means[idx] )
                #RatioParamString += ' '+'{:>6}'.format ( ConvL1Means[idx]*ScalingL1Means[idx] )
            print ( LRString )

        # CONV1 FILTER MAGNITUDE
        if len(Globs.Conv1WeightParams) > 0 :
            Conv1EuclidMagParamString = ''
            Conv1Mag = Globs.Conv1WeightParams[0][1].detach().pow(2).sum(dim=(1,2,3)).sqrt().cpu().numpy()
            for idx in range (len(Conv1Mag)) :
                Conv1EuclidMagParamString += ' '+'{:1.2f}'.format ( Conv1Mag[idx] )
            print('CONV1 EUCLID MAG')
            print ( Conv1EuclidMagParamString )

        # CONV1 SCALING
        Conv1ScalingParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), "Conv1Scaling", False, WithName=True )
        if len(Conv1ScalingParams) > 0 :
            Conv1ScalingNP = Conv1ScalingParams[0][1].detach().cpu().numpy()
            if Globs.UseExpoScaling :
                Conv1ScalingNP = np.exp(Conv1ScalingNP)

            Conv1ScalingString = ''
            for idx in range (len(Conv1ScalingNP)) :
                Conv1ScalingString += ' '+'{:1.2f}'.format ( Conv1ScalingNP[idx] )

            print('CONV1 SCALING')
            print ( Conv1ScalingString )

        # CONV1 BIAS
        if False :  # coding not completed : layer name must be debugged and scaling replaced by bias
            Conv1BiasParams = ParamUtil.GetFromNamedParamsByName ( model.named_parameters(), "Bias", False, WithName=True, LayerName="Conv1" )
            if len(Conv1ScalingParams) > 0 :
                Conv1ScalingNP = Conv1ScalingParams[0][1].detach().cpu().numpy()
                if Globs.UseExpoScaling :
                    Conv1ScalingNP = np.exp(Conv1ScalingNP)

                Conv1ScalingString = ''
                for idx in range (len(Conv1ScalingNP)) :
                    Conv1ScalingString += ' '+'{:1.2f}'.format ( Conv1ScalingNP[idx] )

                print('CONV1 BIAS')
                print ( Conv1ScalingString )

        # VALIDATE TO ENSURE MODEL IS CORRECT
        if Globs.PreValidate :
            (ValPrec,ValLoss) = validate ( val_loader, model, criterion )
            print ( "PRECISION ["+str(ValPrec)+"] : LOSS ["+str(ValLoss)+']')

        sys.exit()
    #########################################################################################

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # CREATE TENSORBOARD WRITER - this is for all results logging to Globs.LogDir under the TensorBoard logging directory so that the experiment can be observed in real-time
    Globs.Log = TensorBoardLogWriter (Globs.LogDir)
    # create all of the logger streams
    Globs.Log.AddStream ("TrainStepLoss")
    Globs.Log.AddStream ("TrainStepPrec")
    Globs.Log.AddStream ("TrainProgMeanLoss")
    Globs.Log.AddStream ("TrainProgMeanPrec")
    Globs.Log.AddStream ("TrainProgMaxLoss")
    Globs.Log.AddStream ("TrainProgMaxPrec")
    Globs.Log.AddStream ("TrainProgMinLoss")
    Globs.Log.AddStream ("TrainProgMinPrec")
    Globs.Log.AddStream ("ValLoss")
    Globs.Log.AddStream ("ValPrec")

    # PRE-WARM before training at low learning rate
    if Globs.PreWarm!=0 :
        PreWarm ( train_loader, model, criterion, optimizer, PreWarmLR=Globs.LR/10.0, NumEpochs=Globs.PreWarm )

    if Globs.Resume :
        FirstEpochIdx = args.last_epoch
    else :
        FirstEpochIdx=0

    LastEpochIdx = Globs.NumEpochs - 1

    # PRE-VALIDATE to record initial loss
    if Globs.PreValidate :
        (ValPrec,ValLoss) = validate ( val_loader, model, criterion )
        Globs.Log.AddToLog ( 'ValLoss', ValLoss )
        Globs.Log.AddToLog ( 'ValPrec', ValPrec)

    # log the initial model for repeatability of training runs
    if not Globs.Resume :
        torch.save ( model.state_dict(), Globs.ModelLogDir + '/ModelEp0.pt' )

        # also save this in the root directory as the most recent model
        torch.save({
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            }, Globs.ExpLogDir + '/CheckPoint.pt')

    # REMOVE_ME: temp code
    if False :
        ResetLowMagitudeFilters ( FilterBankList=Globs.ConvWeightParams, Thresh=Globs.FilterEuclidMagResetThresh, Randomise=Globs.RandomiseFilterReset, ResetThreshIsClamp=Globs.ResetThreshIsClamp)

        # APPLY FILTER EUCLIDEAN MAGNITUDE RESET (if enabled) to any filter whose magnitude is less than the pre-set threshold
        if Globs.FilterEuclidMagResetThresh > 0 :
            ResetLowMagitudeFilters ( FilterBankList=Globs.ResProjWeightParams, Thresh=Globs.FilterEuclidMagResetThresh, Randomise=Globs.RandomiseFilterReset )
            if Globs.UseBatchNorm :
                ResetLowMagitudeFiltersConvBN ( FilterBankList=model.ConvBN, Thresh=Globs.FilterEuclidMagResetThresh, Randomise=Globs.RandomiseFilterReset, ResetMag=1.0, ResetThreshIsClamp=Globs.ResetThreshIsClamp )
            else :
                print("#### CHECK ME or EXTEND ResetLowMagitudeFilters to include conv layers with separate scaling when not UseBatchNorm")
                # note - currently anything but clamp will not be valid so need to pass corresponding
                ResetLowMagitudeFilters ( FilterBankList=Globs.ConvWeightParams, Thresh=Globs.FilterEuclidMagResetThresh, Randomise=Globs.RandomiseFilterReset )

    if False :

        # OPTIONALLY FLIP scaling and filter kernel signs if scaling is -ve
        # note - in this model each conv3x3 layer has a BN constructed in order so the scalings and conv filters have the same position in their respective lists
        if Globs.FlipNegFilterAndScalingSign :
            FlipFiltersWithNegScaling ( model.ConvBN )
        
    if False :
        for epoch in range(100) :
            lr_scheduler.step()
            AnnealL2DecayToLR ( optimizer.param_groups, Globs.L2Decay )
            pass

    if False :
        if Globs.FilterEuclidMagResetThresh > 0 :
            if len(Globs.ResProjWeightParams)>0 :
                ResetLowMagitudeFilters ( FilterBankList=Globs.ResProjWeightParams, Thresh=Globs.FilterEuclidMagResetThresh, Randomise=Globs.RandomiseFilterReset )
            if Globs.UseBatchNorm :
                ResetLowMagitudeFiltersConvBN ( FilterBankList=model.ConvBN, Thresh=Globs.FilterEuclidMagResetThresh, Randomise=Globs.RandomiseFilterReset, ResetMag=1.0, ResetThreshIsClamp=Globs.ResetThreshIsClamp )
            else :
                ResetLowMagitudeFilters ( FilterBankList=Globs.ConvWeightParams, Thresh=Globs.FilterEuclidMagResetThresh, Randomise=Globs.RandomiseFilterReset, ResetThreshIsClamp=Globs.ResetThreshIsClamp, WeightScalingDict=Globs.WeightScalingDict )

    # initialize running mean model inflation (scalar) to zero for each conv filter
    if Globs.MIPScaling>0 :
        Globs.RunningMIP = []
        for param in Globs.ConvWeightParams :
            # ThisMIP = torch.zeros ( [param[1].data.shape[0]], dtype=Config.DefaultType ).to(Config.CurrentDevice)
            ThisMIP = torch.zeros_like ( param[1].data )
            Globs.RunningMIP.append ( ThisMIP )

    # running means for ELR loss
    Globs.RunningFilterScale = [None] * len(Globs.ResConvWeightParams)

    # record all trainable params in Globs for easy access
    Globs.TrainableParams = ParamUtil.TrainableParams ( model )

    # previous gradients if used - stored in globs so persistent between epochs
    if Globs.DualMinibatch :
        Globs.PrevGrads = None

    # TRAIN/LOOP OVER SPECIFIED NUMBER OF EPOCHS
    for epoch in range ( FirstEpochIdx, LastEpochIdx+1 ):

        # train for one epoch
        #print('current lr {:2.6f}'.format(optimizer.param_groups[0]['lr']))
        Globs.ShowLR = False
        if Globs.ShowLR :
            lrs = [group['lr'] for group in optimizer.param_groups]
            lrs_formatted = ['{:2.6f}'.format(lr) for lr in lrs]
            print('current lrs :' + str(lrs_formatted) )

        # output 16 groups of training stats in every epoch (except for last one which may be short)
        NumSamplesInEpoch = len(train_loader.dataset.data)
        MinibatchSize = train_loader.batch_size
        NumMinibatches = NumSamplesInEpoch // MinibatchSize
        NumProgressGroupsPerEpoch = 16
        if Globs.OutputEvery>0 :
            NumMinibatchesInProgressGroup = Globs.OutputEvery
        else :
            NumMinibatchesInProgressGroup = NumMinibatches // NumProgressGroupsPerEpoch

        # perform one training step comprising a minibatch in one epoch within single function
        EpochStart = time.time()
        TrainOneEpoch ( train_loader, model, criterion, optimizer, epoch, StatsOutputEvery=NumMinibatchesInProgressGroup, IsFirstEpoch=(epoch==FirstEpochIdx), DualMinibatch=Globs.DualMinibatch )
        EpochEnd = time.time()

        # update the LR according to schedule
        lr_scheduler.step()

        # optionally manually adjust the L2decay in proportion to the LR reduction - this is a counter-deflation measure
        if Globs.AnnealL2Decay :
            AnnealL2DecayToLR ( optimizer.param_groups, Globs.L2Decay )

        # VALIDATE
        # if BatchNorm is used then perform 20 forward steps so that the batch norm parameters can catch up with the model. Note that only the forward pass if performed so this is very quick and does not interfere with training.
        # The validation dataset is only used to update the minibatch stats as firstly it is smaller (therefore quicker to shuffle and load) and secondly as no learning happens and thirdly as it is more representative of the validation stats
        if Globs.UseBatchNorm and Globs.ValidationPreWarmNumSteps>0 :
            ApplyPreValidationProtocol ( Model=model, NumSteps=Globs.ValidationPreWarmNumSteps, Data=val_loader )

        ( ValPrec, ValLoss ) = validate ( val_loader, model, criterion )

        # log these to TensorBoard
        Globs.Log.AddToLog ( 'ValLoss', ValLoss )
        Globs.Log.AddToLog ( 'ValPrec', ValPrec )

        # remember best prec@1 and save checkpoint
        is_best = ValPrec > Globs.best_prec1

        # console output end of epoch summary
        TotalSteps = (epoch+1)*NumMinibatches
        EpochDurationInSec = EpochEnd - EpochStart
        OuputValidationToConsole ( ValLoss, ValPrec, Globs.best_prec1, EpochDurationInSec, epoch, TotalSteps, TotalSteps, TotalSteps, MinibatchSize, TrainingPrecision=Globs.EpochRunningPrecisionSum / float(NumMinibatches) )

        # record best precision
        Globs.best_prec1 = max(ValPrec, Globs.best_prec1)

        # APPLY FILTER EUCLIDEAN MAGNITUDE RESET (if enabled) to any filter whose magnitude is less than the pre-set threshold
        # note - the filter magnitude is not transferred to the scaling (if present). For that use -euclidnormxfer on each step.
        if Globs.FilterEuclidMagResetThresh > 0 :
            if len(Globs.ResProjWeightParams)>0 :
                ResetLowMagitudeFilters ( FilterBankList=Globs.ResProjWeightParams, Thresh=Globs.FilterEuclidMagResetThresh, Randomise=Globs.RandomiseFilterReset )
            if Globs.UseBatchNorm :
                ResetLowMagitudeFiltersConvBN ( FilterBankList=model.ConvBN, Thresh=Globs.FilterEuclidMagResetThresh, Randomise=Globs.RandomiseFilterReset, ResetThreshIsClamp=Globs.ResetThreshIsClamp )
            else :
                ResetLowMagitudeFilters ( FilterBankList=Globs.ConvWeightParams, Thresh=Globs.FilterEuclidMagResetThresh, Randomise=Globs.RandomiseFilterReset, ResetThreshIsClamp=Globs.ResetThreshIsClamp, WeightScalingDict=Globs.WeightScalingDict )

        # OPTIONALLY FLIP scaling and filter kernel signs if scaling is -ve
        # note - in this model each conv3x3 layer has a BN constructed in order so the scalings and conv filters have the same position in their respective lists
        if Globs.FlipNegFilterAndScalingSign and Globs.UseBatchNorm :
            FlipFiltersWithNegScaling ( model.ConvBN )

        # log the model per epoch by epoch index
        torch.save ( model.state_dict(), Globs.ModelLogDir + '/ModelEp' + str(epoch+1) +'.pt' )

        # CHECKPOINT EVERY EPOCH FOR RESUMING IF TRAINING INTERRUPTED - it is overwritten
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            }, Globs.ExpLogDir + '/CheckPoint.pt')

    #### END OF TRAINING ####
    torch.cuda.empty_cache()

    # close TensorBoard writer
    Globs.Log.Close()


if __name__ == '__main__':

    # deal with command line parameters
    parser = SetupCommandLineArgs ()
    args = parser.parse_args()

    # check that the log directory does not exist and create it - if it does exist then exit. Note this is a path underneath the TensorBoard log directory which is in the environment
    Globs.TBLOG = os.environ['TBLOG']
    Globs.ExpRoot = os.environ['EXPLOGDIR']

    # for validation only there is no logdir
    if args.logdir is not None :
        Globs.LogDir = Globs.TBLOG + "/" + args.logdir
        Globs.ExpLogDir = Globs.ExpRoot + '/' + args.logdir
        Globs.CheckpointPath = Globs.ExpLogDir + '/CheckPoint.pt'
        Globs.ModelLogDir = Globs.ExpLogDir + '/ModelLog'

    if args.resume :
        # check that the checkpoint file exists
        assert os.path.exists(Globs.LogDir), "RESUME : no checkpoint file exists at " + Globs.CheckpointPath

    elif args.validate :
        pass

    else :

        if os.path.exists(Globs.LogDir):
            assert args.overwrite, "The directory %s already exists or is invalid so cannot create or overwrite the output directory. Please check and if necessary manually delete this directory as overwritting is not allowed for safety" % ( args.logdir )
            shutil.rmtree(Globs.LogDir)
            sleep(0.01)
        # attempt to create the output directory - makedirs does not return anything, I guess just excepts
        os.makedirs(Globs.LogDir)

        # store the command line in the output directory for reference - it is very good practice to do this so that the experimental conditions are recorded
        if os.path.exists(Globs.ExpLogDir):
            assert args.overwrite, "The directory %s already exists or is invalid so cannot create or overwrite the output directory. Please check and if necessary manually delete this directory as overwritting is not allowed for safety" % ( args.logdir )
            shutil.rmtree(Globs.ExpLogDir)
            sleep(0.01)
        # attempt to create the output directory - makedirs does not return anything, I guess just excepts
        os.makedirs(Globs.ExpLogDir)
        # directory for model log under the main experiment directory
        Globs.ModelLogDir = Globs.ExpLogDir+'/ModelLog'
        os.makedirs(Globs.ModelLogDir)

        # store command line to log directory 
        ComFile = open ( Globs.ExpLogDir + '/CommandLine.txt', 'wt', encoding='utf-8' )
        for arg in sys.argv :
            ComFile.write ( arg + '\n')  
        ComFile.close ()

    # convert args to Globs settings
    ProcessCommandLineArgs ( Globs, args )

    # main code entry point
    main()
