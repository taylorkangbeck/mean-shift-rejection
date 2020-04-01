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

from torch.utils.tensorboard import SummaryWriter
from TensorBoardLogWriter import TensorBoardLogWriter

import itertools
from AttributeDict import AttributeDict
import ParamUtil

from PerMapBiasLayer import PerMapBiasLayer
from PerMapScalingLayer import PerMapScalingLayer
from PerMapRandomScalingLayer import enumNoiseMode

# imports common functions directly into this namespace for readability
from Common import *

import TorchConfig as Config

import torch.nn.functional as F

model_names = sorted(name for name in ResnetBiasScaling.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(ResnetBiasScaling.__dict__[name]))

# globals
class Globs () :
    UseZeroMeanGrads    = False
    best_prec1          = 0.0

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
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('-zmg', dest='zmg', type=float, default=0.0, help='mean gradient kernel stabilisation fraction : default is zero which is no stabilisation')
parser.add_argument('-preval', dest='preval', action='store_true', help='validate the initial model before training')
parser.add_argument('-perchannel', dest='perchannel', action='store_true', help='selects per channel zero mean i.e. over dim=(2,3) of the grads tensor for conv filter')
parser.add_argument('-l2actloss', dest='l2actloss', type=float, default=0.0, help='L2 activation loss scaling applied to the ouput of convolutional layers regardless of their kernel shape')
parser.add_argument('-filterscaleloss', dest='filterscaleloss', type=float, default=0.0, help='convolution kernel magnitude difference from unity loss scaling applied to the ouput of convolutional layers regardless of their kernel shape')
parser.add_argument('-filterl2scaleloss', dest='filterl2scaleloss', type=float, default=0.0, help='convolution kernel magnitude difference from unity loss scaling applied to the ouput of convolutional layers regardless of their kernel shape')

parser.add_argument('-prewarm', default=0, type=int, help='number of steps in pre-warm : default=0')
parser.add_argument('-ope', default=0, type=int, help='forces console output on the defined number of steps with a default of 0 which indicates 16 outputs per minibatch')
parser.add_argument('-overwrite', dest='overwrite', action='store_true', help='CAUTION : setting this overwrites any existing log directory files')
parser.add_argument('-tvmodel', dest='tvmodel', action='store_true', help='Use torchvision model where available')
parser.add_argument('-noaug', dest='noaug', action='store_true', help='do not augment training using random flip and crop : default is to augment')

parser.add_argument('-batchnorm', dest='batchnorm', action='store_true', help='')
parser.add_argument('-bias', dest='bias', action='store_true', help='')
parser.add_argument('-scaling', dest='scaling', action='store_true', help='')
parser.add_argument('-exposcaling', dest='exposcaling', action='store_true', help='')
parser.add_argument('-zmi', dest='zmi', action='store_true', help='')
parser.add_argument('-zmw', default=0.0, type=float, help='if specified then per per channel L1 zero mean weight decay is applied')
parser.add_argument('-perfiltL1scaling', dest='perfiltL1scaling', action='store_true', help='')
parser.add_argument('-inputhasbias', dest='inputhasbias', action='store_true', help='')
parser.add_argument('-classhasbias', dest='classhasbias', action='store_true', help='')
parser.add_argument('-classhasscaling', dest='classhasscaling', action='store_true', help='')

parser.add_argument('-noise', default=0.0, type=float, help='Random uniform modulation of the output of each convolutional layer at the raw conv output after any random noise shift : default is 0 (no noise) and a typical value is 0.1')
parser.add_argument('-noiseshift', default=0.0, type=float, help='Random uniform shift in of the output of each convolutional layer at the raw conv output : default is 0 (no noise) and a typical value is 0.1')
parser.add_argument('-noisemode', default=enumNoiseMode.Full.value, type=str, help='Mode for noise is half or full Gaussian')
parser.add_argument('-biaslr', default=None, type=float, help='if specified then the bias has this lr separately to all other params lr')
parser.add_argument('-scalinglr', default=None, type=float, help='if specified then the scaling has this lr separately to all other params lr')
parser.add_argument('-classlr', default=None, type=float, help='if specified then the classifier has this lr separately to all other params lr')
parser.add_argument('-lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('-lrgamma', default=0.1, type=float, help='learning rate decay in optimiser schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('-l2decay', default=0.0, type=float, help='weight decay (default:0, typical is 5e-4)')

parser.add_argument('-silentstats', dest='silentstats', action='store_true', help='stops training stats progress bars so only epoch validation is shown')
parser.add_argument('-showtiming', dest='showtiming', action='store_true', help='output timing of each fwd-bwd pass to console')
parser.add_argument('-schedpoints', nargs='+', type=int, default=[50,75], help='Learning rate annealing schedule : default = '+str('[50,75]') )
parser.add_argument('-schedrate', nargs='+', type=float, default=[0.1,0.1], help='Learning rate annealing schedule : default = '+str('[0.1,0.1]') )
parser.add_argument('-validate', dest='validate', action='store_true', help='perform a validation without training')

parser.add_argument('-euclidnormxfer', dest='euclidnormxfer', action='store_true', help='')
parser.add_argument('-adaptconvlr', dest='adaptconvlr', action='store_true', help='')
parser.add_argument('-weightnorm', dest='weightnorm', action='store_true', help='applies PyTorch inbuilt weight norm to 3x3 conv layers so that separate scaling is not needed, and if LRAdapt is used then this is applied using the weight now scaling')
parser.add_argument('-filterl2rescale', dest='filterl2rescale', action='store_true', help='conv filters output are rescaled by the Euclidean norm of the filter kernels i.e. y = conv(x,Wi) / EuclidNorm(Wi)')
parser.add_argument('-adaptbiaslr', dest='adaptbiaslr', action='store_true', help='if -filterl2rescale is active then the bias gradients are adjusted to 1/mag(filter)')

# NOT YET USED
parser.add_argument('-weightscaling', default=1.0, type=float, help='')

parser.add_argument('-save', dest='save_dir', help='The directory used to save the trained models', default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every', help='Saves checkpoints at every specified number of epochs', type=int, default=10)

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
    # and hide it as it is not in use yet
    #StatsGroupProgressBar.Show()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.to(Globs.GPU,non_blocking=True)

        with torch.no_grad() :
            input_var = torch.autograd.Variable(input).to(Globs.GPU)
            target_var = torch.autograd.Variable(target)    #.to(Globs.GPU)

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

        del input_var
        del target_var

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
        return np.asscalar( res[0] )
    else :
        return res

def OuputValidationToConsole ( Loss, Prec, BestPrec, ElapsedTime, Epoch, Iteration, MinibatchIdx, NumMinibatches, MinibatchSize ) :
        """ Output validation loss and error to console"""
        print('epoch %i, test acc %4.3f, val loss %4.3f, took %3.2f min, %4.1f ms per minbatch, and %4.3f ms per sample' % (
                                Epoch+1, Prec, Loss,
                                ElapsedTime / 60.0,
                                ElapsedTime * 1000.0 / NumMinibatches,
                                ElapsedTime * 1000.0 / MinibatchSize / NumMinibatches ) )

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
            Param.grad *= GradLRDict[ParamName].data

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
                    ThisParam.grad -= ParamMeanGrad * MeanMult
            else :
                debug11 = 0

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

            target = target.to(Globs.GPU,non_blocking=True)
            input_var = torch.autograd.Variable(input).to(Globs.GPU)
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
                AllParams = model.parameters()
                ConvParams = GetConvParams ( AllParams )
                ZeroMeanNormaliseGrads ( ConvParams, PerChannel=Globs.PerChannel, MeanMult=Globs.ZeroMeanGrad )

            # perform parameter update step using parameter gradients
            optimizer.step()

            TrainStatsGroupProgressBar.Update()

    # restore the learning rate noting that 1 epoch has been lost from the optimiser
    optimizer.param_groups[0]['lr'] = ParamGroupOrigLR

def TrainOneEpoch ( train_loader, model, criterion, optimizer, EpochIdx, StatsOutputEvery ) :
    """
    Train all the minibatches in an epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

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
        input = input_cpu.to(Globs.GPU,non_blocking=True)
        target = target_cpu.to(Globs.GPU,non_blocking=True)
        input_var = torch.autograd.Variable(input).to(Globs.GPU)
        target_var = torch.autograd.Variable(target).to(Globs.GPU)      # CHECK ME

        if Globs.Half :
            input_var = input_var.half().to(Globs.GPU)      # CHECK ME

        forward_start = time.time()

        # compute output - note that all layers also have their outputs stored in model.Layers[] which is a list of tuples of (name, instance, input, output, tags )
        # The model now returns an ordered list of layers in the order of activation as layer info tuples so the last output is the classifier output
        AllOutputs = model(input_var)
        output = AllOutputs[-1].Out

        loss = criterion(output, target_var)
        forward_duration = time.time() - forward_start

        ############################
        # add any auxiliary losses
        ############################

        # compute gradient and do SGD step
        backward_start = time.time()
        # reset the gradients to zero (is this really necessary ?)
        optimizer.zero_grad()
        # compute gradients - note that there is a parameter loss.backward(retain_variables=True) : does this keep all the gradients ?
        loss.backward()
        backward_duration = time.time() - backward_start

        # DEBUG - print duration of fwd and bwd passes
        if Globs.ShowFwdBwdTiming :
            print ( "fwd time=%4.3f    bwd time=%4.3f" % (forward_duration, backward_duration) )

        # optionally modify the gradients to be zero mean for convolutional params
        if Globs.ZeroMeanGrad > 0.0 :
            ZeroMeanNormaliseGrads ( Globs.ConvWeightParams, PerChannel=Globs.PerChannel, MeanMult=Globs.ZeroMeanGrad )

        # APPLY PER PARAMETER LEARNING RATE ADJUSTMENT TO GRADIENTS
        # note - GradLRDict is indexed by a parameter and if a param is in the dictionary then its grad is multiplied by the tensor it indexes
        # firstly create tuple of (weight,grad) to associate the param's tensor with its gradient for the conv param whose tensor is referred to by p.weight

        # perform parameter update step using parameter gradients
        optimizer.step()

        # convert graph outputs to Python floats for recording and display
        output = output.float()
        loss = loss.float()

        # measure and record accuracy
        ThisMinibatchPrec = ComputePrecision(output.data, target)
        ThisMinibatchLoss = np.asscalar(loss.data)
        losses.update ( ThisMinibatchLoss, MinibatchSize )
        top1.update(ThisMinibatchPrec, MinibatchSize )

        # log these to TensorBoard
        Globs.Log.AddToLog ( 'TrainStepLoss', ThisMinibatchLoss )
        Globs.Log.AddToLog ( 'TrainStepPrec', ThisMinibatchPrec )

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

        # TORCH CLEANUP to prevent ever slowing training
        #del input_var
        #del target_var

    # end of epoch - remove training stats
    if TrainStatsGroupProgressBar.Visible and not Globs.SilentTrainingStats :
        TrainStatsGroupProgressBar.Hide()

    # TORCH CLEANUP to prevent ever slowing training
    #del TrainLoadDraw
    #torch.cuda.empty_cache()


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

    # create empty LR dictionary which is used in the model to setup any per parameter LR adjustment
    Globs.GradLRDict = {}

    # choose net
    if Globs.UsePyTorchModel :
        model = models.resnet101()

    else :
        # create the CNN according to the function whose name corresponds to the model name
        ModelFunc = ResnetBiasScaling.__dict__[Globs.Net]
        model = ModelFunc ( UseBatchNorm            = Globs.UseBatchNorm,
                            HasBias                 = Globs.HasBias,
                            HasScaling              = Globs.HasScaling,
                            UseExpoScaling          = Globs.UseExpoScaling,
                            UsePerFilterL1Rescaling = Globs.UsePerFilterL1Rescaling,
                            UseZMI                  = Globs.UseZMI,
                            InputHasBias            = Globs.InputHasBias,
                            ClassHasBias            = Globs.ClassHasBias,
                            ClassHasScaling         = Globs.ClassHasScaling,
                            ConvNoiseScale          = Globs.ConvNoiseScale,
                            ConvNoiseShift          = Globs.ConvNoiseShift,
                            ConvNoiseMode           = Globs.ConvNoiseMode,
                            LRDict                  = Globs.GradLRDict,
                            UseWeightNorm           = Globs.UseWeightNorm,
                            UseFilterL2Rescale      = Globs.UseFilterL2Rescale,
                            )

    if UseMultiGPU :
        model = torch.nn.DataParallel(model)
    else :
        device = torch.device(CudaDeviceStr)
        print ( 'Using GPU : ' + CudaDeviceStr )

        # construct model using weird direct access to class dictionary and the functions for different resnets that return a constructed class
        print ( 'Architecture is : ' + Globs.Net )

        # set model to the gpu
        model.to(device)

        Globs.GPU = device

    # record the device in global as there is no way find out the current device (at least not in the documentation though no doubt it is there somewhere)
    Globs.Device = device
    Config.CurrentDevice = device

    # create a TensorBoard writer for logging
    #Globs.TB = SummaryWriter()
    # add model to TensorBoard
    #Globs.TB.add_graph(model)

    MinibatchSize = args.batch_size

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if Globs.AugmentTraining :
        TrainLoadTransform = transforms.Compose ( [ transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize ] )
    else :
        TrainLoadTransform = transforms.Compose ( [ transforms.ToTensor(), normalize ] )

    Cifar10Path = 'F:/CIFAR/CIFAR10/'
    TrainDataset = datasets.CIFAR10 ( root=Cifar10Path, train=True, transform=TrainLoadTransform, download=True )

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
    criterion = nn.CrossEntropyLoss().to(Globs.GPU)

    if Globs.Half :
        model.half()
        criterion.half()

    # BATCH NORM OPTION - all parameters have same learning rate in one param group
    Globs.optimizer = torch.optim.SGD ( model.parameters(), args.lr,  momentum=args.momentum, weight_decay=args.l2decay )
    #optimizer = torch.optim.Adam ( OptParams, args.lr, weight_decay=args.l2decay )

    #Globs.Milestones = [70, 90]
    Globs.Milestones = [100, 150]

    # MODEL LOAD OPTION : resume takes precedent
    if Globs.Model is not None :
        ModelPath = Globs.ExpRoot + '/' + Globs.Model + '/CheckPoint.pt'
        if os.path.isfile ( ModelPath ):
            print("=> loading checkpoint '{}'".format(ModelPath))

            checkpoint = torch.load(ModelPath)
            CheckPointEpoch = checkpoint['epoch']
            CheckPointStateDict = checkpoint["model_state_dict"]

            model.load_state_dict ( CheckPointStateDict )
            print ( "=> loaded checkpoint (epoch {})".format(CheckPointEpoch) )

            args.start_epoch = 0
            Globs.best_prec1 = 0.0  # checkpoint['best_prec1']

            # learning rate schedule for training - note last epoch is always set to -1 otherwise there is a daft error and nobody knows what this parameter is actually for
            Globs.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR ( Globs.optimizer, milestones=Globs.Milestones, last_epoch=-1, gamma=Globs.GammaLR )
        else :
            print ('#### model path ' + ModelPath + ' does not exist #####' )
            sys.exit()

    elif Globs.Resume :
        # RESUME OPTION
        if os.path.isfile ( Globs.CheckpointPath ):

            print("=> loading checkpoint '{}'".format(Globs.CheckpointPath))
            checkpoint = torch.load(Globs.CheckpointPath)
            args.start_epoch = checkpoint['epoch']

            # learning rate schedule for training - note last epoch is always set to -1 otherwise there is a daft error and nobody knows what this parameter is actually for
            # NB : last_epoch means the last epoch that happened, i.e. the epoch in the previous training when it stopped, so the scheduler needs to start from previous_run_last_epoch+1, hence the daftly named last_epoch param
            Globs.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR ( Globs.optimizer, milestones=Globs.Milestones, last_epoch=args.start_epoc-1, gamma=Globs.GammaLR )

            model.load_state_dict(checkpoint['model_state_dict'])
            Globs.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            Globs.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            Globs.best_prec1 = 0.0  # checkpoint['best_prec1']

            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))

        else:
            print("=> no checkpoint found at '{}'".format(ModelPath))
    else :
        # TRAIN MODEL FROM SCRATCH OPTION
        # learning rate schedule for training - note last epoch is always set to -1 otherwise there is a daft error and nobody knows what this parameter is actually for
        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR ( optimizer, milestones=[100, 150], last_epoch=-1, gamma=Globs.GammaLR )
        Globs.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR ( Globs.optimizer, milestones=Globs.Milestones, last_epoch=-1, gamma=Globs.GammaLR )

        # optional Euclidean filter norm xfer to scaling xfer(weights,scaling) - this changes the balance of filter magnitude and scaling once at the beginning of training
        if Globs.UsePerFilterL1Rescaling :
            # xfer weights magnitude to scaling - note both weights and scalings are in same order due to construction order and each 3x3 conv layer must have a scaling
            if Globs.HasScaling :
                FilterEuclidNormXferToScaling ( Globs.ConvWeightParams, Globs.ScalingParams, ExpoScaling=Globs.UseExpoScaling )

            if Globs.ClassHasScaling :
                FilterEuclidNormXferToScaling ( Globs.ClassifierWeightParams, Globs.ClassifierScalingParams, ExpoScaling=Globs.UseExpoScaling )

    # optionally validate and exit
    if Globs.ValidateOnly :
        (ValPrec,ValLoss) = validate ( val_loader, model, criterion )
        print('val precision=%4.3f, val loss=%4.3f' % (ValPrec, ValLoss) )
        sys.exit()

    # DEBUG PRINT OUT SCALING
    if Globs.DebugPrintParams :
        if Globs.HasScaling :
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
        Globs.RawConvWeightParams = [p[1].data for p in LocalConvWeightParams]
        ConvMinsParamString = ''
        ConvMaxsParamString = ''
        ConvMeansParamString = ''
        ConvEuclidMeanParamString = ''
        ConvEuclidMaxParamString = ''
        ConvEuclidMinParamString = ''
        ConvL1Means = []
        ConvL1Mean = 0.0
        for idx in range (len(Globs.RawConvWeightParams)) :
            ConvL1Means.append (torch.mean(Globs.RawConvWeightParams[idx].abs()))
            ConvL1Mean += ConvL1Means[-1]
            ConvMinsParamString += ' '+'{:1.2f}'.format ( torch.min(Globs.RawConvWeightParams[idx].abs()) )
            ConvMaxsParamString += ' '+'{:1.2f}'.format ( torch.max(Globs.RawConvWeightParams[idx].abs()) )
            ConvMeansParamString += ' '+'{:1.2f}'.format ( torch.mean(Globs.RawConvWeightParams[idx].abs()) )
            ConvEuclidMeanParamString += ' '+'{:1.2f}'.format ( torch.mean(Globs.RawConvWeightParams[idx].pow(2).sum(dim=(1,2,3)).sqrt()) )
            ConvEuclidMinParamString += ' '+'{:1.2f}'.format ( torch.min(Globs.RawConvWeightParams[idx].pow(2).sum(dim=(1,2,3)).sqrt()) )
            ConvEuclidMaxParamString += ' '+'{:1.2f}'.format ( torch.max(Globs.RawConvWeightParams[idx].pow(2).sum(dim=(1,2,3)).sqrt()) )

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

        if Globs.HasScaling :
            ScalingMinsParamString = ''
            ScalingMaxsParamString = ''
            ScalingMeansParamString = ''
            ScalingL1Means = []
            ScalingL1Mean = 0.0
            for idx in range (len(ScalingParams)) :
                ScalingL1Means.append (torch.mean(ScalingParams[idx].abs()))
                ScalingL1Mean += ScalingL1Means[-1]
                ScalingMinsParamString += ' '+'{:1.2f}'.format ( torch.min(ScalingParams[idx]).item() )
                ScalingMaxsParamString += ' '+'{:1.2f}'.format ( torch.max(ScalingParams[idx]).item() )
                ScalingMeansParamString += ' '+'{:1.2f}'.format ( torch.mean(ScalingParams[idx]).item() )
            ScalingL1Mean /= float(len(ScalingL1Means))
            print('SCALING MINS PARAMS')
            print ( ScalingMinsParamString )
            print('SCALING MAXS PARAMS')
            print ( ScalingMaxsParamString )
            print('SCALING MEANS PARAMS')
            print ( ScalingMeansParamString )

        # conv and scaling statistics
        # model
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

        sys.exit()

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    # CREATE TENSORBOARD WRITE - this is for all results logging to Globs.LogDir under the TensorBoard logging directory so that the experiment can be observed in real-time
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
        PreWarm ( train_loader, model, criterion, Globs.optimizer, PreWarmLR=Globs.LR/10.0, NumEpochs=Globs.PreWarm )

    FirstEpochIdx = args.start_epoch
    LastEpochIdx = args.epochs - 1

    # PRE-VALIDATE to record initial loss
    if Globs.PreValidate :
        (ValPrec,ValLoss) = validate ( val_loader, model, criterion )
        Globs.Log.AddToLog ( 'ValLoss', ValLoss )
        Globs.Log.AddToLog ( 'ValPrec', ValPrec)

    # log the initial model for repeatability of training runs
    if not Globs.Resume :
        torch.save ( model.state_dict(), Globs.ModelLogDir + '/ModelEp0.pt' )

    # DEBUG TEST LR_SCHEDULER
    #for epoch in range(200) :
    #    lr_scheduler.step()
    #    lrs = [group['lr'] for group in optimizer.param_groups]
    #    lrs_formatted = ['{:2.6f}'.format(lr) for lr in lrs]
    #    print('current lrs :' + str(lrs_formatted) )

    # TRAIN/LOOP OVER SPECIFIED NUMBER OF EPOCHS
    for epoch in range ( FirstEpochIdx, LastEpochIdx+1 ):

        # train for one epoch
        #print('current lr {:2.6f}'.format(optimizer.param_groups[0]['lr']))
        lrs = [group['lr'] for group in Globs.optimizer.param_groups]
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
        TrainOneEpoch ( train_loader, model, criterion, Globs.optimizer, epoch, StatsOutputEvery=NumMinibatchesInProgressGroup )
        EpochEnd = time.time()

        # update the LR according to schedule
        Globs.lr_scheduler.step()

        # evaluate on validation set
        ( ValPrec, ValLoss ) = validate ( val_loader, model, criterion )

        # log these to TensorBoard
        Globs.Log.AddToLog ( 'ValLoss', ValLoss )
        Globs.Log.AddToLog ( 'ValPrec', ValPrec )

        # remember best prec@1 and save checkpoint
        is_best = ValPrec > Globs.best_prec1

        # console output end of epoch summary
        TotalSteps = (epoch+1)*NumMinibatches
        EpochDurationInSec = EpochEnd - EpochStart
        OuputValidationToConsole ( ValLoss, ValPrec, Globs.best_prec1, EpochDurationInSec, epoch, TotalSteps, TotalSteps, TotalSteps, MinibatchSize )

        # record best precision
        Globs.best_prec1 = max(ValPrec, Globs.best_prec1)

        # log the model per epoch by epoch index
        torch.save ( model.state_dict(), Globs.ModelLogDir + '/ModelEp' + str(epoch+1) +'.pt' )

        # CHECKPOINT EVERY EPOCH FOR RESUMING IF TRAINING INTERRUPTED - it is overwritten
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': Globs.optimizer.state_dict(),
            'scheduler_state_dict': Globs.lr_scheduler.state_dict(),
            }, Globs.ExpLogDir + '/CheckPoint.pt')

    #### END OF TRAINING ####

    # close TensorBoard writer
    Globs.Log.Close()

if __name__ == '__main__':

    # deal with command line parameters
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

    Globs.DebugPrintParams      = args.debugprintparams
    Globs.Net                   = args.arch
    Globs.NumEpochs             = args.epochs
    Globs.Resume                = args.resume
    Globs.ValidateOnly          = args.validate
    Globs.Model                 = args.model
    Globs.ZeroMeanGrad          = args.zmg
    Globs.PreValidate           = args.preval
    Globs.PerChannel            = args.perchannel
    Globs.PreWarm               = args.prewarm
    Globs.SchedPointsLR         = args.schedpoints
    Globs.SchedRateLR           = args.schedrate
    Globs.GammaLR               = args.lrgamma
    Globs.LR                    = args.lr
    Globs.BiasLR                = args.biaslr
    Globs.ScalingLR             = args.scalinglr
    Globs.ClassLR               = args.classlr
    Globs.OutputEvery           = args.ope
    Globs.UsePyTorchModel       = args.tvmodel
    Globs.AugmentTraining       = not args.noaug
    Globs.UseBatchNorm          = args.batchnorm
    Globs.HasBias               = args.bias
    Globs.HasScaling            = args.scaling
    Globs.UseExpoScaling        = args.exposcaling
    Globs.UsePerFilterL1Rescaling = args.perfiltL1scaling
    Globs.UseZMI                = args.zmi
    Globs.InputHasBias          = args.inputhasbias
    Globs.ClassHasBias          = args.classhasbias
    Globs.ClassHasScaling       = args.classhasscaling
    Globs.ConvNoiseScale        = args.noise
    Globs.ConvNoiseShift        = args.noiseshift
    Globs.ConvNoiseMode         = enumNoiseMode.fromstring ( args.noisemode )
    Globs.SilentTrainingStats   = args.silentstats
    Globs.ShowFwdBwdTiming      = args.showtiming
    Globs.ZMW                   = args.zmw
    Globs.L2ActLossScaling      = args.l2actloss
    Globs.UseFilterEuclidNormXferToScaling = args.euclidnormxfer
    Globs.AdaptConvLR           = args.adaptconvlr
    Globs.UseWeightNorm         = args.weightnorm
    Globs.UseFilterL2Rescale    = args.filterl2rescale
    Globs.FilterScaleLoss       = args.filterscaleloss
    Globs.FilterL2ScaleLoss     = args.filterl2scaleloss
    Globs.AdaptBiasLR2FilterMag = args.adaptbiaslr

    # main code entry point
    main()

