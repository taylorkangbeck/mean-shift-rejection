import torch
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import numpy as np
from functools import partial

from Common import *

def TensorConstantInit ( Tensor, Value ) :
    Tensor[...] = torch.ones ( Tensor.shape, dtype=Tensor.dtype) * Value
    return Tensor

def RandUniformInit ( Tensor, Low=0.001, High=0.995 ) :
    """
    Both in place updates the passed tensor and also returns the tensor so both method call approaces can be used
    """
    Tensor.uniform_ ( Low, High )
    return Tensor

def BatchNormInit ( BN_Module, ScalingInit=None ) :
    """
    Initialises the bias and scaling of a batch norm layer to (0,1) whereas PyTorch strangely initializes the scaling to a uniform distribution !!
    """
    # just guarantee biases are zero
    BN_Module.bias.data[...] = torch.zeros ( BN_Module.weight.shape, dtype=Config.DefaultType)

    if ScalingInit is None :
        # default init
        BN_Module.weight.data[...] = torch.ones ( BN_Module.weight.shape, dtype=Config.DefaultType)
    else :
        BN_Module.weight.data[...] = ScalingInit ( BN_Module.weight.data )

    #if ScalingInitLow==ScalingInitHigh :
    #    BN_Module.weight.data[...] = torch.ones ( BN_Module.weight.shape, dtype=Config.DefaultType) * ScalingInitLow
    #else :
    #    # scaling init linear range for initial LR and error asymmetry in aggregate parameter that can adjust rapidly
    #    # note - in this case PyTorch fills the tensor in place so operate directly on the data tensor i.e. not the parameter
    #    BN_Module.weight.data.uniform_ ( ScalingInitLow, ScalingInitHigh )    #.to(Globs.GPU,non_blocking=True)
    #    pass
    pass

def LinearWeightInit ( ParamData, Method, UseZMI=False, UsePerFilterMagBalancing=False, InitMagnitudeScale=None ) :
    """
    Linear layer weight parameter initialisation method taking parameters :
   
        Method                      : passed method that initializes the weights parameter e.g. to xavier or uniform distribution
        UseZMI                      : if true then each filter kernel is zero mean shifted per 2D channel
        UsePerFilterMagBalancing     : ?
        FilterInitMagnitudeScale    : if specified then the scale of each filter is re-normalised to this scale


    """

    # initialise weights in-place using the callback
    Method (ParamData )

    # optionally each filter kernel is zero mean shifted per 2D channel
    if UseZMI :
        ParamData -= torch.mean ( ParamData, dim=(1,), keepdim=True )

    # optionally each filter is re-scaled to the mean of the scale of the filter bank so that all filters have the same Euclidean magnitude as random init leaves filters with different scales which is noticeable for small fan-in filters
    if UsePerFilterMagBalancing :
        PerFilterEuclidMag = ParamData.pow(2).sum(dim=(1,),keepdim=True).sqrt()
        MeanFilterMag = PerFilterEuclidMag.mean ( keepdim=True )
        # rescale each unity mag filter to the mean magnitude of the filter bank
        ParamData /= PerFilterEuclidMag * MeanFilterMag

    # optionally each filter is rescaled to the specified Euclidean scale e.g. if originally Xavier or He with mag of 1 then rescaled to W' = W/mag(W)*FilterInitMagnitudeScale
    if InitMagnitudeScale :
        ParamData *= ( InitMagnitudeScale / ParamData.pow(2).sum(dim=(1,),keepdim=True).sqrt() )

def ConvParamLikeXavierUniformInit ( Param, Mag=1.0 ) :
    """
    Uniform random param init with magnitude set to Mag*Mag(Xavier(param)) .
    """

    ParamData = Param.data
    ParamShape = ParamData.shape
    AllFilterDims = tuple( range(0,len(ParamShape)) )
    PerFilterDims = AllFilterDims[1:]

    # compute mean Xavier init i.e. the mean magnitude of the filter bank with Xavier init
    init.xavier_normal_(ParamData)

    if False :
        XavierMean = ParamData.pow(2).sum(dim=PerFilterDims,keepdim=False).sqrt().mean()

        ParamData.uniform_ ( 0.0, 1.0 )
        PerFilterEuclidMag = ParamData.pow(2).sum(dim=PerFilterDims,keepdim=True).sqrt()

        # rescale each unity mag filter to the mean magnitude of the filter bank
        ParamData *= ( XavierMean * Mag / PerFilterEuclidMag )
    else :
        L1_All = torch.mean ( torch.abs(ParamData), dim=(0,1,2,3), keepdim=True )
        # remake the weights as uniform with the same L1
        UniformWeightData  = np.random.uniform ( -1.0, 1.0, size=ParamData.shape ).astype('float32')
        UniformWeightData -= np.mean ( UniformWeightData, axis=(2,3), keepdims=True )
        UniformWeightData = torch.from_numpy(UniformWeightData).to(Config.CurrentDevice)
        L1_PerFilter = torch.mean ( torch.abs(UniformWeightData), dim=(1,2,3), keepdim=True )
        L1Scale = L1_All / L1_PerFilter
        ParamData[...] = UniformWeightData * L1Scale

    pass

def ConvParamUniformInit ( Param, Low=-1.0, High=1.0 ) :

    ParamData = Param.data
    ParamShape = ParamData.shape
    AllFilterDims = tuple( range(0,len(ParamShape)) )
    PerFilterDims = AllFilterDims[1:]

    ParamData.uniform_ ( Low, High )
    PerFilterEuclidMag = ParamData.pow(2).sum(dim=PerFilterDims,keepdim=True).sqrt()

    # rescale each unity mag filter to the mean magnitude of the filter bank
    #ParamData *= ( Mag / PerFilterEuclidMag )

    pass

def ConvParamUniformMaxNormInit ( Param, Mag=1.0 ) :

    ParamData = Param.data
    ParamShape = ParamData.shape
    AllFilterDims = tuple( range(0,len(ParamShape)) )
    PerFilterDims = AllFilterDims[1:]

    # uniform distribution across all filters
    ParamData.uniform_ ( 0.0, 1.0 )

    # rescale each unity mag filter to the max magnitude coefficient and further rescaled to Mag
    KernelAxes = tuple ( range ( 1, len(ParamData.shape) ) )
    PerFiltMax = torch.max ( ParamData.abs().reshape( (ParamData.shape[0],-1) ), dim=1 )
    PerFiltMax = PerFiltMax.reshape ( (ParamData.shape[0],) * ( len(ParamData.shape) - 1 ) )
    KernelNumParams = np.prod(Param.shape[1:])
    ParamData *= ( PerFiltMax / np.sqrt(KernelNumParams) / Mag )

    pass

def ConvWeightInit ( ParamData, Method, UseZMI=False, UsePerFilterMagBalancing=False, InitMagnitudeScale=None, InitMaxScale=None ) :
    """
    Convolutional layer weight parameter initialisation method taking parameters :
   
        Method                      : passed method that initializes the weights parameter e.g. to xavier or uniform distribution
        UseZMI                      : if true then each filter kernel is zero mean shifted per 2D channel followed by rebalancing the filter magnitude back to its original value
        UsePerFilterMagBalancing     : ?
        FilterInitMagnitudeScale    : if specified then the scale of each filter is re-normalised to this scale


    """

    # initialise weights in-place using the callback - note ParamData is not differentiable
    Method ( ParamData )

    # dims for each filter and all filters
    ParamShape = ParamData.shape
    AllFilterDims = tuple( range(0,len(ParamShape)) )
    PerFilterDims = AllFilterDims[1:]

    # note - 1x1 conv cannot have zmi per channel but could have per filter. For now zmi is simply disabled for 1x1 convolutions
    #UseZMI = UseZMI and ParamShape[-1] != 1

    # optionally each filter is re-scaled to the mean of the scale of the filter bank so that all filters have the same Euclidean magnitude as random init leaves filters with different scales which is noticeable for small fan-in filters
    if UsePerFilterMagBalancing :
        PerFilterEuclidMag = ParamData.pow(2).sum(dim=PerFilterDims,keepdim=True).sqrt()
        MeanFilterMag = PerFilterEuclidMag.mean(dim=AllFilterDims,keepdim=True)
        # rescale each unity mag filter to the mean magnitude of the filter bank
        ParamData *= ( MeanFilterMag / PerFilterEuclidMag )

    # optionally each filter kernel is zero mean shifted per 2D channel and the filter magnitude is restored to the pre- meanshifted
    if UseZMI :
        # record per filter Euclidean magnitude before the per channel mean subtraction so that this magnitude can be restored
        PerFilterEuclidMag = ParamData.pow(2).sum(dim=PerFilterDims,keepdim=True).sqrt()

        # PER CHANNEL - alternatively if 1x1 conv then per filter
        if ParamShape[-1] == 1 :
            ParamData -= torch.mean ( ParamData, dim=(1,), keepdim=True )
        else :
            ParamData -= torch.mean ( ParamData, dim=(2,3), keepdim=True )

        # mean subtracted mag
        PerFilterEuclidMagMS = ParamData.pow(2).sum(dim=PerFilterDims,keepdim=True).sqrt()

        # restore the per filter magnitude
        ParamData *= ( PerFilterEuclidMag / PerFilterEuclidMagMS )

    # optionally each filter is rescaled to the specified Euclidean scale e.g. if originally Xavier or He with mag of 1 then rescaled to W' = W/mag(W)*InitMagnitudeScale
    if InitMagnitudeScale is not None :
        ParamData *= ( InitMagnitudeScale / ParamData.pow(2).sum(dim=PerFilterDims,keepdim=True).sqrt() )

    # optionally rescale the filter to the max value in each filter
    if InitMaxScale is not None :
        ParamShape = ParamData.shape
        AllFilterDims = tuple( range(0,len(ParamShape)) )
        PerFilterDims = AllFilterDims[1:]
        KernelAxes = tuple ( range ( 1, len(ParamData.shape) ) )
        PerFiltMax = torch.max ( ParamData.abs().reshape( (ParamData.shape[0],-1) ), dim=1 )[0]
        MaxBroadcastShape = (ParamData.shape[0],) + (1,) * ( len(ParamData.shape) - 1 )
        PerFiltMax = PerFiltMax.reshape ( MaxBroadcastShape )
        KernelNumParams = np.prod(ParamShape[1:])
        ParamData *= ( InitMaxScale / PerFiltMax / np.sqrt(KernelNumParams) )

    pass

