
from collections import namedtuple
from collections import OrderedDict
from enum import Enum, IntEnum, unique
from functools import partial

import torch
import torch.tensor as T
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

#
# TODO: PyTorch has no infrastructure for hooking layers together informatively that supports the purpose of nodes in the graph. It is planned to extend PyTorch with a graph support infrastructure for easy
#       hook up of layers as with the previous Theano implementation but in the case of PyTorch the declarative nature of graph activation makes this process simpler. In particular an active graph parsing class
#       may be passed the graph input (e.g. RGB image) as a tensor and call the computational sub-graph of each layer in turn storing the output tensor within the computational graph as a node.
#       Then as with the Theano system the tags associated with a node may be used to filter and query for subsets of nodes for side losses and debug purposes. This will include also tagging of layer parameters.
#
#       As a very temporary measure a simple LayerInfo tuple is provided to return the ouput of the forward() method with tags to support filtering for auxiliary activation losses.
#       This approach requires that each module outputs a single map tupLayerInfo.out so a block with more than one output should be constructured using a Python method programmatically.
#       Note that this supports only single inputs to a block though in principle that could be extended with kwargs for as many as wished.
#
#       The programming pattern is :
#
#           self.Layers.append ( HookUpModule ( name='conv1', module=self.conv1, input=x, tags=['conv'] ) )
#           self.Layers.append ( HookUpModule ( name='conv2', module=self.conv2, input=self.Layers[-1].Out, tags=['conv'] ) )
#           ..
#

class EnumBase ( Enum ) :

    @classmethod
    def fromstring ( cls, thestr) :
        """ Return the enum corresponding to the passed string """
        #lut = dict([(cls._member_map_[key].value,key) for key in cls._member_map_])
        #ret = getattr(cls, lut[thestr], None)
        #return ret
        # case insensitive version
        #lut = dict([(str.lower(key),cls._member_map_[key].value) for key in cls._member_map_])
        lut = dict([(str.lower(key),key) for key in cls._member_map_])
        ret = getattr(cls, lut[thestr.lower()], None)
        return ret

    @classmethod
    def tostring ( cls, type=None ) :
        """ Return the string corresponding to the passed enum value """
        if type == None :
            return cls.value
        else :
            return type.value

"""
tupLayerInfo    : return for layer(i.e. module) forward method to return more than just a tensor and instead returns info including tags so that the output of the layer may be used for side lossses filtered by the tags.

Name        : str                   : given name of layer
Layer       : module based layer    : the instantiated layer
FanInShape  : tuple of ints         : shape dimensions of FanIn e.g. conv layer has FanInShape = (kern_d, kern_h, kern_w)
InShape     : tuple of ints         : shape of the input tensor to the layer
OutShape    : tuple of ints         : shape of the output tensor of the layer
Tags        : list of str           : list of tag strings relating to the purpose of the layer e.g. ['conv', 'encoder']
"""
tupLayerInfo = namedtuple ( "tupLayerInfo", "Name Layer In Out Tags" )

class AttributeDict(dict):
    __getattr__ = dict.__getitem__

    def __setattr__(self, a, b):
        self.__setitem__(a, b)

"""
Lambda layer wraps the passed function lambd as a PyTorch module aka layer so operates much like a Python lambda as a lightweight layerless layer. Note that kwargs are passed to func by converting it into a partial.
"""
class LambdaLayer ( nn.Module ):
    def __init__( self, func, **kwargs ) :
        super(LambdaLayer, self).__init__()
        self.func = partial ( func, **kwargs )

    def forward(self, x):
        return self.func(x)

class LambdaLayerNoArgs ( nn.Module ):
    def __init__( self, func, **kwargs ) :
        super(LambdaLayerNoArgs, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def HookUpLayer ( name, module, input, tags=[], **kwargs ) :
    """
        Calls the passed module's computational graph (i.e. forward() ) on the input and returns a tuple tupLayerInfo which includes the layer's output and tags and other info
        such as the input and name for auxiliary losses and visualisation and debugging.

        The reasons this is a method rather than using the tuple directly is to allow the operation to be extended or changed without affecting the calling code.
    """
    return tupLayerInfo ( name, module, input, module(input, **kwargs), tags )

def HookUpLayers ( name, modules, input, tags=[], **kwargs ) :
    """
        Calls the passed module's computational graph (i.e. forward() ) on the input and returns a tuple tupLayerInfo which includes the layer's output and tags and other info
        such as the input and name for auxiliary losses and visualisation and debugging.

        The reasons this is a method rather than using the tuple directly is to allow the operation to be extended or changed without affecting the calling code.
    """
    RetList = []

    LastOutput = input
    for (module,idx) in zip(modules,range(len(modules))) :
        # call the default function, here forward(), for each module (aka layer) which returns a list of its layer outputs as tuples where each module has a list of layerinfo tuples (potentially of length 1)
        ThisModuleLayerInfoList = module ( LastOutput, **kwargs )
        LastOutput = ThisModuleLayerInfoList[-1].Out
        RetList += ThisModuleLayerInfoList

    return RetList

def PutBlockListIntoModuleDict ( ModuleSelf, BlockList ) :
    """
    Adds each block (multiple layers within an nn.module) into the module list of the class which must be derived from nn.module. This is to support the weight initialisation.
    Note that a module may have many layers in its class vars and all are picked up by the base nn.module class.
    """
    for block in BlockList :
        ModuleSelf.add_module ( block.Name, block )

def PutLayerListIntoClassDict ( ClassSelf, BlockList ) :
    """
    Puts each layer (based on module) in LayersList into the dictionary of the passed class noting that the list of layers may be nested i.e. with sub-lists/
    """
    for layer in LayersList :
        if type(layer) is type([]) :
            PutLayersIntoClassDict ( ClassSelf, layer )
        else :
            ClassSelf.add_module ( layer.Name, layer.Layer )

def HasAllTags ( Tags, QueryTags ) :
    """
    Returns True is all query tags are in the tags which is a list of tags. Note if no tags are specified then returns True.
    """
    ret = True
    for ThisTag in QueryTags :
        if not ThisTag in Tags :
            ret = False

    return ret
    

def GetActivationsByTags ( LayerInfoList, Tags='' ) :

    Outputs = []

    for info in LayerInfoList :
        if HasAllTags ( info.Tags, Tags ) :
            Outputs.append ( info.Out )

    return Outputs

