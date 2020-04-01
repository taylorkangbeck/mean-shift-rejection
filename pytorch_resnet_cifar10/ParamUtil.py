from collections import OrderedDict

def GetFromListByName ( ParamList, Name, WithName=False, ndim=None, LayerName=None ) :

    # convert name(s) to lower case and in the case where only a single name string is passed (rather than list) then convert that to a list of just that name for compatibility with downstream code
    if type(Name) != type([]) :
        Name = [Name.lower()]
    else :
        Name = Name.lower()

    # get all parameters with the specified name(s) 
    NamedParamList = [ p for p in ParamList if p[0].split ('.')[-1].lower() in Name ]

    # optionally filter by LayerName
    if LayerName is not None :
        if type(LayerName) != type([]) :
            LayerName = [LayerName.lower()]
        else :
            LayerName = LayerName.lower()
        NamedParamList = [ p for p in NamedParamList ]

    # optionally strip any params not of the specified dimensionality
    if ndim is not None :
        NamedParamList = [ p for p in NamedParamList if len(p[1].shape)==ndim ]

    # optionally strip the name from the tuple (name, param)
    if not WithName :
        NamedParamList = [ p[1] for p in NamedParamList ]

    return NamedParamList

def GetFromListByName_ORIG ( ParamList, Name, WithName=False, ndim=None, LayerName=None ) :

    if type(Name) != type([]) :
        Name = [Name.lower()]
    else :
        Name = Name.lower()

    if type(LayerName) != type([]) :
        LayerName = [LayerName.lower()]
    else :
        LayerName = LayerName.lower()

    if WithName :
        if LayerName is None :
            NamedParamList = [ p for p in ParamList if p[0].split ('.')[-1].lower() in Name ]
        else :
            NamedParamList = [ p for p in ParamList if p[0].split ('.')[-1].lower in Name and p[0].split ('.')[:-1].lower() in LayerName]

        ParamNames = None
    else :
        NamedParamList = [ p[1] for p in ParamList if p[0].split ('.')[-1].lower() in Name ]
        ParamNames = [ p[0] for p in ParamList if p[0].split ('.')[-1].lower() in Name ]

    if ndim is not None :
        if WithName :
            NamedParamList = [ p for p in NamedParamList if len(p[1].shape)==ndim ]
        else :
            NamedParamList = [ p for p in NamedParamList if len(p.shape)==ndim ]

    return NamedParamList

def GetFromListByExcludeName ( ParamList, Name, WithName=False, ndim=None, LayerName=None ) :

    if type(Name) != type([]) :
        Name = [Name.lower()]
    else :
        Name = Name.lower()

    if WithName :
        NamedParamList = [ p for p in ParamList if p[0].split ('.')[-1].lower() not in Name ]
    else :
        NamedParamList = [ p[1] for p in ParamList if p[0].split ('.')[-1].lower() not in Name ]

    if ndim is not None :
        if WithName :
            NamedParamList = [ p for p in NamedParamList if len(p[1].shape)==ndim ]
        else :
            NamedParamList = [ p for p in NamedParamList if len(p.shape)==ndim ]

    return NamedParamList

def GetFromModelByName ( Model, Name, Exclude=False, WithName=False, ndim=None ) :

    NameParams = Model.named_parameters()
    return GetFromNamedParamsByName ( NameParams, Name, Exclude, WithName=WithName )

def GetFromNamedParamsByName ( NamedParams, Name, Exclude=False, WithName=False, ndim=None, LayerName=None ) :

    if Exclude :
        return GetFromListByExcludeName ( list(NamedParams), Name, WithName=WithName, ndim=ndim, LayerName=LayerName )
    if not Exclude :
        return GetFromListByName ( list(NamedParams), Name, WithName=WithName, ndim=ndim, LayerName=LayerName )

def DeepCopyGrads ( params ) :
    Grads = []
    for namedparam in params :
        p = namedparam[1]
        if p.requires_grad:
            Grads.append ( p.grad.data.clone() )

    return Grads

def TrainableParams ( model ) :
    TrainableParams = []
    for p in model.parameters():
        if p.requires_grad:
            TrainableParams.append ( p )

    return TrainableParams
