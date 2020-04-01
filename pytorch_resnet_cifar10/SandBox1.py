

class MethodImplicitCallTest () :

    def __init__ ( self ) :
        pass

    def __call__ ( self, param ) :
        return param

if __name__ == '__main__':

    MethodImplicitCallTest_instance = MethodImplicitCallTest()

    val = MethodImplicitCallTest_instance(5)

    pass
