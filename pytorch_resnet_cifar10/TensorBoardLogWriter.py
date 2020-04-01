from torch.utils.tensorboard import SummaryWriter
from time import sleep
import os

"""
TODO:
[]  If log file already exists then when adding a stream check if it already exists and in which case load the iteration counter to the length of the stream
"""

class TensorBoardLogWriter () :

    def __init__ ( self, LogDir ) :
        self.TB = SummaryWriter(log_dir=LogDir)
        self.StreamIt = {}

    def AddStream ( self, StreamName, FirstIt=0 ) :
        """
        Creates a new TensorBoard named stream but does not write anything until the first item is written as that is how TensorBoard works i.e. no empty streams.
        The named stream holds the iteration counter for that stream.
        """
        if StreamName in self.StreamIt :
            assert False, "StreamName %s is already in the TensorBoardLogWriter" % (StreamName)
        
        # create the stream with iteration counter cleared to zero
        self.StreamIt[StreamName] = FirstIt

    def AddToLog ( self, StreamName, Value ) :
        """
        Write the scalar to the named stream at the current iteration and increment the iteration counter for that stream.
        """

        assert StreamName in self.StreamIt
       
        try :
            #Value = float(Value)
            StreamName = str(StreamName)
            ThisIter = int(self.StreamIt[StreamName])
            self.TB.add_scalar(StreamName, Value, ThisIter )

            # DEBUG WORKAROUND - in debugger need to sleep 1ms because otherwise TensorBoard on Windows has an internal issue and never returns i.e. blocks
            if 'VS_DEBUG' in os.environ :
                sleep(0.001)

        except Exception as e :
            print ("Unexpected error:", str(e) )

        self.StreamIt[StreamName] += 1

    def AddToLogX ( self, StreamName, Value, Iter ) :
        try :
            self.TB.add_scalar(StreamName, Value, Iter )
            # DEBUG WORKAROUND - in debugger need to sleep 1ms because otherwise TensorBoard on Windows has an internal issue and never returns i.e. blocks
            if 'VS_DEBUG' in os.environ :
                sleep(0.001)

        except Exception as e :
            print ("Unexpected error:", str(e) )

    def Close ( self ) :
        self.TB.close()

    def Open ( self ) :
        self.TB.close()
