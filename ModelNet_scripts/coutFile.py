import os
import sys

class CoutToFile():

    def __init__(self, stream = None, fileName = "stdout.txt"):

        self.origstream = stream
        if self.origstream is None:
            self.origstream = sys.stdout
        self.origstreamfd = self.origstream.fileno()
        self.fName = fileName
        
        # Clear the file
        with open(self.fName, 'w') as f:
            f.write('')

    def start(self):

        self.streamfd = os.dup(self.origstreamfd)
        self.file = open(self.fName, 'a')
        os.dup2(self.file.fileno(), self.origstreamfd)

    def stop(self):

        self.origstream.flush()
        self.file.close()
        os.dup2(self.streamfd, self.origstreamfd)
        os.close(self.streamfd)

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, type, value, traceback):
        self.stop()

