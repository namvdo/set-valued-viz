import os, sys

from __equation import *
from __system import *
from __points_and_masks import *

WORKDIR, FILENAME = os.path.abspath(sys.argv[0]).rsplit(os.path.sep, 1)

class ModelBase():
    epsilon = 0.01
    start_point = None
    function = None
    
    def __init__(self):
        self.start_point = Point2D(0,0)
        self.function = MappingFunction2D("1-a*x*x+y", "b*x")
        self.function.set_constants(a=1.4, b=0.3)

    def copyattr(self, target, attr):
        if hasattr(target, attr):
            value = getattr(target, attr)
            if hasattr(value, "copy"): value = value.copy()
            setattr(self, attr, value)
            return True
        return False
    
    def copy(self):
        new = type(self)()
        new.epsilon = self.epsilon
        new.start_point = self.start_point.copy()
        new.function = self.function.copy()
        return new

