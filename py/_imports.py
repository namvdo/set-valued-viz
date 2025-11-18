import os, sys
import matplotlib.pyplot as plt

from __equation import *
from __mask import *
from __system import *
from __vectors import *

WORKDIR, FILENAME = os.path.abspath(sys.argv[0]).rsplit(os.path.sep, 1)

class Point2D:
    x = y = 0
    def __init__(self, x, y):
        self.x, self.y = x, y
    def copy(self): return type(self)(self.x, self.y)
    def __str__(self):
        return f"({self.x}, {self.y})"

class EquationObject():
    def __init__(self, string):
        self.string = string
    
    def required_inputs(self): # return a set of variables needed to solve
        return set(re.findall(RE_ALGEBRAIC_POS, self.string)) - ALGEBRAIC_BLACKLIST

    def derivative(self, target_variable="x"):
        return type(self)(derivative(self.string, target=target_variable))

    def solve(self, **inputs):
        return solve(self.string, **inputs)
    
    def copy(self): return type(self)(self.string)
    
    def __str__(self): return self.string


class MappingFunction2D:
    def __init__(self, fx:str, fy:str):
        self.fx = EquationObject(fx)
        self.fy = EquationObject(fy)
        self.constants = {}

    def set_constants(self, **values):
        self.constants.update(values)
        
    def required_constants(self): # return a set of keyword arguments __call__ requires
        need = self.fx.required_inputs()
        need |= self.fy.required_inputs()
        if "x" in need: need.remove("x")
        if "y" in need: need.remove("y")
        return need

    def missing_constants(self):
        return self.required_constants()-set(self.constants.keys())

    def trim_excess_constants(self):
        required = self.required_constants()
        for k in list(self.constants.keys()):
            if k not in required: del self.constants[k]

    def copy(self):
        new = type(self)(self.fx.string, self.fy.string)
        new.constants = self.constants.copy()
        return new
    
    def __str__(self):
        x = self.fx.string
        y = self.fy.string
        for k,v in self.constants.items():
            v = str(v)
            x = x.replace(k, v)
            y = y.replace(k, v)
        missing = self.missing_constants()
        return f"({x}, {y})" + (str(" (has undefined constants)") if missing else "")
    
    def __call__(self, x, y, **inputs):
        return (self.fx.solve(x=x, y=y, **self.constants|inputs),
                self.fy.solve(x=x, y=y, **self.constants|inputs))

    def jacobian(self):
        return [
            [self.fx.derivative("x"), self.fx.derivative("y")],
            [self.fy.derivative("x"), self.fy.derivative("y")],
            ]

    def transposed_inverse_jacobian(self): # T(L)^-1
        jacobian = self.jacobian()
        # transpose
        jacobian[1][0], jacobian[0][1] = jacobian[0][1], jacobian[1][0]
        
        # determinant
        det = f"({jacobian[0][0].string})*({jacobian[1][1].string})-({jacobian[0][1].string})*({jacobian[1][0].string})"
        det = expand(shrink(det))
        if det=="0":
            print("non invertible matrix")
            return None
        
        # inverse
        jacobian[0][0], jacobian[1][1] = jacobian[1][1], jacobian[0][0]
        jacobian[0][1].string = f"-({jacobian[0][1].string})"
        jacobian[1][0].string = f"-({jacobian[1][0].string})"
        for i in range(2):
            for j in range(2):
                o = jacobian[i][j]
                o.string = f"({o.string})/({det})" # divide with the determinant
        return jacobian


class ModelBase():
    epsilon = 0.01
    
    def __init__(self):
        self.start_point = Point2D(0,0)
        self.function = MappingFunction2D("1-a*x*x+y", "b*x")
        self.function.set_constants(a=1.4, b=0.3)

    def copy(self):
        new = type(self)()
        new.epsilon = self.epsilon
        new.start_point = self.start_point.copy()
        new.function = self.function.copy()
        return new


def test_plotting_grid(width=1, height=1, timestep=0, figsize=(9,9)):
    while 1:
        fig,ax = plt.subplots(height, width, figsize=figsize)
        for i in range(width):
            for j in range(height):
                if width>1 and height>1: ax_target = ax[i][j]
                elif width*height>1: ax_target = ax[i*width+j]
                else: ax_target = ax
                ax_target.set_title(f"step: {timestep}")
                timestep += 1
                yield ax_target
        plt.show()

