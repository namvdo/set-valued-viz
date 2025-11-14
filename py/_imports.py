import os, sys
import matplotlib.pyplot as plt

from func_equation import *
from func_mask import *
from func_system import *
from func_vectors import *

WORKDIR, FILENAME = os.path.abspath(sys.argv[0]).rsplit(os.path.sep, 1)


class Point2D:
    x = y = 0
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __str__(self):
        return f"({self.x}, {self.y})"

class EquationObject():
    string = ""
    
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
    def __init__(self):
        self.x = EquationObject("1-a*x*x+y")
        self.y = EquationObject("b*x")
        self.constants = {"a":1.4, "b":0.3}
        
    def required_constants(self): # return a set of keyword arguments __call__ requires
        need = self.x.required_inputs()
        need |= self.y.required_inputs()
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
        new = type(self)()
        new.constants = self.constants.copy()
        new.x = self.x.copy()
        new.y = self.y.copy()
        return new
    
    def __str__(self):
        x = str(self.x)
        y = str(self.y)
        for k,v in self.constants.items():
            v = str(v)
            x = x.replace(k, v)
            y = y.replace(k, v)
        missing = self.missing_constants()
        return f"(x={x}, y={y})" + (str(" (has undefined constants)") if missing else "")

    def __call__(self, x, y, **inputs):
        return (self.x.solve(x=x, y=y, **self.constants|inputs),
                self.y.solve(x=x, y=y, **self.constants|inputs))

    def jacobian(self):
        return [
            [self.x.derivative("x"), self.x.derivative("y")],
            [self.y.derivative("x"), self.y.derivative("y")],
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
