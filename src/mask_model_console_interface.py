import re
import math
import matplotlib.pyplot as plt
import numpy as np


# from arrayprc

##def around(a):
##    signs = np.sign(a)
##    i = np.int_(a)
##    a *= signs
##    a %= 1
##    a *= 2*signs
##    return i+np.int_(a)
##
##def distance(a, b):
##    return np.linalg.norm(np.subtract(a, b), axis=-1) # n-dimensional point to point distance
##
##def nearestpoint_i(a, *b):
##    return np.argmin([distance(a, x) for x in b])

def circle_mask(r, inner=0, border=0):
    x = np.expand_dims(np.arange(-r, r+1), axis=1)
    y = np.expand_dims(np.arange(-r, r+1), axis=0)
    x *= x
    y *= y
    dist_from_center = np.sqrt(x + y)
    mask = dist_from_center<=r
    if inner>0: mask *= dist_from_center>inner
    elif inner<0: mask *= dist_from_center>r+inner
    return np.pad(mask, (border, border))
#





RE_INT = re.compile(r"-?\d+")
RE_FLOAT = re.compile(r"-?\d*\.\d+|-?\d+\.\d*")
RE_INT_OR_FLOAT_str = RE_FLOAT.pattern+"|"+RE_INT.pattern

RE_2D_POINT = re.compile(r"\s*\(?\s*("+RE_INT_OR_FLOAT_str+r")\s*\,\s*("+RE_INT_OR_FLOAT_str+r")\s*\)?\s*")


# from engine_base
RE_FACTORIAL = re.compile(r"((\d+)\!+)")
RE_PARENTHESIS_OPS = re.compile(r"(\(([^\(\)]+)\))")

RE_NUM = r"(?:("+RE_FLOAT.pattern+")|("+RE_INT.pattern+"))"
RE_SOLVE_EXP_LOG = re.compile(r"("+RE_NUM+r"(\*\*+|//+)"+RE_NUM+")")
RE_SOLVE_PROD_DIV = re.compile(r"("+RE_NUM+r"(\*|/)"+RE_NUM+")")
RE_SOLVE_ADD_SUB = re.compile(r"("+RE_NUM+r"(\++|\-+)"+RE_NUM+")")

RE_INT_GET = re.compile(r"^(\("+RE_INT.pattern+r"\)|"+RE_INT.pattern+")$")
RE_FLOAT_GET = re.compile(r"^(\("+RE_FLOAT.pattern+r"\)|"+RE_FLOAT.pattern+")$")
def solveoperation(w): # solve a number operation
    for x,z in RE_FACTORIAL.findall(w): w = w.replace(x, str(int(np.prod(arange(0, int(z))+1))))
    while l:=RE_PARENTHESIS_OPS.findall(w):
        for x,z in l: w = w.replace(x, str(solveoperation(z)))
    
    while l:=RE_SOLVE_EXP_LOG.findall(w): # exp (x**y) and log (x//y)
        for o,xf,xi,y,zf,zi in l:
            x = float(xf) if xf else int(xi)
            z = float(zf) if zf else int(zi)
            if "//" in y: w = w.replace(o, str(math.log(x, z)), 1)
            else: w = w.replace(o, str(x**z), 1)
            
    while l:=RE_SOLVE_PROD_DIV.findall(w): # prod (x*y) and div (x/y)
        for o,xf,xi,y,zf,zi in l:
            x = float(xf) if xf else int(xi)
            z = float(zf) if zf else int(zi)
            if "/" in y: w = w.replace(o, str(x/z), 1)
            else: w = w.replace(o, str(x*z), 1)
    while l:=RE_SOLVE_ADD_SUB.findall(w): # add (x+y) and subtract (x-y)
        for o,xf,xi,y,zf,zi in l:
            x = float(xf) if xf else int(xi)
            z = float(zf) if zf else int(zi)
            if y.count("-")%2: w = w.replace(o, str(x-z), 1)
            else: w = w.replace(o, str(x+z), 1)
    if m:=RE_INT_GET.match(w): w = int(m[0])
    else: w = float(w)
##        if had_dice: return www # can't save
##        SOLVED_OPERATIONS[w] = www
    return w
##    return SOLVED_OPERATIONS[w]


RE_NUM_2 = r"(?:(?:-?\d*\.\d+|-?\d+\.\d*)|(?:-?\d+))"
RE_NUM_2_PAR = r"\("+RE_NUM_2+r"\)"
RE_NUM_2 = r"(?:"+RE_NUM_2_PAR+"|"+RE_NUM_2+")" # or in parentheses # \([^\(\)]*\)

RE_PARENTHESIS_PROD = re.compile(r"(\d+|\))(\()")
RE_ANY_OPERATOR = r"[\+\-\*\/\!]+"
RE_SOLVE = re.compile(
    r"(?:"+RE_NUM_2+RE_ANY_OPERATOR+RE_NUM_2+"(?:"+RE_ANY_OPERATOR+RE_NUM_2+")?|"+RE_NUM_2_PAR+")+", flags=re.IGNORECASE)

def solve(x:str):
    for y,z in RE_PARENTHESIS_PROD.findall(x):
        x = x.replace(f"{y}{z}", f"{y}*{z}")
    while l:=RE_SOLVE.findall(x):
        for w in l: x = x.replace(w, str(solveoperation(w)))
    return x

##print(solve("2+2"))
##print(solve("2-2"))
##print(solve("3*2"))
##print(solve("3/2"))
##print(solve("2**3"))
##print(solve("2//3"))
##print(solve("16**(.5)"))




RE_ADV_PARENTHESIS_OPS = re.compile(r"((a?(?:sin|cos|tan))?\(([^\(\)]+)\))")

RE_ALGEBRAIC = r"[a-zA-Z]+|(?:\{\d+\})"
algebraic_blacklist = {"sin","cos","tan","asin","acos","atan"}

RE_ADV_NUM = re.compile(r"(?:("+RE_FLOAT.pattern+")|("+RE_INT.pattern+")|("+RE_ALGEBRAIC+"))")
RE_ADV_SOLVE_EXP_LOG = re.compile(r"("+RE_ADV_NUM.pattern+r"(\*\*+|//+)"+RE_ADV_NUM.pattern+")")
RE_ADV_SOLVE_PROD_DIV = re.compile(r"("+RE_ADV_NUM.pattern+r"(\*|/)"+RE_ADV_NUM.pattern+")")
RE_ADV_SOLVE_ADD_SUB = re.compile(r"("+RE_ADV_NUM.pattern+r"(\++|\-+)"+RE_ADV_NUM.pattern+")")

RE_SIN_COS_TAN = re.compile(r"((a?(?:sin|cos|tan))\((.+)\))")

RE_ADV_PARENTHESIS_PROD = re.compile(r"("+RE_ALGEBRAIC+r"|\d+|\))(\(|"+RE_ALGEBRAIC+r"|\d+)")

# apply an equation to values given in keyword arguments; x=1, y=np.array([1,8,.5])
def advanced_solveoperation(equation, **kwargs):
    i = 0
    values = {}
    
    substitutions = {}
    for k in re.findall(RE_ALGEBRAIC, equation):
        if k not in algebraic_blacklist:
            a = "{"+str(i)+"}"
            substitutions[a] = k
            if k not in kwargs: return None # not solvable, missing inputs
            equation = equation.replace(k, a, 1)
            i += 1
        
    def get_current_value(a):
        if a not in values:
            k = substitutions[a]
            if hasattr(kwargs[k], "copy"): values[a] = kwargs[k].copy()
            else: values[a] = kwargs[k]
        return values[a]
    
    def single_operation(w):
        for x,z in RE_FACTORIAL.findall(w):
            w = w.replace(x, str(int(np.prod(np.arange(0, int(z))+1))))
        
        for o,sct,z in RE_ADV_PARENTHESIS_OPS.findall(w):
            ww = single_operation(z)
            
            xf,xi,xA = RE_ADV_NUM.match(ww).groups()
            if xA: x = get_current_value(xA)
            else: x = float(xf) if xf else int(xi)
            match sct[-3:]:
                case "sin":
                    if sct[0]=="a": r = np.asin(x)
                    else: r = np.sin(x)
                case "cos":
                    if sct[0]=="a": r = np.acos(x)
                    else: r = np.cos(x)
                case "tan":
                    if sct[0]=="a": r = np.atan(x)
                    else: r = np.tan(x)
                case _: r = x
            
            if xA:
                values[xA] = r
                ww = xA
            else: ww = str(r)
            
            w = w.replace(o, ww)
        
        while l:=RE_ADV_SOLVE_EXP_LOG.findall(w): # exp (x**y) and log (x//y)
            for o,xf,xi,xA,y,zf,zi,zA in l:
                if xA: x = get_current_value(xA)
                else: x = float(xf) if xf else int(xi)
                if zA: z = get_current_value(zA)
                else: z = float(zf) if zf else int(zi)
                
                if "//" in y: r = np.log(x, z)
                else: r = x**z
                
                if xA and zA: # combined
                    values[xA] = r
                    del values[zA]
                    w = w.replace(o, xA, 1)
                elif xA:
                    values[xA] = r
                    w = w.replace(o, xA, 1)
                elif zA:
                    values[zA] = r
                    w = w.replace(o, zA, 1)
                else: w = w.replace(o, str(r), 1)
                
        while l:=RE_ADV_SOLVE_PROD_DIV.findall(w): # prod (x*y) and div (x/y)
            for o,xf,xi,xA,y,zf,zi,zA in l:
                if xA: x = get_current_value(xA)
                else: x = float(xf) if xf else int(xi)
                if zA: z = get_current_value(zA)
                else: z = float(zf) if zf else int(zi)
                
                if "/" in y: r = x/z
                else: r = x*z
                    
                if xA and zA: # combined
                    values[xA] = r
                    del values[zA]
                    w = w.replace(o, xA, 1)
                elif xA:
                    values[xA] = r
                    w = w.replace(o, xA, 1)
                elif zA:
                    values[zA] = r
                    w = w.replace(o, zA, 1)
                else: w = w.replace(o, str(r), 1)
                
        while l:=RE_ADV_SOLVE_ADD_SUB.findall(w): # add (x+y) and subtract (x-y)
            for o,xf,xi,xA,y,zf,zi,zA in l:
                if xA: x = get_current_value(xA)
                else: x = float(xf) if xf else int(xi)
                if zA: z = get_current_value(zA)
                else: z = float(zf) if zf else int(zi)
                
                if y.count("-")%2: r = x-z
                else: r = x+z
                
                if xA and zA: # combined
                    values[xA] = r
                    del values[zA]
                    w = w.replace(o, xA, 1)
                elif xA:
                    values[xA] = r
                    w = w.replace(o, xA, 1)
                elif zA:
                    values[zA] = r
                    w = w.replace(o, zA, 1)
                else: w = w.replace(o, str(r), 1)
        return w
    
    single_operation(equation) # start the solving chain
    
    for k in values.keys():
        if k not in kwargs: return values[k] # must be the result
    return 0

def advanced_solve(equation, **inputs):
    for y,z in RE_ADV_PARENTHESIS_PROD.findall(equation):
        if y not in algebraic_blacklist and z not in algebraic_blacklist:
            equation = equation.replace(f"{y}{z}", f"{y}*{z}")
    return advanced_solveoperation(equation, **inputs)


##inputs = {"x": 3,"y": np.array((3.5,6))}
##print(inputs)
##print(advanced_solve("cos(x+1)3", **inputs))
##print(advanced_solveoperation("x**2+4!", **inputs))
##print(advanced_solveoperation("(x-2)/y", **inputs))
##print(advanced_solveoperation("(x**2+2)+(x-2)/y", **inputs))
##print(advanced_solveoperation("((x-2)/(y+x))+z", **inputs))
##print(advanced_solveoperation("sin(-x)", **inputs))
##print(advanced_solveoperation("cos(-x)", **inputs))
##print(advanced_solveoperation("tan(-x)", **inputs))
##input()
#






def get_mask_borders(mask):
    borders = mask.copy()
    borders[1:,:] = mask[1:]*~mask[:-1]
    borders[:,1:] |= mask[:,1:]*~mask[:,:-1]
    borders[:-1,:] |= ~mask[1:,]*mask[:-1]
    borders[:,:-1] |= ~mask[:,1:]*mask[:,:-1]
    return borders

def calc_mask_indexes(mask):
    indexes = np.expand_dims(np.arange(mask.shape[0]), axis=1)
    indexes = np.expand_dims(indexes, axis=2)
    indexes = np.repeat(indexes, mask.shape[1], axis=1)
    indexes = np.repeat(indexes, 2, axis=2)
    indexes[:,:,1] = np.arange(mask.shape[1])
    return indexes


def fill_closed_areas(mask): # very crude
    fill_mask_hor = np.zeros(mask.shape, dtype=np.bool_).copy()
    fill_mask_ver = fill_mask_hor.copy()
    
    for r,row in enumerate(mask):
        i = row.argmax() # first one
        while 1:
            j = i+row[i:].argmin() # first zero after that
            ii = j+row[j:].argmax() # closing one
            if j==ii:
                break
            fill_mask_hor[r,j:ii] = True
            i = ii
            
    for c in range(mask.shape[1]):
        col = mask[:,c]
        i = col.argmax() # first one
        while 1:
            j = i+col[i:].argmin() # first zero after that
            ii = j+col[j:].argmax() # closing one
            if j==ii:
                break
            fill_mask_ver[j:ii,c] = True
            i = ii
    
    return fill_mask_hor*fill_mask_ver





def text_box(text,
             t="_", l="|", r="|", b="#",
             pad_t=1, pad_l=1, pad_b=1, pad_r=1,
             tl=None, tr=None, bl=None, br=None):
    
    lines = text.split("\n")#RE_NOTEMPTY_LINE.findall(text)
    text_width = len(max(lines, key=lambda x:len(x)))
    for i,line in enumerate(lines):
        lines[i] += " "*(text_width-len(line))
    lines = [" "*text_width]*pad_t+lines+[" "*text_width]*pad_b
    text_width += pad_l+pad_r+len(r)+len(l)

    text = "\n".join(lines)

    l = (l+" "*pad_l)
    r = (" "*pad_r+r)
    
    text = l+text.replace("\n", r+"\n"+l)+r
    text = (t*text_width+"\n" if t else "")+text+("\n"+b*text_width if b else "")

    if tl is not None: text = tl+text[1:]
    if tr is not None:
        text1, text2 = text.split("\n", 1)
        text = text1[:-1]+tr+"\n"+text2
    if bl is not None:
        text1, text2 = text.rsplit("\n", 1)
        text = text1+"\n"+bl+text2[1:]
    if br is not None: text = text[:-1]+br
    return text

def clean_text_box(text, **kwargs_override):
    kwargs = {
        "t": "─",
        "b": "─",
        "l": "│",
        "r": "│",
        "tl": "┌",
        "tr": "┐",
        "bl": "└",
        "br": "┘",
        }
    kwargs |= kwargs_override
    return text_box(text, **kwargs)




##def halve_the_array_size(array):
##    pad = np.mod(array.shape, 2)
##    smaller = np.zeros(np.true_divide(array.shape, 2).astype(np.int32)+pad, dtype=np.float64)
##    array = np.pad(array, pad)
##    smaller[:,:] = array[:-1:2,:-1:2].astype(np.float64) + array[1::2,1::2]
##    return smaller/2
##
##def third_the_array_size(array):
##    smaller = np.zeros(np.true_divide(array.shape, 3).astype(np.int32), dtype=np.float64)
##    smaller[:,:] = array[:-2:3,:-2:3].astype(np.float64)
##    smaller += array[1:-1:3,1:-1:3].astype(np.float64)
##    smaller += array[2::3,2::3].astype(np.float64)
##    return smaller/3




class UserModifiablePoint:
    x = 0
    y = 0
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __str__(self):
        return f"({self.x}, {self.y})"

    
class UserModifiableMappingFunction:
    x = "1-a*x*x+y"
    y = "b*x"

    def __init__(self):
        self.constants = {"a":1.4, "b":0.3}
        
    def required_constants(self): # return a set of keyword arguments __call__ requires
        need = set(re.findall(RE_ALGEBRAIC, self.x))
        need |= set(re.findall(RE_ALGEBRAIC, self.y))
        need -= algebraic_blacklist
        if "x" in need: need.remove("x")
        if "y" in need: need.remove("y")
        return need

    def missing_constants(self):
        return self.required_constants()-set(self.constants.keys())

    def trim_excess_constants(self):
        required = self.required_constants()
        for k in list(self.constants.keys()):
            if k not in required: del self.constants[k]

    def save(self):
        pass
    
    def load(self):
        pass
    
    def __str__(self):
        x = self.x
        y = self.y
        for k,v in self.constants.items():
            x = x.replace(k, str(v))
            y = y.replace(k, str(v))
        missing = self.missing_constants()
        return f"(x= {x}, y= {y})" + (str(" Undefined values!") if missing else "")

    def __call__(self, x, y, **inputs):
        return (advanced_solveoperation(self.x, x=x, y=y, **inputs|self.constants),
                advanced_solveoperation(self.y, x=x, y=y, **inputs|self.constants))


class ConsoleInterface():
    interface_depth = 0
    
    PRECISION = 0.005 #ZOOM = 200
    START_POINT = UserModifiablePoint(0,0)
    FUNCTION = None
    ALT_VISUALS = False
    AUTOFILL = False

    BORDER = 1
    EPSILON = 0.1
    
    TIMESTEP = 0
    TOPLEFT = (0,0)
    BOTTOMRIGHT = (0,0)
    IMAGE_HISTORY = None
    IMAGE = None

    
    RUNUNTIL_TIMESTEP = 0
    RUNUNTIL_HAUSDORFF = None
    
    
    def __init__(self):
        self.FUNCTION = UserModifiableMappingFunction()

    def _show_user_options(self, desc, options):
        desc = clean_text_box(desc)
        
        string = ""
        for i,x in enumerate(options):
            if i!=0: string += "\n"
            string += str(i)+". "+x[0]
        
        string = clean_text_box(string, l="* ", pad_t=0, pad_b=0)
        self._depth_print(desc+"\n"+string)
        
    def _depth_print(self, string, offset=0):
        depth = " "*(max(self.interface_depth+offset, 0)*4)
        string = string.replace("\n", "\n"+depth)
        print(depth+string)
        
    def _depth_input(self, string, *args, offset=0, **kwargs):
        depth = " "*(max(self.interface_depth+offset, 0)*4)
        string = string.replace("\n", "\n"+depth)
        return input(depth+string, *args, **kwargs)

    def _process_user_input(self, options):
        user_input = self._depth_input("\nSELECTION: ")
        if user_input and user_input.isnumeric():
            user_input = int(user_input)
            if 0<=user_input<len(options):
                f = options[user_input][1]
                if f is not None:
                    self._depth_print("-> "+options[user_input][0])
                    self.interface_depth += 1
                    output = f()
                    self.interface_depth -= 1
                    return output
                else:
                    self._depth_print("Not implemented")
            else: self._depth_print("User input was not in range")
        else: self._depth_print("Invalid input")
        print("")
        return False

    def quit(self): return True

    def mainmenu(self):
        while not self.mainmenu_options(): pass
    
    def mainmenu_options(self):
        options = [
            ("Start", self.activerun),
            ("Set Precision", self.set_precision),
            ("Set Start Point", self.set_startpoint),
            ("Load State", None),
            ("Exit", self.quit),
            ]
        desc = "Simple interface for dynamic model visualization"
        desc += f"\nPrecision: {self.PRECISION}"
        desc += f"\nStarting point: {self.START_POINT}"
        self._show_user_options(desc, options)
        return self._process_user_input(options)
    
    def set_precision(self):
        desc = "Precision of the simulation."
        desc = clean_text_box(desc)
        self._depth_print(desc)
        self._set_attr("PRECISION")
        
    def set_epsilon(self):
        desc = "Radius of the random uniform noise"
        desc = clean_text_box(desc)
        self._depth_print(desc)
        self._set_attr("EPSILON")
        
    def set_startpoint(self):
        desc = "Starting point of the simulation"
        desc = clean_text_box(desc)
        self._depth_print(desc)
        self._set_attr("START_POINT")
        
        
    def _set_attr(self, attr_name):
        current_value = getattr(self, attr_name, None)
        t = type(current_value)
        
        user_input = self._depth_input(f"{current_value} -> ")
        success = False
        if user_input:
            if t in [int, float]:
                user_input = user_input.replace(",",".")
                if RE_INT.match(user_input) or RE_FLOAT.match(user_input):
                    user_input = t(user_input)
                    if user_input>0:
                        setattr(self, attr_name, user_input)
                        self._depth_print("Value set")
                        success = True
                else:
                    self._depth_print("Invalid input")
            elif t is str:
                setattr(self, attr_name, user_input)
                self._depth_print("Value set")
                success = True
            elif t is UserModifiablePoint:
                m = RE_2D_POINT.match(user_input)
                if m:
                    new_point = UserModifiablePoint(float(m.group(1)), float(m.group(2)))
                    setattr(self, attr_name, new_point)
                    self._depth_print("Value set")
                    success = True
                else:
                    self._depth_print("Invalid input")
            else:
                self._depth_print(f"Value setting of type ({t}) is not defined")
        else: self._depth_print("Invalid input")
        print("")
        return success
        
    
    def modifyfunction(self):
        while not self.modifyfunction_options():
            self.FUNCTION.trim_excess_constants()
        
    def modifyfunction_options(self):
        options = [
            ("Modify the x-axis equation", self._modifyfunction_axis_x),
            ("Modify the y-axis equation", self._modifyfunction_axis_y),
            ("Define constants", self._modifyfunction_define),
            ("Save as", None),
            ("Load", None),
            ("Done", self.quit),
            ]
        desc = "asdasd"
        desc += f"\nX-axis: {self.FUNCTION.x}"
        desc += f"\nY-axis: {self.FUNCTION.y}"
        
        if self.FUNCTION.constants:
            desc += "\nDefined:"
            for k,v in self.FUNCTION.constants.items():
                desc += f"\n  {k} = {v}"
        missing_constants = self.FUNCTION.missing_constants()
        if missing_constants:
            desc += "\nUndefined: " + ", ".join(missing_constants)
        
        self._show_user_options(desc, options)
        return self._process_user_input(options)

    def _modifyfunction_axis_x(self):
        desc = "Give a new x-axis equation for the function"
        desc = clean_text_box(desc)
        self._depth_print(desc)
        user_input = self._depth_input(self.FUNCTION.x+" -> ")
        if user_input:
            self.FUNCTION.x = user_input
            self._depth_print("Value set")
        
    def _modifyfunction_axis_y(self):
        desc = "Give a new y-axis equation for the function."
        desc = clean_text_box(desc)
        self._depth_print(desc)
        user_input = self._depth_input(self.FUNCTION.y+" -> ")
        if user_input:
            self.FUNCTION.y = user_input
            self._depth_print("Value set")

    def _modifyfunction_define(self):
        desc = "Define or redefine function constants."
        if self.FUNCTION.constants:
            desc += "\nDefined:"
            for k,v in self.FUNCTION.constants.items():
                desc += f"\n  {k} = {v}"
        missing_constants = self.FUNCTION.missing_constants()
        if missing_constants:
            desc += "\nUndefined: " + ", ".join(missing_constants)
        desc = clean_text_box(desc)
        self._depth_print(desc)

        k = None
        v = None
        if 1:
##        while 1:
            user_input = self._depth_input("Constant: ")
            if not user_input: return
            if user_input in self.FUNCTION.constants:
                k = user_input
                user_input = self._depth_input(f"Redefine '{user_input}' as: ")
                user_input = user_input.replace(",",".")
                if m:=RE_FLOAT.match(user_input): v = float(user_input)
                elif m:=RE_INT.match(user_input): v = int(user_input)
            elif user_input in missing_constants:
                k = user_input
                user_input = self._depth_input(f"Define '{user_input}' as: ")
                user_input = user_input.replace(",",".")
                if m:=RE_FLOAT.match(user_input): v = float(user_input)
                elif m:=RE_INT.match(user_input): v = int(user_input)
            else:
                self._depth_print("Invalid variable")
        
        if v is not None:
            self.FUNCTION.constants[k] = v
            self._depth_print("Value set")
        elif k is not None:
            self._depth_print("Invalid value")


    def activerun(self):
        self._activerun_init()
        while not self.activerun_options(): pass
        self._activerun_clear()
        
    def activerun_options(self):
        options = [
            ("Next State", self._activerun_next),
            ("Open Matplotlib Figure", self._activerun_visuals),
            ("Modify Epsilon", self.set_epsilon),
            ("Modify Function", self.modifyfunction),
            ("Run Until...", self.rununtil),
            ("Save State", None),
            ("Stop", self.quit),
            ]

        tl, br = np.multiply(self.TOPLEFT, self.PRECISION), np.multiply(self.BOTTOMRIGHT, self.PRECISION)
        desc = f"""Timestep: {self.TIMESTEP}
Precision: {self.PRECISION}
Epsilon radius: {self.EPSILON} ({int(self.EPSILON/self.PRECISION)} pixels)
Function: {self.FUNCTION}
Array shape: {self.IMAGE.shape}
Array domain: ({tl[0]} ... {br[0]}, {tl[1]} ... {br[1]})
Hausdorff: ???"""
        if self.RUNUNTIL_TIMESTEP>0:
            self.RUNUNTIL_TIMESTEP -= 1
            self._activerun_next()
            self._depth_print(desc+"\n")
            return False
        
        self._show_user_options(desc, options)
        return self._process_user_input(options)

    def _activerun_init(self):
        # place the starting point on the mask
        self.IMAGE = np.zeros((3,3), dtype=np.bool_)
        self.IMAGE[1,1] = True

        start = (self.START_POINT.x, self.START_POINT.y)
        self.TOPLEFT = np.subtract(start, np.divide(self.IMAGE.shape, 2))
        self.BOTTOMRIGHT = np.add(self.IMAGE.shape, self.TOPLEFT)
        #
        
        self.IMAGE_HISTORY = self.IMAGE.astype(np.uint16)
        self.TOPLEFT_PREV = self.TOPLEFT
        self.TIMESTEP = 0
        
    def _activerun_clear(self):
        plt.close()
        self.IMAGE_HISTORY = None
        self.IMAGE = None
        self.TIMESTEP = 0

    def _activerun_visuals(self):
        axis_range = (self.TOPLEFT[0],self.BOTTOMRIGHT[0],self.TOPLEFT[1],self.BOTTOMRIGHT[1])
        axis_range = np.multiply(axis_range, self.PRECISION)
        
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(self.IMAGE_HISTORY, extent=axis_range)
        ax[1].imshow(self.IMAGE, extent=axis_range)
        ax[0].set_title(f"Cumulative Timesteps")
        ax[1].set_title(f"Timestep: {self.TIMESTEP}")
        self._depth_print("Close matplotlib window to continue", -1)
        plt.show()
        print("")
        
    def _activerun_next(self):
        #######
        epsilon_units = int(self.EPSILON/self.PRECISION)
        
        eps_border_units = epsilon_units+self.BORDER
        eps_circle = circle_mask(epsilon_units, border=self.BORDER)
        
        borders = get_mask_borders(self.IMAGE)
        indexes = calc_mask_indexes(self.IMAGE)
        
        points = indexes[borders].astype(np.float64)-np.divide(self.IMAGE.shape, 2)+.5
        
        points *= self.PRECISION
        points[:,0], points[:,1] = self.FUNCTION(points[:,0], points[:,1])
        points /= self.PRECISION # back to units
        
        self.TOPLEFT = np.minimum(self.TOPLEFT, np.min(np.subtract(points, eps_border_units), axis=0))
        self.BOTTOMRIGHT = np.maximum(self.BOTTOMRIGHT, np.max(np.add(points, eps_border_units), axis=0))
        
        new_mask_shape = np.subtract(self.BOTTOMRIGHT, self.TOPLEFT).astype(np.int32)
        
        new_mask = np.zeros(new_mask_shape+1, dtype=np.bool_)
        for x,y in points:
            x = int(x-self.TOPLEFT[0]-eps_border_units)
            y = int(y-self.TOPLEFT[1]-eps_border_units)
            x_slice = slice(max(x, 0), x+eps_circle.shape[0])
            y_slice = slice(max(y, 0), y+eps_circle.shape[1])
            new_mask[x_slice, y_slice] |= eps_circle
        self.IMAGE = new_mask

        if self.AUTOFILL:
            self.IMAGE |= fill_closed_areas(self.IMAGE)
        #########
        
        # expand the total image array
        offset = (self.TOPLEFT_PREV-self.TOPLEFT)
        pad_r = np.subtract(self.IMAGE.shape, self.IMAGE_HISTORY.shape)/2
        offset = (offset-pad_r).astype(np.int32)
        pad_l = pad_r.copy()
        pad_r += pad_r%1>=.5
        pad_r = np.clip(pad_r, a_min=0, a_max=None).astype(np.int32)
        pad_l = np.clip(pad_l, a_min=0, a_max=None).astype(np.int32)
        pad = ((pad_l[0]+offset[0],pad_r[0]-offset[0]), (pad_l[1]+offset[1],pad_r[1]-offset[1]))
        self.IMAGE_HISTORY = np.pad(self.IMAGE_HISTORY, pad)
        #

        # add the mask to create a full image
        if self.ALT_VISUALS: self.IMAGE_HISTORY[self.IMAGE] = self.IMAGE[self.IMAGE]*(self.TIMESTEP+1)
        else: self.IMAGE_HISTORY += self.IMAGE
        #
        
        self.TOPLEFT_PREV = self.TOPLEFT
        self.TIMESTEP += 1



    
    def set_rununtil_timestep(self):
        desc = f"{self.TIMESTEP+1}..."
        self._depth_print(desc)
        if self._set_attr("RUNUNTIL_TIMESTEP"):
            self.RUNUNTIL_TIMESTEP = max(self.RUNUNTIL_TIMESTEP-self.TIMESTEP, 0)
            return True

    def rununtil(self):
        while not self.rununtil_options(): pass
    
    def rununtil_options(self):
        options = [
            ("Timestep is reached", self.set_rununtil_timestep),
            ("Hausdorff distance is exceeded", None),
            ("Cancel", self.quit),
            ]
        self._show_user_options("Run until...", options)
        return self._process_user_input(options)





if __name__ == "__main__":
    interface = ConsoleInterface()
    interface.mainmenu()



    
##    test_image = circle_mask(50, border=5)
##    print(test_image.shape)
##
##    half_image = halve_the_array_size(test_image)
##    print(half_image.shape)
##    quarter_image = halve_the_array_size(half_image)
##    quarter_image = (quarter_image*4).astype(np.int8)
##    print(quarter_image.shape)
##    print(quarter_image)
##
##    print("")
##    third_image = third_the_array_size(test_image)
##    print(third_image.shape)
##    sixth_image = third_the_array_size(third_image)
##    sixth_image = (sixth_image*9).astype(np.int8)
##    print(sixth_image.shape)
##    print(sixth_image)

    









