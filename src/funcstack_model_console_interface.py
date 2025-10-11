import os, sys
import math
import re
import matplotlib.pyplot as plt
import numpy as np
import pickle
from time import perf_counter_ns as nspec # nanoseconds

WORKDIR, FILENAME = os.path.abspath(sys.argv[0]).rsplit(os.path.sep, 1)
SAVE_DIR = os.path.join(WORKDIR, "mask_model_saves")

# from timepck
def function_timer(func):
    def wrapper(*args, **kwargs):
        print(func)
        t_start = nspec()
        out = func(*args, **kwargs)
        t_stop = nspec()
        print(f"{(t_stop-t_start)/1e6:.3f} ms")
        return out
    return wrapper
#

# from ospck
def list_files(path):
    try:
        if os.path.isdir(path):
            for i in os.listdir(path):
                i = os.path.join(path, i)
                if os.path.isfile(i): yield i
    except PermissionError: pass # denied

def list_folders(path):
    try:
        if os.path.isdir(path):
            for f in os.listdir(path):
                f = os.path.join(path, f)
                if os.path.isdir(f): yield f
    except PermissionError: pass # denied

def list_files_recur(path):
    for f in list_folders(path):
        yield from list_files_recur(f)
    yield from list_files(path)

def get_folder(path):
    if "." in os.path.basename(path): path = os.path.dirname(path)
    return path

def makedirs(path):
    if path:=get_folder(path):
        os.makedirs(path, exist_ok=True)

def getsize(path):
    if os.path.isfile(path): return os.path.getsize(path)
    return getsize_folder(path)

def getsize_folder(path):
    size = 0
    for i in list_files_recur(path):
        size += os.path.getsize(i)
    return size
#

SI_VALUES_UNIT_PREFIXES_4 = "kMGTPEZY"
def readable_filesize(size):
    p = 0
    if size>0: p += min(int(math.log(size, 1024)), len(SI_VALUES_UNIT_PREFIXES_4))
    if p>0: size /= 1024**p
    return f"{size:.1f} "+(SI_VALUES_UNIT_PREFIXES_4[p-1] if p>0 else "")+"B"


# from datapck
def load(path, default=None, makedirs_ok=True):
    if makedirs_ok: makedirs(path)
    if os.path.isfile(path):
        with open(path, "rb") as f:
            try: obj = pickle.load(f)
            except TypeError: obj = default
            except pickle.UnpicklingError: obj = default
        return obj
    return default
def save(path, obj):
    makedirs(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
#


# from arrayprc
def distance(a, b):
    return np.linalg.norm(np.subtract(a, b), axis=-1) # n-dimensional point to point distance

def nearestpoint_i(a, *b):
    return np.argmin([distance(a, x) for x in b])

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



# from engine_base
RE_INT = re.compile(r"-?\d+")
RE_FLOAT = re.compile(r"-?\d*\.\d+|-?\d+\.\d*")
RE_INT_OR_FLOAT_str = RE_FLOAT.pattern+"|"+RE_INT.pattern
RE_2D_POINT = re.compile(r"\s*\(?\s*("+RE_INT_OR_FLOAT_str+r")\s*\,\s*("+RE_INT_OR_FLOAT_str+r")\s*\)?\s*")

RE_FACTORIAL = re.compile(r"((\d+)\!+)")
RE_ADV_PARENTHESIS_OPS = re.compile(r"((a?(?:sin|cos|tan))?\(([^\(\)]+)\))")
RE_ALGEBRAIC = r"[a-zA-Z]+|(?:\{\d+\})"
RE_ADV_NUM = re.compile(r"(?:("+RE_FLOAT.pattern+")|("+RE_INT.pattern+")|("+RE_ALGEBRAIC+"))")
RE_ADV_SOLVE_EXP_LOG = re.compile(r"("+RE_ADV_NUM.pattern+r"(\*\*+|//+)"+RE_ADV_NUM.pattern+")")
RE_ADV_SOLVE_PROD_DIV = re.compile(r"("+RE_ADV_NUM.pattern+r"(\*|/)"+RE_ADV_NUM.pattern+")")
RE_ADV_SOLVE_ADD_SUB = re.compile(r"("+RE_ADV_NUM.pattern+r"(\++|\-+)"+RE_ADV_NUM.pattern+")")
RE_SIN_COS_TAN = re.compile(r"((a?(?:sin|cos|tan))\((.+)\))")
RE_ADV_PARENTHESIS_PROD = re.compile(r"("+RE_ALGEBRAIC+r"|\d+|\))(\(|"+RE_ALGEBRAIC+r"|\d+)")

algebraic_blacklist = {"sin","cos","tan","asin","acos","atan"} # not valid variable names in equations

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
        if w[0]=="-": w = "0"+w # catch any negative algebraic variables
        
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
    equation = equation.replace(" ", "")
    for y,z in RE_ADV_PARENTHESIS_PROD.findall(equation):
        if y not in algebraic_blacklist and z not in algebraic_blacklist:
            equation = equation.replace(f"{y}{z}", f"{y}*{z}")
    return advanced_solveoperation(equation, **inputs)
#













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
    epsilon = 0

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

    def copy(self):
        new = type(self)()
        new.constants = self.constants.copy()
        new.epsilon = self.epsilon
        new.x = self.x
        new.y = self.y
        return new
    
    def __str__(self):
        x = self.x
        y = self.y
        for k,v in self.constants.items():
            v = str(v)
            x = x.replace(k, v)
            y = y.replace(k, v)
        missing = self.missing_constants()
        return f"(x={x}, y={y})" + (str(" undefined constants!") if missing else "")

    def __call__(self, x, y, **inputs):
        return (advanced_solve(self.x, x=x, y=y, **inputs|self.constants),
                advanced_solve(self.y, x=x, y=y, **inputs|self.constants))




class ModelConfiguration():
    timestep = 0
    start_point = UserModifiablePoint(0,0)
    
    border_width = 1
    epsilon_radius = 0.1

    def __init__(self):
        self.timesteps = [] # stack of UserModifiableMappingFunction
        
        pass
        #

class ConsoleInterface():
    interface_depth = 0
    autorun_timesteps = 0
    autorun_hausdorff = 0
    
    def __init__(self):
        self.config = ModelConfiguration()
        self.function = UserModifiableMappingFunction()

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
        user_input = self._depth_input("SELECT: ")
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
            else:
                self._depth_print("Not a valid option")
        else: self._depth_print("Invalid input")
        print("")
        return False

    def quit(self): return True

    def mainmenu(self):
        while not self.mainmenu_options(): pass
    
    def mainmenu_options(self):
        options = [
            ("Start", self.activerun),
            ("Set Start Point", self.set_startpoint),
            ("Exit", self.quit),
            ]
        desc = "Simple interface for dynamic model visualization"
        desc += f"\nStarting point: {self.config.start_point}"
        self._show_user_options(desc, options)
        return self._process_user_input(options)


    def set_epsilon(self):
        desc = "Radius of the random uniform noise"
        desc = clean_text_box(desc)
        self._depth_print(desc)
        self._set_attr("epsilon_radius", zero_ok=True, target=self.config)
        
    def set_startpoint(self):
        desc = "Starting point of the simulation"
        desc = clean_text_box(desc)
        self._depth_print(desc)
        self._set_attr("start_point", target=self.config)
        
    def set_autorun_timesteps(self):
        self.autorun_timesteps = self.config.timestep
        if self._set_attr("autorun_timesteps", zero_ok=True):
            self.autorun_timesteps = max(self.autorun_timesteps-self.config.timestep, 0)
        else: self.autorun_timesteps = 0
        
        
    def _set_attr(self, attr_name, negative_ok=False, zero_ok=False, target=None):
        if target is None: target = self
        current_value = getattr(target, attr_name, None)
        t = type(current_value)
        
        user_input = self._depth_input(f"{current_value} -> ")
        success = False
        if user_input:
            if t in [int, float]:
                user_input = user_input.replace(",",".")
                if RE_INT.match(user_input) or (t is float and RE_FLOAT.match(user_input)):
                    user_input = t(user_input)
                    if user_input>0 or (negative_ok and user_input<0) or zero_ok:
                        setattr(target, attr_name, user_input)
                        success = True
                if not success:
                    complain = "Given value must be a "
                    if not zero_ok: complain += "non-zero "
                    if not negative_ok: complain += "positive "
                    complain += "integer" if t is int else "float"
                    self._depth_print(complain)
            elif t is str:
                setattr(target, attr_name, user_input)
                success = True
            elif t is UserModifiablePoint:
                m = RE_2D_POINT.match(user_input)
                if m:
                    new_point = UserModifiablePoint(float(m.group(1)), float(m.group(2)))
                    setattr(target, attr_name, new_point)
                    success = True
                else:
                    self._depth_print("Not a recognizable 2 dimensional point")
            else:
                self._depth_print(f"Value setting of type ({t}) is not defined")
        else: self._depth_print("Invalid input")
        if success:
            self._depth_print("Value set")
        print("")
        return success
        
    
    def modifyfunction(self):
        while not self.modifyfunction_options():
            self.function.trim_excess_constants()
        
    def modifyfunction_options(self):
        options = [
            ("Modify the X-axis Equation", self._modifyfunction_axis_x),
            ("Modify the Y-axis Equation", self._modifyfunction_axis_y),
            ("Define Constants", self._modifyfunction_define),
            ("Save Function", self.save_function),
            ("Load Function", self.load_function),
            ("Done", self.quit),
            ]
        desc = f"X-axis: {self.function.x}"
        desc += f"\nY-axis: {self.function.y}"
        
        if self.function.constants:
            desc += "\nDefined:"
            for k,v in self.function.constants.items():
                desc += f"\n  {k} = {v}"
        missing_constants = self.function.missing_constants()
        if missing_constants:
            desc += "\nUndefined: " + ", ".join(missing_constants)
        
        self._show_user_options(desc, options)
        return self._process_user_input(options)

    def _modifyfunction_axis_x(self):
        desc = "Set a new X-axis equation for the function"
        desc = clean_text_box(desc)
        self._depth_print(desc)
        user_input = self._depth_input(self.function.x+" -> ")
        if user_input:
            self.function.x = user_input
            self._depth_print("Value set")
        
    def _modifyfunction_axis_y(self):
        desc = "Set a new Y-axis equation for the function"
        desc = clean_text_box(desc)
        self._depth_print(desc)
        user_input = self._depth_input(self.function.y+" -> ")
        if user_input:
            self.function.y = user_input
            self._depth_print("Value set")

    def _modifyfunction_define(self):
        desc = "Define function constants"
        if self.function.constants:
            desc += "\nDefined:"
            for k,v in self.function.constants.items():
                desc += f"\n  {k} = {v}"
        missing_constants = self.function.missing_constants()
        if missing_constants:
            desc += "\nUndefined: " + ", ".join(missing_constants)
        desc = clean_text_box(desc)
        self._depth_print(desc)

        k = v = None
        
        user_input = self._depth_input("Constant: ")
        if not user_input:
            return
        if user_input in self.function.constants:
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
            self._depth_print("Invalid constant")
        
        if v is not None:
            self.function.constants[k] = v
            self._depth_print("Value set")
        elif k is not None:
            self._depth_print("Invalid value")


    def activerun(self):
        self._activerun_init()
        while not self.activerun_options(): pass
        
    def activerun_options(self):
        options = [
            ("Next Timestep", self._activerun_next),
            ("Open Matplotlib Figure", self._activerun_visuals),
            ("Modify Epsilon", self.set_epsilon),
            ("Modify Function", self.modifyfunction),
            ("Run Until Timestep", self.set_autorun_timesteps),
            ("Save State", self.save_timestep),
            ("Load State", self.load_timestep),
            ("More Data...", self._activerun_moredata),
            ("Stop", self.quit),
            ]

        desc = f"Timestep: {self.config.timestep}"
        desc += f"\nEpsilon radius: {self.config.epsilon_radius}"
        desc += f"\nFunction: {self.function}"
        if self.autorun_timesteps>0:
            self.autorun_timesteps -= 1
            desc = clean_text_box(desc)
            self._depth_print(desc+"\n")
            self._activerun_next(print_steps=False)
            return False
        
        self._show_user_options(desc, options)
        return self._process_user_input(options)

    def _activerun_moredata(self):
        desc = "Configuration"
        desc = clean_text_box(desc)
        self._depth_print(desc)
        self._depth_input("Back...")


    def _activerun_init(self):
        self.config = ModelConfiguration()

    def _activerun_visuals(self):
        self._depth_print("Close matplotlib window to continue", -1)
##        self.config.visualization()
        print("")

    def _activerun_fail_check(self): # would it crash
        return bool(self.function.missing_constants())
    
    def _activerun_next(self, print_steps=True):
        if self._activerun_fail_check():
            self._depth_print("Run failed")
            return
        # TODO








    

    def list_savefiles(self, ext):
        saves = []
        for f in list_files(SAVE_DIR):
            if f.endswith(f".{ext}"):
                name = os.path.basename(f).rsplit(".", 1)[0]
                size = readable_filesize(getsize(f))
                saves.append((name,size,f))
        
        if saves:
            max_name_len = len(max(saves, key=lambda x:len(x[0])))
            max_size_len = len(max(saves, key=lambda x:len(x[1])))
            
            desc = ""
            for name,size,f in saves:
                if desc: desc += "\n"
                desc += f"{name}"+"."*(10 + max_name_len*2-len(name) + max_size_len*2-len(size))+size
            
            desc = clean_text_box(desc, l=">", pad_t=0, pad_b=0)
            self._depth_print(desc)

    def save_timestep(self, name=None):
        if name is None:
            name = self._depth_input("Save as: ")
        if name:
            obj = None
            save(os.path.join(SAVE_DIR, name+".timestep"), obj)

    def save_function(self, name=None):
        if name is None:
            name = self._depth_input("Save as: ")
        if name:
            obj = self.function
            save(os.path.join(SAVE_DIR, name+".function"), obj)
    
    def load_timestep(self, name=None):
        if name is None:
            self.list_savefiles("timestep")
            name = self._depth_input("Timestep name: ")
        if name:
            obj = self._load_file(name, "timestep")
                
    def load_function(self, name=None):
        if name is None:
            self.list_savefiles("function")
            name = self._depth_input("Function name: ")
        if name:
            obj = self._load_file(name, "function")
            if obj is not None:
                self.function = obj

    def _load_file(self, name, ext):
        if name:
            f = os.path.join(SAVE_DIR, f"{name}.{ext}")
            if os.path.isfile(f):
                obj = load(f)
                if obj is not None:
                    self._depth_print("Success")
                    return obj
                else: self._depth_print("Failed")
            else: self._depth_print(f"File not found: '{f}'")
        else: self._depth_print("Cancelled")



def radians_absolute(radians): # -inf...inf -> 0...2*np.pi
    radians = np.mod(radians, np.pi*2)
    radians[radians<0] = radians[radians<0]+np.pi*2
    return radians

def radians_to_offsets(radians):
    offsets = np.repeat(np.expand_dims(radians, axis=1), 2, axis=1)
    offsets[:,0] = np.sin(offsets[:,0])
    offsets[:,1] = np.cos(offsets[:,1])
    return offsets

def offsets_to_radians(offsets):
    return np.arctan2(offsets[:,0], offsets[:,1])



def toggle_mask_to_continous_mask(toggle_mask):
    continous = np.zeros_like(toggle_mask)
    value = True
    i = j = 0
    while j<toggle_mask.size:
        i = toggle_mask[j:].argmax() # next one
        ii = toggle_mask[j:].argmin()
        if i==ii: break
        j += i
        continous[j:] = value
        value = not value
        j += 1
    return continous















def function_stack_test():
    config = ModelConfiguration()
    config.resolution = 512
    n = int(np.pi*config.resolution)*2
    config.radians = np.linspace(0, np.pi*2, n)

    start_func = UserModifiableMappingFunction()
    start_func.x = "x/5"
    start_func.y = "y/2"
    start_func.epsilon = 0.5
    start_point = UserModifiablePoint(0,0.2)
    timesteps = [] # stack of UserModifiableMappingFunction
    
    current_func = start_func.copy()
    
    def bounding_box(points):
        topleft = np.zeros(2)
        bottomright = np.zeros(2)
        topleft[0] = points[:,0].min()
        topleft[1] = points[:,1].min()
        bottomright[0] = points[:,0].max()
        bottomright[1] = points[:,1].max()
        return topleft, bottomright

    def point_normals(points):
        diff = np.diff(points, prepend=points[-1:], axis=0)
        return offsets_to_radians(diff)-np.pi/2

    def process_with_function_stack(radians):
        # starting
        inner_points = np.repeat([(start_point.x, -start_point.y)], radians.size, axis=0).astype(np.float64)
        outer_points = np.zeros_like(inner_points)
        outer_points[:,0], outer_points[:,1] = start_func(inner_points[:,0], inner_points[:,1])
        outer_points += radians_to_offsets(radians) * start_func.epsilon
        
        for timestep_func in timesteps:
            outer_points[:,0], outer_points[:,1] = timestep_func(outer_points[:,0], outer_points[:,1])
            
            # now outer_points must be expanded outward by the epsilon
            normals = point_normals(outer_points)
            
            candidates0 = radians_to_offsets(normals) # either inside or outside
            candidates1 = radians_to_offsets(normals+np.pi) # either inside or outside
            candidates0 *= timestep_func.epsilon
            candidates1 *= timestep_func.epsilon
            candidates0 += outer_points
            candidates1 += outer_points


##            normals0 = point_normals(candidates0)
##            normals1 = point_normals(candidates1)
##            edges0 = abs(np.diff(normals0, prepend=normals0[-1:]))>(np.pi/2)
##            edges1 = abs(np.diff(normals1, prepend=normals1[-1:]))>(np.pi/2)
##            classification = toggle_mask_to_continous_mask(edges0)
##            temp = candidates0[classification]
##            candidates0[classification] = candidates1[classification]
##            candidates1[classification] = temp
            

##            # the one with less edgy normals wins
##            score0 = abs(np.diff(point_normals(candidates0))).sum()
##            score1 = abs(np.diff(point_normals(candidates1))).sum()
##            if score0<score1:
##                outer_points = candidates0
##                inner_points = candidates1
##            else:
##                outer_points = candidates1
##                inner_points = candidates0
            
            # findout which has the larger bounding box -> the outside
            # and make the outside points the new outer_points
            cand0_topleft, cand0_bottomright = bounding_box(candidates0)
            cand1_topleft, cand1_bottomright = bounding_box(candidates1)
            cand0_area = np.prod(np.subtract(cand0_bottomright, cand0_topleft))
            cand1_area = np.prod(np.subtract(cand1_bottomright, cand1_topleft))
            if cand0_area>cand1_area:
                outer_points = candidates0
                inner_points = candidates1
            else:
                outer_points = candidates1
                inner_points = candidates0

        return outer_points, inner_points

    def pixelize_points(points, topleft, bottomright):
        pixelized_points = points-topleft
        pixelized_points /= (bottomright-topleft).max()#max(pixelized_points.max(), 1)
        pixelized_points *= config.resolution-1
        return pixelized_points
    
    # now every timestep forward use the ever increasingly accurate radian array to create the new boundary
    def draw_timestep_image():
        outer_points, inner_points = process_with_function_stack(config.radians)
        
        done = False
        while not done:
            # bounding box
            topleft, bottomright = bounding_box(outer_points)
            topleft2, bottomright2 = bounding_box(inner_points)
            topleft = np.minimum(topleft, topleft2)
            bottomright = np.maximum(bottomright, bottomright2)
            domain = np.subtract(bottomright, topleft)
            #
            
            # bring the points to the pixel range
            outer_pixels = pixelize_points(outer_points, topleft, bottomright)
            
            # resulting image might not appear continuous if there are pixel wide differences
            # solution is to expand the radians array in those areas and calculate their values through the stack
            diff = np.diff(outer_pixels, axis=0) # , prepend=outer_pixels[-1:], append=outer_pixels[:1]
            gap_size = (diff[:,0]+diff[:,1]).astype(np.int32)-2
            gap_mask = gap_size>0
            gap_ratio = gap_mask.sum()/gap_mask.size
            if gap_ratio>0.01 and config.radians.size<1e5:
                # extend the radians at the gap points
                lower_limit_mask = np.pad(gap_mask, (0,1))
                upper_limit_mask = np.pad(gap_mask, (1,0))
                
                lower_limits = config.radians[lower_limit_mask]
                upper_limits = config.radians[upper_limit_mask]
                additional_radians = np.linspace(lower_limits, upper_limits, 3)[1]
                
                config.radians = np.append(config.radians, additional_radians)
                config.radians.sort()
                print(config.radians.shape)
                additional_outer_points, additional_inner_points = process_with_function_stack(additional_radians)
                outer_points = np.append(outer_points, additional_outer_points, axis=0)
                inner_points = np.append(inner_points, additional_inner_points, axis=0)

            else:
                done = True
        
        # values are now continuous -> ready to draw
        inner_pixels = pixelize_points(inner_points, topleft, bottomright)
        
        outer_pixels += .5 # center to pixels
        inner_pixels += .5 # center to pixels
        aspect = max(outer_pixels[:,0].max(), inner_pixels[:,0].max())/max(outer_pixels[:,1].max(), inner_pixels[:,1].max())
        x_aspect = min(aspect, 1)
        y_aspect = min(1/aspect, 1)
        shape = (int(config.resolution*x_aspect)+1, int(config.resolution*y_aspect)+1)
##        shape = (config.resolution, config.resolution)

        image = np.zeros(shape)
        for x,y in outer_pixels.astype(np.uint16):
            image[x,y] = 2
        for x,y in inner_pixels.astype(np.uint16):
            image[x,y] = 1
        return image.astype(np.uint8), topleft, bottomright
        
##        image = np.zeros((*shape, 3))
##        for x,y in outer_pixels.astype(np.uint16):
##            image[x,y,0] += 1
##        for x,y in inner_pixels.astype(np.uint16):
##            image[x,y,1] += 1
##        image[:,:,0] /= max(image[:,:,0].max(), 1)
##        image[:,:,1] /= max(image[:,:,1].max(), 1)
##        image[:,:,2] /= max(image[:,:,2].max(), 1)
##        image *= 255
##        return image.astype(np.uint8), topleft, bottomright

    while 1:
        image, topleft, bottomright = draw_timestep_image()
        extent = (topleft[0],bottomright[0],-bottomright[1],-topleft[1])
        plt.imshow(image.swapaxes(0, 1), extent=extent)
        plt.show()
        
        timesteps.append(current_func.copy())
    

if __name__ == "__main__":
    function_stack_test()

    ## toggle_mask_to_continous_mask
##    a = np.zeros(8, dtype=np.bool_)
##    a[0] = True
##    a[7] = True
##    print(a)
##    b = toggle_mask_to_continous_mask(a)
##    print(b)
    
##    interface = ConsoleInterface()
##    interface.mainmenu()










