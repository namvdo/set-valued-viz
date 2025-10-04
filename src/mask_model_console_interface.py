import re

RE_INT = re.compile(r"-?\d+")
RE_FLOAT = re.compile(r"-?\d*\.\d+|-?\d+\.\d*")
##RE_NOTEMPTY_LINE = re.compile(r"\s*(\S.*)(?:\n|$)")

import matplotlib.pyplot as plt
import numpy as np


# from arrayprc
def around(a):
    signs = np.sign(a)
    i = np.int_(a)
    a *= signs
    a %= 1
    a *= 2*signs
    return i+np.int_(a)

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

    
def mapping_wrapper(one_of_the_mapping_functions):
    def wrapper(precision, x, y, *args, **kwargs):
        x *= precision
        y *= precision
        x,y = one_of_the_mapping_functions(x,y,*args,**kwargs)
        x /= precision
        y /= precision
        return x, y
    return wrapper

@mapping_wrapper
def mapping_henon(x, y, a=1.4, b=0.3):
    return (1-a*x*x+y), b*x
@mapping_wrapper
def mapping_func0(x, y):
    return x/2, y/2
@mapping_wrapper
def mapping_func1(x, y):
    return x * np.sin(np.pi*y*1.3), y * np.sin(np.pi*y)
@mapping_wrapper
def mapping_func2(x, y):
    return (x*2 + x*np.sin(y**2))/3, (y + x*2*np.cos(x**2))/3
@mapping_wrapper
def mapping_func3(x, y):
    return (y*np.cos(y+x**2)), x*np.sin(y**2+x)

class ConsoleInterface():
    interface_depth = 0
    
    PRECISION = 0.005 #ZOOM = 200
    START_POINT = (0,0)
    FUNCTION = None
    ALT_VISUALS = False
    AUTOFILL = False

    BORDER = 1
    EPSILON = 1.
    
    TIMESTEP = 0
    TOPLEFT = (0,0)
    BOTTOMRIGHT = (0,0)
    IMAGE_HISTORY = None
    IMAGE = None

    
    RUNUNTIL_TIMESTEP = 0
    RUNUNTIL_HAUSDORFF = None
    
    
    def __init__(self):
        self.FUNCTION = mapping_func0

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
            (f"Set Precision [{self.PRECISION}]", self.set_precision),
            (f"Set Epsilon [{self.EPSILON}]", self.set_epsilon),
            (f"Set Start Point [{self.START_POINT}]", self.set_startpoint),
            (f"Mapping Functions [{self.FUNCTION}]", None),
            (f"Load Configuration", None),
            (f"Save Configuration", None),
            (f"Exit", self.quit),
            ]
        self._show_user_options("Simple interface for dynamic model visualization", options)
        return self._process_user_input(options)
    
    def set_precision(self):
        desc = "Set the precision of the simulation."
        desc = clean_text_box(desc)
        self._depth_print(desc)
        self._set_attr("PRECISION")
        
    def set_epsilon(self):
        desc = "Set the radius of random uniform noise"
        desc = clean_text_box(desc)
        self._depth_print(desc)
        self._set_attr("EPSILON")
        
    def set_startpoint(self):
        desc = "Set the starting point of the simulation"
        desc = clean_text_box(desc)
        self._depth_print(desc)
        self._set_attr("START_POINT")
        
    def set_function(self):
        desc = "Set the mapping function to use"
        self._depth_print(desc)
        self._set_attr("FUNCTION")
        
        
    def _set_attr(self, attr_name):
        current_value = getattr(self, attr_name, None)
        t = type(current_value)
        
        user_input = self._depth_input(f"{current_value} -> ")
        success = False
        if user_input:
            if t in [int, float]:
                if RE_INT.match(user_input) or RE_FLOAT.match(user_input):
                    user_input = t(user_input.replace(",","."))
                    if user_input>0:
                        setattr(self, attr_name, user_input)
                        self._depth_print("Value set")
                        success = True
                else:
                    self._depth_print("Invalid input")
            else:
                self._depth_print(f"Value setting of type ({t}) is not defined")
        else: self._depth_print("Invalid input")
        print("")
        return success
        
    

    def activerun(self):
        self._activerun_init()
        while not self.activerun_options(): pass
        self._activerun_clear()
        
    def activerun_options(self):
        options = [
            ("Next", self._activerun_next),
            ("Open Matplotlib Figure", self._activerun_visuals),
            (f"Mapping Functions [{self.FUNCTION}]", None),
            (f"Set Epsilon [{self.EPSILON}]", self.set_epsilon),
            ("Run Until...", self.rununtil),
            ("Stop", self.quit),
            ]


        tl, br = np.multiply(self.TOPLEFT, self.PRECISION), np.multiply(self.BOTTOMRIGHT, self.PRECISION)
        desc = f"""Timestep: {self.TIMESTEP}
Precision: {self.PRECISION:.6f}
Domain: ({tl[0]}..{br[0]}, {tl[1]}..{br[1]})
Array shape: {self.IMAGE.shape}
Hausdorff: ???
"""
        if self.RUNUNTIL_TIMESTEP>0:
            self.RUNUNTIL_TIMESTEP -= 1
            self._activerun_next()
            self._depth_print(desc)
            return False
        
        self._show_user_options(desc, options)
        return self._process_user_input(options)

    def _activerun_init(self):
        # place the starting point on the mask
        self.IMAGE = np.zeros((3,3), dtype=np.bool_)
        self.IMAGE[1,1] = True

        adjusted_start = np.divide(self.START_POINT, self.PRECISION)
        self.TOPLEFT = np.subtract(adjusted_start, np.divide(self.IMAGE.shape, 2))
        self.BOTTOMRIGHT = np.add(self.IMAGE.shape, self.TOPLEFT)
        #
        
        self.IMAGE_HISTORY = self.IMAGE.astype(np.uint64)
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
        adjusted_epsilon = int(self.EPSILON/self.PRECISION)
        
        eps_border = adjusted_epsilon+self.BORDER
        eps_circle = circle_mask(adjusted_epsilon, border=self.BORDER)
        
        borders = get_mask_borders(self.IMAGE)
        indexes = calc_mask_indexes(self.IMAGE)

        points_to_process = list(indexes[borders].astype(np.float64)+self.TOPLEFT)
        points_processed = []
        for x,y in points_to_process:
            points_processed.append(self.FUNCTION(self.PRECISION,x,y))

            self.TOPLEFT = np.minimum(self.TOPLEFT, np.subtract(points_processed[-1], eps_border))
            self.BOTTOMRIGHT = np.maximum(self.BOTTOMRIGHT, np.add(points_processed[-1], eps_border))
        
        new_mask_shape = np.subtract(self.BOTTOMRIGHT, self.TOPLEFT).astype(np.int64)
            
        new_mask = np.zeros(new_mask_shape+1, dtype=np.bool_)
        for x,y in points_processed:
            x = int(x-self.TOPLEFT[0]-eps_border)
            y = int(y-self.TOPLEFT[1]-eps_border)
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
        offset = (offset-pad_r).astype(np.int64)
        pad_l = pad_r.copy()
        pad_r += pad_r%1>=.5
        pad_r = np.clip(pad_r, a_min=0, a_max=None).astype(np.int64)
        pad_l = np.clip(pad_l, a_min=0, a_max=None).astype(np.int64)
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











