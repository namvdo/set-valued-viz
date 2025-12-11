import os, sys
import matplotlib.pyplot as plt

from __equation import *
from __system import *
from __points_and_masks import *

WORKDIR, FILENAME = os.path.abspath(sys.argv[0]).rsplit(os.path.sep, 1)

class Point2D:
    x = y = 0
    def __init__(self, x, y):
        self.x, self.y = x, y
    def copy(self): return type(self)(self.x, self.y)
    def __str__(self):
        return f"({self.x}, {self.y})"

    def as_tuple(self): return self.x, self.y

class ModelBase():
    epsilon = 0.01
    start_point = None
    function = None
    
    def __init__(self):
        self.start_point = Point2D(0,0)
        self.function = MappingFunction2D("1-a*x*x+y", "b*x")
        self.function.set_constants(a=1.4, b=0.3)

    def copy_attributes_from(self, obj):
        for attr in ["epsilon","start_point","function"]:
            if hasattr(obj, attr):
                value = getattr(obj, attr)
                if hasattr(value, "copy"): value = value.copy()
                setattr(self, attr, value)
    
    def copy(self):
        new = type(self)()
        new.epsilon = self.epsilon
        new.start_point = self.start_point.copy()
        new.function = self.function.copy()
        return new




class ImageDrawing:
    class LinesObj:
        starts = None
        ends = None
    class CirclesObj:
        points = None
        radius = 1
        inside = 0
    class GridObj:
        center = (0,0)
        size = .1
    
    tl = None
    br = None
    
    def __init__(self, *args, **kwargs):
        self.background = self._color_check(*args, **kwargs)
        self.objects = []
        self.colors = [] # list of RGBA color arrays (0..1)

    def _color_check(self, r=0, g=0, b=0, a=1, **kwargs):
        return np.asarray([r,g,b,a])

    def clear(self):
        self.tl = self.br = None
        self.objects.clear()
        self.colors.clear()

    def update_tl_br(self, tl, br):
        if self.tl is None: self.tl = tl
        else: self.tl = np.min([tl,self.tl], axis=0)
        if self.br is None: self.br = br
        else: self.br = np.max([br,self.br], axis=0)

    def get_extent(self):
        return (self.tl[0], self.br[0], self.tl[1], self.br[1])
        
    def points(self, points, *args, **kwargs):
        points = np.asarray(points, dtype=np.float64)
        self.update_tl_br(*bounding_box(points))
        self.objects.append(points)
        self.colors.append(self._color_check(*args, **kwargs))
    
    def lines(self, starts, ends, *args, **kwargs):
        obj = self.LinesObj()
        obj.starts = np.asarray(starts, dtype=np.float64)
        obj.ends = np.asarray(ends, dtype=np.float64)
        self.update_tl_br(*bounding_box(obj.starts))
        self.update_tl_br(*bounding_box(obj.ends))
        self.objects.append(obj)
        self.colors.append(self._color_check(*args, **kwargs))
    
    def circles(self, points, radius, *args, inside=0, **kwargs):
        obj = self.CirclesObj()
        obj.points = np.asarray(points, dtype=np.float64)
        obj.radius = radius
        obj.inside = inside
        self.update_tl_br(*bounding_box(obj.points-radius))
        self.update_tl_br(*bounding_box(obj.points+radius))
        self.objects.append(obj)
        self.colors.append(self._color_check(*args, **kwargs))

    def grid(self, center, size, *args, **kwargs):
        obj = self.GridObj()
        obj.center = np.asarray(center, dtype=np.float64)
        obj.size = size
        self.objects.append(obj)
        self.colors.append(self._color_check(*args, **kwargs))

    def draw(self, resolution:int, topright_is_the_positive_corner=True):
        pixels = []
        for x in self.objects:
            if isinstance(x, self.LinesObj):
                s = pixelize_points(x.starts, self.tl, self.br, resolution)
                e = pixelize_points(x.ends, self.tl, self.br, resolution)
                pixels.append((0, s, e))
                
            elif isinstance(x, self.CirclesObj):
                c = pixelize_points(x.points, self.tl, self.br, resolution)
                radii = pixelize_distances([x.radius, x.inside], self.tl, self.br, resolution)
                m = circle_mask(*radii)
                pixels.append((1, c, m))
                
            elif isinstance(x, self.GridObj):
                c = x.center
                s = x.size
                pixels.append((2, c, s))
                
            elif type(x)==np.ndarray:
                pixels.append(pixelize_points(x, self.tl, self.br, resolution))
        
        limits = pixelize_points(self.br, self.tl, self.br, resolution)
        
        image = np.zeros((*image_shape(resolution, *limits), len(self.background)))
        image[:,:] = self.background

        def color_blend_on_array(array, color):
            if color[3]<1:
                array[:,:3] = color[:3]*color[3] + array[:,:3]*array[:,3:4]*(1-color[3])
                array[:,3] = 1-(1-array[:,3])*(1-color[3])
                array[:,:3] /= array[:,3:4]
            else:
                array[:] = color
            return array
        
        masks = {} # storage for mask reuse
        
        def draw_line_on_image(start, end, color):
            key = (int(start[0]-end[0]), int(start[1]-end[1]))
            alt_key = (-key[0], -key[1])
            if key in masks: mask = masks[key]
            elif alt_key in masks: mask = masks[alt_key]
            else:
                mask = line_mask(start, end)
                masks[key] = mask
            
            if mask is None: return False
            tl = np.min([start,end], axis=0)
            x_slice = slice(tl[0], tl[0]+mask.shape[0])
            y_slice = slice(tl[1], tl[1]+mask.shape[1])
            
            image[x_slice, y_slice][mask] = color_blend_on_array(image[x_slice, y_slice][mask], color)
            return True
        
        def draw_circle_on_image(center, mask, color):
            r = mask.shape[0]//2
            x_slice = slice(center[0]-r, center[0]+r+1)
            y_slice = slice(center[1]-r, center[1]+r+1)
            image[x_slice, y_slice][mask] = color_blend_on_array(image[x_slice, y_slice][mask], color)
        
        for index,x in enumerate(pixels):
            color = self.colors[index]
            if type(x)==tuple:
                match x[0]:
                    case 0: # lines
                        for i,s in enumerate(x[1]):
                            draw_line_on_image(s, x[2][i], color)
                    case 1: # circles
                        for c in x[1]:
                            draw_circle_on_image(c, x[2], color)
                    case 2: # grid
                        grid_lines_n = int(np.divide(self.br-self.tl, x[2]).max())+2
                        offset = np.divide(x[1]-self.tl, x[2]).astype(np.int32)
                        for i in range(grid_lines_n):
                            i -= offset
                            target = x[2]*i
                            target = pixelize_points(target, self.tl, self.br, resolution)
                            
                            if target[0]>=0 and target[0]<image.shape[0]:
                                color_blend_on_array(image[target[0],:], color)
                            if target[1]>=0 and target[1]<image.shape[1]:
                                color_blend_on_array(image[:,target[1]], color)
            else: # points
                image[x[:,0],x[:,1]] = color_blend_on_array(image[x[:,0],x[:,1]], color)
        
        if image.any():
            image /= image.max()
            image *= 255
        
        image = image.astype(np.uint8)
        if topright_is_the_positive_corner:
            return np.flip(image.swapaxes(0, 1), axis=0)
        return image


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

