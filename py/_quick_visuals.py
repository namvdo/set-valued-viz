import numpy as np
import matplotlib.pyplot as plt

from __points_and_masks import *

from __system import function_timer

def plotting_grid(width=1, height=1, figsize=(8,8)):
    fig,ax = plt.subplots(height, width, figsize=figsize)
    for i in range(width):
        for j in range(height):
            if width>1 and height>1: ax_target = ax[i][j]
            elif width>1 or height>1: ax_target = ax[i+j]
            else: ax_target = ax
            yield ax_target


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
    
    ndim = 0
    
    yaw = 0. # rotation around y axis
    tilt = 0. # rotation around z axis
    pitch = 0. # rotation around x axis
    
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
        ndim = tl.shape[0]
        if ndim<self.ndim: # bounds were lower dimensional
            tl = np.pad(tl, (0,self.ndim-ndim))
            br = np.pad(br, (0,self.ndim-ndim))
            tl[ndim:] = self.tl[ndim:]
            br[ndim:] = self.br[ndim:]
            
        if self.tl is None: self.tl = tl
        else: self.tl = np.min([tl,self.tl], axis=0)
        if self.br is None: self.br = br
        else: self.br = np.max([br,self.br], axis=0)
        self.ndim = self.tl.shape[0]
        
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
        tl, br = self.get_rotated_bounds()
        camera_pos = np.mean([tl,br], axis=0)
        
        limits = pixelize_points(br, tl, br, resolution)
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
            line_tl = np.min([start,end], axis=0)
            x_slice = slice(line_tl[0], line_tl[0]+mask.shape[0])
            y_slice = slice(line_tl[1], line_tl[1]+mask.shape[1])
            
            image[x_slice, y_slice][mask] = color_blend_on_array(image[x_slice, y_slice][mask], color)
            return True
        
        def draw_circle_on_image(center, mask, color):
            r = mask.shape[0]//2
            x_slice = slice(center[0]-r, center[0]+r+1)
            y_slice = slice(center[1]-r, center[1]+r+1)
            image[x_slice, y_slice][mask] = color_blend_on_array(image[x_slice, y_slice][mask], color)
        
        for index,x in enumerate(self.objects):
            color = self.colors[index]
            if isinstance(x, self.LinesObj):
                starts = self.rotate_points(x.starts, camera_pos)
                ends = self.rotate_points(x.ends, camera_pos)
                starts = pixelize_points(starts, tl, br, resolution)
                ends = pixelize_points(ends, tl, br, resolution)
                for i,s in enumerate(starts):
                    draw_line_on_image(s, ends[i], color)
                
            elif isinstance(x, self.CirclesObj):
                centers = self.rotate_points(x.points, camera_pos)
                centers = pixelize_points(centers, tl, br, resolution)
                radii = pixelize_distances([x.radius, x.inside], tl, br, resolution)
                mask = circle_mask(*radii)
                for c in centers:
                    draw_circle_on_image(c, mask, color)
                
            elif isinstance(x, self.GridObj):
                grid_lines_n = int(np.divide(br-tl, x.size).max())+2
                offset = np.divide(x.center-tl, x.size).astype(np.int32)
                for i in range(grid_lines_n):
                    i -= offset
                    target = x.size*i
                    target = pixelize_points(target, tl, br, resolution)
                    
                    if target[0]>=0 and target[0]<image.shape[0]:
                        color_blend_on_array(image[target[0],:], color)
                    if target[1]>=0 and target[1]<image.shape[1]:
                        color_blend_on_array(image[:,target[1]], color)    
                
            elif type(x)==np.ndarray:
                x = self.rotate_points(x, camera_pos)
                pixels = pixelize_points(x, tl, br, resolution)
                image[pixels[:,0],pixels[:,1]] = color_blend_on_array(image[pixels[:,0],pixels[:,1]], color)
        
        if image.any():
            image /= image.max()
            image *= 255
        
        image = image.astype(np.uint8)
        if topright_is_the_positive_corner:
            return np.flip(image.swapaxes(0, 1), axis=0)
        return image

    def draw_to_plot(self, plot, resolution:int):
        plot.imshow(self.draw(resolution), extent=self.get_extent())

    def test_draw(self, resolution:int):
        print("test draw", self)
        self.draw_to_plot(plt, resolution)
        plt.show()

    
    def get_rotated_bounds(self):
        bounds = np.array([self.tl, self.br])
        corners = np.array(list(bounding_corners(bounds)))
        rotated_corners = self.rotate_points(corners, center=np.mean(bounds, axis=0))
        return bounding_box(rotated_corners)
    
    def rotate_points(self, points, center):
        p_ndim = points.shape[1]
        
        if p_ndim<self.ndim:
            new_points = np.pad(points, ((0,0),(0,self.ndim-p_ndim)))
            new_points[:,p_ndim:] = center[p_ndim:] # fill with center values
        else:
            new_points = points.copy()
        
        new_points -= center
        if self.ndim>2: # 3D
            new_points = rotate_vectors_3d(new_points, self.pitch, self.yaw, self.tilt)
        else:
            new_points = rotate_vectors(new_points, self.tilt)
        new_points += center
        return new_points
    
    def get_unrotated_extent(self):
        return (self.tl[0], self.br[0], self.tl[1], self.br[1])

    def get_extent(self):
        tl, br = self.get_rotated_bounds()
        return (tl[0], br[0], tl[1], br[1])



if __name__ == "__main__":
    drawing = ImageDrawing(r=1, g=1, b=1)
    drawing.points(np.random.random((100,3)))
    drawing.lines(*point_lines(np.random.random((100,3))), b=1)
    drawing.circles(np.random.random((100,2))+2, 0.05, r=1)
    
##    print((drawing.draw(512) == drawing.draw2(512)).all())
##    drawing.tilt = np.pi/4
##    drawing.test_draw(200)
##    drawing.tilt = np.pi/4
    for plot in plotting_grid(9, 1, (20,4)):
        drawing.draw_to_plot(plot, 200)
        drawing.yaw += np.pi/8
        drawing.pitch += np.pi/8
    plt.show()
        
        
    
