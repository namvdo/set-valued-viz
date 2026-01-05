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


class ColorObj:
    color = None
    color_bg = None # depth -> color transitions to color_bg
    def set_color(self, r=0, g=0, b=0, a=1):
        self.color = np.asarray([r,g,b,a], dtype=np.float64)
        if (self.color>1).any(): self.color /= 255
    def set_color_bg(self, r=0, g=0, b=0, a=1):
        self.color_bg = np.asarray([r,g,b,a], dtype=np.float64)
        if (self.color_bg>1).any(): self.color_bg /= 255
    def get_color_blend(self, ratio):
        if self.color_bg is None: return self.color
        return self.color*(1-ratio) + self.color_bg*ratio
    def get_color_blends(self, ratios):
        if self.color_bg is None: return np.expand_dims(self.color, axis=0)*np.ones((len(ratios), 1))
        return np.expand_dims(self.color, axis=0)*(1-ratios) + np.expand_dims(self.color_bg, axis=0)*ratios

class ImageDrawing(ColorObj):
    class PointsObj(ColorObj):
        points = None
    class LinesObj(ColorObj):
        starts = None
        ends = None
    class CirclesObj(ColorObj):
        points = None
        radius = 1
        inside = 0
    class GridObj(ColorObj):
        center = (0,0) # origin
        size = .1
    
    ndim = None
    tl = None
    br = None
    
    yaw = 0. # rotation around y axis
    tilt = 0. # rotation around z axis
    pitch = 0. # rotation around x axis
    
    color = np.array([0,0,0,1.]) # default obj color
    color_bg = np.ones(4)

    plane_depth = None # 2d points are given this value as the depth if not None
    
    def __init__(self):
        self.objects = []

    def clear(self):
        self.tl = self.br = None
        self.objects.clear()
        self.colors.clear()

    def _check_points(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.shape[1]==2 and self.plane_depth is not None:
            return np.pad(points, ((0,0),(0,1)))
        return points

    def update_tl_br(self, tl, br):
        ndim = tl.shape[0]
        if self.ndim is None:
            self.tl = tl
            self.br = br
        else:
            if ndim<self.ndim: # update bounds have lower dimensionality than current
                tl = np.pad(tl, (0,self.ndim-ndim))
                br = np.pad(br, (0,self.ndim-ndim))
                tl[ndim:] = self.tl[ndim:]
                br[ndim:] = self.br[ndim:]
            elif ndim>self.ndim: # update bounds have higher dimensionality than current
                self.tl = np.pad(self.tl, (0,ndim-self.ndim))
                self.br = np.pad(self.br, (0,ndim-self.ndim))
                self.tl[self.ndim:] = tl[self.ndim:]
                self.br[self.ndim:] = br[self.ndim:]
            
            self.tl, self.br = bounding_box([tl,self.tl,br,self.br])
        self.ndim = self.tl.shape[0]
        
    def points(self, points):
        obj = self.PointsObj()
        obj.points = self._check_points(points)
        self.update_tl_br(*bounding_box(points))
        self.objects.append(obj)
        
        obj.color = self.color
        return obj
    
    def lines(self, starts, ends):
        obj = self.LinesObj()
        obj.starts = self._check_points(starts)
        obj.ends = self._check_points(ends)
        self.update_tl_br(*bounding_box(obj.starts))
        self.update_tl_br(*bounding_box(obj.ends))
        self.objects.append(obj)
        
        obj.color = self.color
        return obj
    
    def circles(self, points, radius, inside=0):
        obj = self.CirclesObj()
        obj.points = self._check_points(points)
        obj.radius = radius
        obj.inside = inside
        self.update_tl_br(*bounding_box(obj.points-radius))
        self.update_tl_br(*bounding_box(obj.points+radius))
        self.objects.append(obj)
        
        obj.color = self.color
        return obj

    def grid(self, center, size):
        obj = self.GridObj()
        obj.center = self._check_points([center])[0]
        obj.size = size
        self.objects.append(obj)
        
        obj.color = self.color
        return obj


    def calc_perspective_zoom(self, camera_pos):
        corners = np.array(list(bounding_corners(np.array([self.tl, self.br]))))
        apply_curvilinear_perspective(corners, camera_pos, zoom=1.)
        return distance(self.tl, self.br)/distance(*bounding_box(corners))
        
    
    
    def draw(self, resolution:int, topright_is_the_positive_corner:bool=True, camera_dist:float=None):
        tl, br = self.get_rotated_bounds()
        camera_orbit_point = np.mean([tl,br], axis=0)
        
        # static camera position == negative z axis
        can_show_depth = self.ndim>2
        can_do_perspective = camera_dist is not None and can_show_depth
        camera_pos = camera_orbit_point.copy()
        if can_show_depth:
            camera_pos[2] = tl[2]
        if can_do_perspective:
            camera_pos[2] -= camera_dist
            zoom = 1
##            zoom /= self.calc_perspective_zoom(camera_pos)
            zoom += tl[2]-camera_pos[2]
##            zoom *= distance(tl, br)/distance(self.tl, self.br)
##            zoom **= 2
##            print(zoom)
        
        def depth_ratio_array(points):
            if can_do_perspective:
                # by distance instead
                dists = np.linalg.norm(points-camera_pos, axis=1)
                dists -= camera_dist
                dists /= br[2]-tl[2]
                return dists.reshape(-1, 1)
            return (points[:,2:3]-camera_pos[2])/(br[2]-camera_pos[2]) # 0...1
        #
        
        limits = pixelize_points(br[:2], tl[:2], br[:2], resolution)
        image = np.zeros((*image_shape(resolution, *limits), self.color_bg.shape[-1]))
        image[:,:] = self.color_bg
        
        def color_blend_on_array(array, color):
            if color[3]<1:
                array[:,:3] = color[:3]*color[3] + array[:,:3]*array[:,3:4]*(1-color[3])
                array[:,3] = 1-(1-array[:,3])*(1-color[3])
                array[:,:3] /= array[:,3:4]
            else:
                array[:] = color
            return array
        
        def color_array_blend_on_array(array, color):
            array[:,:3] = color[:,:3]*color[:,3:4] + array[:,:3]*array[:,3:4]*(1-color[:,3:4])
            array[:,3] = 1-(1-array[:,3])*(1-color[:,3])
            array[:,:3] /= array[:,3:4]
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
            if isinstance(x, self.PointsObj):
                points = self.rotate_points(x.points, camera_orbit_point)
                
                if can_do_perspective:
                    apply_curvilinear_perspective(points, camera_pos, zoom)
                    
                pixels = pixelize_points(points[:,:2], tl[:2], br[:2], resolution)
                if can_show_depth:
                    ratios = depth_ratio_array(points)
                    color = x.get_color_blends(ratios)
                    image[pixels[:,0],pixels[:,1]] = color_array_blend_on_array(image[pixels[:,0],pixels[:,1]], color)
                else:
                    image[pixels[:,0],pixels[:,1]] = color_blend_on_array(image[pixels[:,0],pixels[:,1]], x.color)
                
            elif isinstance(x, self.LinesObj):
                points_starts = self.rotate_points(x.starts, camera_orbit_point)
                points_ends = self.rotate_points(x.ends, camera_orbit_point)
                
                if can_do_perspective:
                    apply_curvilinear_perspective(points_starts, camera_pos, zoom)
                    apply_curvilinear_perspective(points_ends, camera_pos, zoom)
                    
                starts = pixelize_points(points_starts[:,:2], tl[:2], br[:2], resolution)
                ends = pixelize_points(points_ends[:,:2], tl[:2], br[:2], resolution)
                
                i = 0
                if can_show_depth:
                    ratios = depth_ratio_array((points_starts+points_ends)/2)
                    color = x.get_color_blends(ratios)
                    for s in starts:
                        draw_line_on_image(s, ends[i], color[i])
                        i += 1
                else:
                    for s in starts:
                        draw_line_on_image(s, ends[i], x.color)
                        i += 1
                
            elif isinstance(x, self.CirclesObj):
                points = self.rotate_points(x.points, camera_orbit_point)
                
                if can_do_perspective:
                    apply_curvilinear_perspective(points, camera_pos, zoom)
                
                pixels = pixelize_points(points[:,:2], tl[:2], br[:2], resolution)
                radii = pixelize_distances([x.radius, x.inside], tl[:2], br[:2], resolution)
                mask = circle_mask(*radii)
                if can_show_depth:
                    ratios = depth_ratio_array(points)
                    color = x.get_color_blends(ratios)
                    i = 0
                    for p in pixels:
                        draw_circle_on_image(p, mask, color[i])
                        i += 1
                else:
                    for p in pixels:
                        draw_circle_on_image(p, mask, x.color)
                
            elif isinstance(x, self.GridObj):
                color = x.color
                grid_lines_n = int(np.divide(br[:2]-tl[:2], x.size).max())+2
                offset = np.divide(x.center-tl[:2], x.size).astype(np.int32)
                for i in range(grid_lines_n):
                    i -= offset
                    target = x.size*i
                    target = pixelize_points(target, tl[:2], br[:2], resolution)
                    
                    if target[0]>=0 and target[0]<image.shape[0]:
                        color_blend_on_array(image[target[0],:], color)
                    if target[1]>=0 and target[1]<image.shape[1]:
                        color_blend_on_array(image[:,target[1]], color)
        
        if image.any():
            image *= 255
            image = np.clip(image, a_min=0, a_max=255)
        
        image = image.astype(np.uint8)
        if topright_is_the_positive_corner:
            return np.flip(image.swapaxes(0, 1), axis=0)
        return image

    def draw_to_plot(self, plot, *args, **kwargs):
        plot.imshow(self.draw(*args, **kwargs), extent=self.get_extent())

    def test_draw(self, *args, **kwargs):
        print("test draw", self)
        self.draw_to_plot(plt, *args, **kwargs)
        plt.show()

    
    def get_rotated_bounds(self):
        bounds = np.array([self.tl, self.br])
        corners = np.array(list(bounding_corners(bounds)))
        rotated_corners = self.rotate_points(corners, center=np.mean(bounds, axis=0))
        return bounding_box(rotated_corners)
    
    def rotate_points(self, points, center, depth=0):
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

    def get_axis_lines(self):
        starts = np.zeros((self.ndim,self.ndim))
        starts[:,:] = self.tl
        ends = starts.copy()
        for i in range(self.ndim): ends[i,i] = self.br[i]
        return starts, ends
    
    def set_isometric_rotation(self):
        self.pitch = -np.pi/4
        self.yaw = np.pi/5.1
        self.tilt = -np.pi/6

def apply_curvilinear_perspective(points, observer, zoom=1.):
    distances = distance(points, observer)
    valid = distances>0
    points[valid,:2] -= observer[:2]
    points[valid,0] /= 1+distances[valid]
    points[valid,1] /= 1+distances[valid]
    points[valid,:2] *= zoom#+1
    points[valid,:2] += observer[:2]


if __name__ == "__main__":
    drawing = ImageDrawing()
    drawing.set_color_bg(r=1, g=1, b=1)

    points = points_in_gridlike_shape(np.zeros(2), np.ones(2), (10,10))
    points = points.reshape(-1, 2)
    points = np.pad(points, ((0,0),(0,1)), mode="edge")
    
    obj = drawing.lines(*point_lines(points))
    obj.set_color(g=1)
    obj.set_color_bg(b=1, a=.1)
    
    obj = drawing.circles(points, 0.01)
    obj.set_color(r=1)
    obj.set_color_bg(b=0, a=.1)
    
##    circles = np.zeros((500,3))
##    circles[:,:2] = np.random.random((500,2))#+2
##    circles[:,0] *= 2
##    obj = drawing.circles(circles, 0.05)
##    obj.set_color(r=1)
    
##    print((drawing.draw(512) == drawing.draw2(512)).all())
##    drawing.tilt = np.pi/4
##    drawing.test_draw(200)
##    drawing.tilt = np.pi/4
    for plot in plotting_grid(9, 1, (20,4)):
        drawing.draw_to_plot(plot, 1000, camera_dist=0)
        drawing.pitch += np.pi/6
        drawing.tilt += np.pi/8
        drawing.yaw += np.pi/10
    plt.show()
        
        
    
