from __points_and_masks import *
from __gui import *

class ColorObj:
    color = None
    color_bg = None # depth -> color transitions to color_bg
    def set_color(self, r=0, g=0, b=0):
        self.color = np.asarray([r,g,b], dtype=np.float64)
        if (self.color>1).any(): self.color /= 255
    def set_color_bg(self, r=0, g=0, b=0):
        self.color_bg = np.asarray([r,g,b], dtype=np.float64)
        if (self.color_bg>1).any(): self.color_bg /= 255
    def get_color_blend(self, ratio):
        if self.color_bg is None: return self.color
        return self.color*(1-ratio) + self.color_bg*ratio
    def get_color_blends(self, ratios):
        if self.color_bg is None: return np.expand_dims(self.color, axis=0)*np.ones((len(ratios), 1))
        return np.expand_dims(self.color, axis=0)*(1-ratios) + np.expand_dims(self.color_bg, axis=0)*ratios

def depths_ratio_perspective(points, observer, tl, br):
    dists = np.linalg.norm(points-observer, axis=1)
    dists /= br[2]-tl[2]
    dists[dists>1] = 1
    return dists.reshape(-1, 1) # 0...1

def depths_ratio(points, observer, tl, br):
    return (points[:,2:3]-observer[2])/(br[2]-observer[2]) # 0...1

def apply_curvilinear_perspective(points, observer, zoom=1.):
    distances = distance(points, observer)
    valid = distances>0
    points[valid,:2] -= observer[:2]
    points[valid,0] /= distances[valid]
    points[valid,1] /= distances[valid]
    points[valid,:2] *= zoom
    points[valid,:2] += observer[:2]
    
def apply_zoom(points, observer, zoom=1.):
    points[:,:2] -= observer[:2]
    points[:,:2] *= zoom
    points[:,:2] += observer[:2]

class CanvasDrawing(Canvas):
    # RGBA not possible (except for images)
    # and not very scalable
    
    class DrawnObj(ColorObj):
        key = None # int or a set of ints
    class ImageObj(DrawnObj):
        topleft = None
        bottomright = None
        array = None
        image = None
    class TextObj(DrawnObj):
        point = None
        anchor = "nw"
        text = ""
    class PointsObj(DrawnObj):
        points = None
    class LinesObj(DrawnObj):
        starts = None
        ends = None
    class CirclesObj(DrawnObj):
        points = None
        radius = 1
    
    ndim = None
    tl = None
    br = None
    
    yaw = 0. # rotation around y axis
    tilt = 0. # rotation around z axis
    pitch = 0. # rotation around x axis
    
    color = np.zeros(3) # default obj color
    color_bg = np.ones(3)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._panning_init()
        self._zooming_init()
        self.objects = []
    
    def _panning_init(self):
        self.panning = np.zeros(2)
        
        def drag_handler(event):
            prev = self.drag_start_pos[self.canvas]
            now = np.array((event.x, event.y))
            self.drag_start_pos[self.canvas] = now
            self.panning_move(now-prev)
        
        def button_handler(event):
            if self.canvas in self.drag_start_pos: del self.drag_start_pos[self.canvas]
            self.drag_start_pos[self.canvas] = np.array((event.x, event.y))
        
        self.canvas.bind("<Button-1>", button_handler, add="+")
        self.canvas.bind("<Button-2>", self.panning_reset, add="+")
        self.canvas.bind("<B1-Motion>", drag_handler, add="+")
    
    def panning_reset(self, event=None):
        if self.panning.any():
            self.panning *= 0
            self.redraw()
        
    def panning_move(self, move):
        self.panning += move
        for obj in self.objects:
            if type(obj.key) is set:
                for k in obj.key: self.move(k, move)
            else: self.move(obj.key, move)

    def _zooming_init(self):
        self.zoom = 0
        self._zoom = 1.
        def wheel_handler(event):
            d = scroll_delta_translate(event.delta)
            self.zoom = max(min(self.zoom+d, 8), -8)
            self._zoom = 1.2**self.zoom
            self.redraw()
        self.canvas.bind("<MouseWheel>", wheel_handler, add="+")

    ######
    def clear(self):
        for obj in self.objects: self.delete(obj.key)
    
    def reset(self):
        self.tl = self.br = self.ndim = None
        for obj in self.objects: self.delete(obj.key)
        self.objects.clear()
    
    def delete(self, key):
        if key is not None:
            if type(key) is set:
                for k in key: self.canvas.delete(k)
            else: self.canvas.delete(key)
    
    def delete_object(self, obj):
        self.objects.remove(obj)
        self.delete(obj.key)
    
    

    def _check_points(self, points):
        return np.asarray(points, dtype=np.float64)
    
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
    
    
    def image(self, topleft, bottomright, array):
        obj = self.ImageObj()
        obj.topleft = topleft
        obj.bottomright = bottomright
        obj.array = array
        self.update_tl_br(*bounding_box([obj.topleft,obj.bottomright]))
        self.objects.append(obj)
        return obj

    def text(self, point, text, anchor="nw"):
        obj = self.TextObj()
        obj.point = point
        obj.anchor = anchor
        obj.text = text
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
        
    
    def circles(self, points, radius):
        obj = self.CirclesObj()
        obj.points = self._check_points(points)
        obj.radius = radius
        self.update_tl_br(*bounding_box(obj.points-radius))
        self.update_tl_br(*bounding_box(obj.points+radius))
        self.objects.append(obj)
        
        obj.color = self.color
        return obj
    
    def points(self, points):
        obj = self.PointsObj()
        obj.points = self._check_points(points)
        self.update_tl_br(*bounding_box(points))
        self.objects.append(obj)
        
        obj.color = self.color
        return obj

        
    def draw_image(self, obj, observer, zoom, perspective=False):
        tl, br = self.tl, self.br#self.get_rotated_bounds()
        camera_target = np.mean([tl, br], axis=0)
        points = self.rotate_points([obj.topleft, obj.bottomright], camera_target)
        
        if perspective:
            apply_curvilinear_perspective(points, observer, zoom)
            ratios = depths_ratio_perspective(points, observer, tl, br)
        else:
            apply_zoom(points, observer, zoom)
            ratios = depths_ratio(points, observer, tl, br)
        
        valid = (ratios>0).flatten() # in front of camera
        if not valid.any(): return
        
        pixels = pixelize_points(points[:,:2], tl[:2], br[:2], self.shape.max())
        pixels[:,:2] += self.panning.astype(np.int32)
        
        array = obj.array
        # scale image array correctly using topleft and bottomright from obj
        
        obj.image = array_to_imagetk(self.canvas, array)
        obj.key = self.canvas.create_image(*pixels[0], image=obj.image, anchor="nw")
        
    def draw_text(self, obj, observer, zoom, perspective=False):
        tl, br = self.tl, self.br#self.get_rotated_bounds()
        camera_target = np.mean([tl, br], axis=0)
        points = self.rotate_points([obj.point], camera_target)
        
        if perspective:
            apply_curvilinear_perspective(points, observer, zoom)
            ratios = depths_ratio_perspective(points, observer, tl, br)
        else:
            apply_zoom(points, observer, zoom)
            ratios = depths_ratio(points, observer, tl, br)
        
        valid = (ratios>0).flatten() # in front of camera
        if not valid.any(): return
        
        colors = obj.get_color_blends(ratios)
        colors = (colors*255).astype(np.uint16)
        
        pixels = pixelize_points(points[:,:2], tl[:2], br[:2], self.shape.max())
        pixels[:,:2] += self.panning.astype(np.int32)
        
        color = color_as_hex(colors[0])
        obj.key = self.canvas.create_text(*pixels[0], text=obj.text, anchor=obj.anchor, fill=color)

    
    def draw_lines(self, obj, observer, zoom, perspective=False):
        tl, br = self.tl, self.br#self.get_rotated_bounds()
        camera_target = np.mean([tl, br], axis=0)
        starts = self.rotate_points(obj.starts, camera_target)
        ends = self.rotate_points(obj.ends, camera_target)
        
        if perspective:
            apply_curvilinear_perspective(starts, observer, zoom)
            apply_curvilinear_perspective(ends, observer, zoom)
            
            ratios = depths_ratio_perspective((starts+ends)/2, observer, tl, br)
        else:
            apply_zoom(points, observer, zoom)
            ratios = depths_ratio((starts+ends)/2, observer, tl, br)
        
        valid = (ratios>0).flatten() # in front of camera
        if not valid.any(): return
        ratios = ratios[valid]
        starts = starts[valid]
        ends = ends[valid]
        
        colors = obj.get_color_blends(ratios)
        colors = (colors*255).astype(np.uint16)
        
        starts = pixelize_points(starts[:,:2], tl[:2], br[:2], self.shape.max())
        ends = pixelize_points(ends[:,:2], tl[:2], br[:2], self.shape.max())
        
        starts[:,:2] += self.panning.astype(np.int32)
        ends[:,:2] += self.panning.astype(np.int32)
        
        key_set = set()
        for i in np.argsort((starts[:,2]+ends[:,2])/2):
            color = color_as_hex(colors[i])
            key = self.canvas.create_line(*starts[i], *ends[i], fill=color)
            key_set.add(key)
        obj.key = key_set
    
    def draw_circles(self, obj, observer, zoom, perspective=False):
        tl, br = self.tl, self.br#self.get_rotated_bounds()
        camera_target = np.mean([tl, br], axis=0)
        points = self.rotate_points(obj.points, camera_target)
        
        if perspective:
            apply_curvilinear_perspective(points, observer, zoom)
            ratios = depths_ratio_perspective(points, observer, tl, br)
        else:
            apply_zoom(points, observer, zoom)
            ratios = depths_ratio(points, observer, tl, br)
        
        valid = (ratios>0).flatten() # in front of camera
        if not valid.any(): return
        ratios = ratios[valid]
        points = points[valid]
        
        colors = obj.get_color_blends(ratios)
        colors = (colors*255).astype(np.uint16)
        
        pixels = pixelize_points(points[:,:2], tl[:2], br[:2], self.shape.max())
        radius = pixelize_distances([obj.radius], tl[:2], br[:2], self.shape.max())[0]
        
        pixels[:,:2] += self.panning.astype(np.int32)
        
        toplefts = pixels-radius*(1-ratios)
        bottomrights = pixels+radius*(1-ratios)
        
        key_set = set()
        for i in np.argsort(points[:,2])[::-1]:
            color = color_as_hex(colors[i])
            key = self.canvas.create_oval(*toplefts[i], *bottomrights[i], fill=color)
            key_set.add(key)
        obj.key = key_set
            
    def draw_points(self, obj, observer, zoom, perspective=False):
        tl, br = self.tl, self.br#self.get_rotated_bounds()
        camera_target = np.mean([tl, br], axis=0)
        points = self.rotate_points(obj.points, camera_target)
        
        if perspective:
            apply_curvilinear_perspective(points, observer, zoom)
            ratios = depths_ratio_perspective(points, observer, tl, br)
        else:
            apply_zoom(points, observer, zoom)
            ratios = depths_ratio(points, observer, tl, br)
        
        valid = (ratios>0).flatten() # in front of camera
        if not valid.any(): return
        ratios = ratios[valid]
        points = points[valid]
        
        colors = obj.get_color_blends(ratios)
        colors = (colors*255).astype(np.uint16)
        
        pixels = pixelize_points(points[:,:2], tl[:2], br[:2], self.shape.max())
        pixels[:,:2] += self.panning.astype(np.int32)
##        print(self.panning)
        
        valid = np.all(pixels>0, axis=1)
        pixels = pixels[valid]
        colors = colors[valid]
        
        key_set = set()
        for i in np.argsort(points[valid,2]):
            color = color_as_hex(colors[i])
            key = self.canvas.create_rectangle(*pixels[i], *(pixels[i]+1), fill=color)
            key_set.add(key)
        obj.key = key_set
    

    def draw_object(self, obj, *args, **kwargs):
##        print(obj)
        if isinstance(obj, self.PointsObj): self.draw_points(obj, *args, **kwargs)
        elif isinstance(obj, self.LinesObj): self.draw_lines(obj, *args, **kwargs)
        elif isinstance(obj, self.CirclesObj): self.draw_circles(obj, *args, **kwargs)
        elif isinstance(obj, self.ImageObj): self.draw_image(obj, *args, **kwargs)
        elif isinstance(obj, self.TextObj): self.draw_text(obj, *args, **kwargs)


    last_used_distance = None
    def draw(self, distance=None):
        if len(self.objects)==0: return
        self.last_used_distance = distance
        tl, br = self.tl, self.br
        observer_target = np.mean([tl,br], axis=0)
        observer = observer_target.copy()
        zoom = self._zoom
        perspective = distance is not None
        if perspective: observer[2] = tl[2]-distance
        else: observer[2] -= br[2]-tl[2]
        self.clear() # clear first -> otherwise creates orphans
        for obj in self.objects:
            self.draw_object(obj, observer, zoom, perspective)

    def redraw(self):
        self.draw(self.last_used_distance)
    
    def get_rotated_bounds(self):
        bounds = np.array([self.tl, self.br])
        corners = np.array(list(bounding_corners(bounds)))
        rotated_corners = self.rotate_points(corners, center=np.mean(bounds, axis=0))
        return bounding_box(rotated_corners)
    
    def rotate_points(self, points, center, depth=0):
        p_ndim = len(points[0])
        
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
    
    def get_full_extent(self):
        tl, br = self.get_rotated_bounds()
        extent = []
        for i in range(self.ndim):
            extent.append(tl[i])
            extent.append(br[i])
        return extent
    
    def get_full_unrotated_extent(self):
        extent = []
        for i in range(self.ndim):
            extent.append(self.tl[i])
            extent.append(self.br[i])
        return extent

    def get_axis_lines(self):
        starts = np.zeros((self.ndim,self.ndim))
        starts[:,:] = self.tl
        ends = starts.copy()
        for i in range(self.ndim): ends[i,i] = self.br[i]
        return starts, ends
    
    #









if __name__ == "__main__":
    win = nice_window("test", resizeable=True)
    set_padding(win)
##    win.configure(width=500, height=500)
##    win.pack_propagate(0) # do not let children trigger resizing
    
    topframe = nice_titled_frame(win, "TOPFRAME", fill=tk.BOTH)
    canvas = CanvasDrawing(nice_canvas(topframe, 500, 500))
    
    f = nice_button(canvas.canvas, text="button")
    canvas.window(f, (0,0))
    
    k,f1 = canvas.movable_window("title")
    canvas.coords(k, (100,50))
    b1 = nice_button(f1, text="button")
    
    k,f2 = canvas.movable_window("title2", (100,100))
    b2 = nice_button(f2, text="button")
    
##    canvas.hide_trigger(b1, f2.master)
##    canvas.hide_trigger(b2, f1.master)
    
##    key = canvas.create_image(np.random.random((100,100))*255, (200,50))
##    canvas.move(key, (-200,0))
##    canvas.coords(key, (0,10))
    
    obj = canvas.text((.1,.6,.1), "asd")
    obj.set_color(r=1)
    obj.set_color_bg(b=1)
    canvas.reset()
    
    obj = canvas.circles(np.random.random((100,3)), .1)
    obj.set_color(r=1)
    obj.set_color_bg(*canvas.color_bg)
    obj = canvas.points(np.random.random((100,3)))
    obj.set_color(r=1)
    obj.set_color_bg(*canvas.color_bg)

##    canvas.draw()
    def _draw(event):
##        print(event)
        canvas.yaw += .01#event.delta/4200
        canvas.tilt += .005#event.delta/4200
        canvas.pitch += .03#event.delta/4200
        canvas.draw()
    canvas.canvas.bind("<Motion>", _draw, add="+")

##    last_pan = np.ones(3)
##    def _pan(event):
####        if last_pan[2]: last_pan[2] = 0
####        else: canvas.panning_move(np.subtract(last_pan[:2], (event.x, event.y)))
####        last_pan[0] = event.x
####        last_pan[1] = event.y
##        _draw(event)
    canvas.canvas.bind("<B1-Motion>", _draw, add="+")
    
    win.mainloop()

