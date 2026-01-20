from __equation import *
from __points_and_masks import *

class ModelBase(): # 2D
    epsilon = 0.01
    start_point = None
    function = None
    
    def __init__(self):
        self.start_point = [0,0]
        self.function = MappingFunction(x="1-a*x*x+y", y="b*x")
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

class Model(ModelBase):
    gap_detection_attempts = 10
    point_density = 10 # point density (relative to epsilon)
    point_ceiling = 500 # soft ceiling for points (increases allowed distance between points after the ceiling)
    
    def __init__(self):
        self.start_point = np.zeros(3)
        self.function = MappingFunction(x="x", y="y", z="z")
        self.point_density = 10
        self.point_ceiling = 500 # soft ceiling for points (increases allowed distance between points after the ceiling)
    
    def update_start(self, x:float=None, y:float=None, z:float=None):
        if x is not None: self.start_point[0] = x
        if y is not None: self.start_point[1] = y
        if z is not None: self.start_point[2] = z

    def update_function(self, x:str=None, y:str=None, z:str=None):
        if x is not None: self.function.x.string = x
        if y is not None: self.function.y.string = y
        if z is not None: self.function.z.string = z

    def update_constants(self, **constants):
        self.function.set_constants(**constants)

    def copy(self):
        new = type(self)()
        attrs = [
            "epsilon",
            "start_point",
            "function",
            
            "points",
            "normals",
            "prev_points",
            "prev_normals",
            
            "gap_detection_attempts",
            "point_density",
            "point_ceiling",
            ]
        for attr in attrs:
            new.copyattr(self, attr)
        return new
    
    def first_step(self, three_dimensional=False):
        n = self.point_density
        
        if three_dimensional:
            # ball
            self.normals = np.concatenate(point_ball(n, self.epsilon), axis=0)
            self.normals[self.normals.shape[0]//2+1:,0] *= -1
        else:
            # circle
            radians = np.linspace(0, np.pi*2, n+1)[:-1]
            self.normals = radians_to_vectors(radians)*self.epsilon
        
        self.points = np.zeros_like(self.normals)
        self.points[:] = self.start_point[:self.points.shape[1]]
        self.prev_points = self.points.copy()
        self.prev_normals = self.normals.copy()
    
    def next_step(self):
        points, normals = self.__process_step(self.points, self.normals)
        if points is None: return False
        self.prev_points[:] = self.points
        self.prev_normals[:] = self.normals
        self.points, self.normals = points, normals
        
        if self.points.shape[1]==3:
            # TODO: gap detection for 3D objects
            pass
        else:
            self._gap_detection_loops_2D()
        return True
    
    def __process_step(self, input_points, input_normals):
        ndim = input_points.shape[1]
        solve_inputs = self.function.constants.copy()
        points = input_points.copy()
        normals = input_normals.copy()
        
        # solve transposed inverse jacobian matrix for current points
        solve_inputs["x"] = points[:,0]
        solve_inputs["y"] = points[:,1]
        
        if ndim==3:
            solve_inputs["z"] = points[:,2]
            tij = self.function.tij3D()
        else: tij = self.function.tij2D()
        if tij is None: # matrix failed
            return None, None
        
        output = solve_matrix(tij, **solve_inputs)
        
        # calculate new normals by rotating prev_normals using the solved matrix
        if ndim==3:
            normals[:,0] = (input_normals[:,0]*output[0][0]+input_normals[:,1]*output[0][1]+input_normals[:,2]*output[0][2])
            normals[:,1] = (input_normals[:,0]*output[1][0]+input_normals[:,1]*output[1][1]+input_normals[:,2]*output[1][2])
            normals[:,2] = (input_normals[:,0]*output[2][0]+input_normals[:,1]*output[2][1]+input_normals[:,2]*output[2][2])
        else:
            normals[:,0] = (input_normals[:,0]*output[0][0]+input_normals[:,1]*output[0][1])
            normals[:,1] = (input_normals[:,0]*output[1][0]+input_normals[:,1]*output[1][1])
        
        # scale the normals to unit length
        lengths = np.expand_dims(np.linalg.norm(normals, axis=1), axis=1)
        zero_mask = (lengths == 0).flatten()
        if zero_mask.any():
            lengths[zero_mask] = 1  
            normals[zero_mask] = input_normals[zero_mask]

        normals /= lengths
        # scale the normals to epsilon length
        normals *= self.epsilon
        
        # do the function mapping and add the normals to the mapped points
        if ndim==3:
            points[:,0], points[:,1], points[:,2] = self.function(x=points[:,0], y=points[:,1], z=points[:,2])
        else:
            points[:,0], points[:,1], _ = self.function(x=points[:,0], y=points[:,1])
        
        points += normals
        return points, normals
    
    
    def get_boundary_lines(self):
        return point_lines(self.points)
    
    def get_inner_normals(self, length=1):
        return self.points, self.points-self.normals*length
    
    def get_outer_normals(self, length=1):
        return self.points, self.points+self.normals*length




    def _remove_items_with_mask(self, mask):
        self.points = self.points[mask]
        self.normals = self.normals[mask]
        self.prev_points = self.prev_points[mask]
        self.prev_normals = self.prev_normals[mask]
    
    def _min_distance_between_points(self):
        # calculate the minimum distance between points allowed for gap detection
        # -> keeps the point amount from rising impossibly high
        e = self.epsilon
        n = len(self.points)
        c = self.point_ceiling
        e *= n/c
        e /= self.point_density
        return e
    
    def _trim_inside_mask(self): # n-dimensional
        d = self.epsilon*.98
        inner_points = self.points-self.normals
        inner_points = np.repeat(np.expand_dims(inner_points, axis=0), len(self.points), axis=0)
        diffs = np.subtract(inner_points, np.expand_dims(self.points, axis=1))
        return np.any(np.linalg.norm(diffs, axis=2)<d, axis=1)
    
    def _too_sparse_mask_2D(self):
        d = self._min_distance_between_points()
        dist = np.linalg.norm(np.diff(self.points, axis=0, append=self.points[:1]), axis=1)
        mask = dist>(d*2)
        return mask

    def _too_dense_mask_2D(self):
        d = self._min_distance_between_points()
        dist = np.linalg.norm(np.diff(self.points, axis=0, append=self.points[:1]), axis=1)
        mask = dist<d
        mask = repeat_mask_ones_until_divisible(mask, 2)
        mask[1::2] = False
        return mask
        
    def _increase_precision_2D(self, gap_mask):
        # create extra points to fill the model with using previous step's points
        
        # create a average point between two points, that were marked as a gap
        # do the same with normals
        prev_points_wraparound = np.concatenate([self.prev_points, self.prev_points[:1]])
        starts = prev_points_wraparound[:-1][gap_mask]
        ends = prev_points_wraparound[1:][gap_mask]
        more_points = (starts+ends)/2
        
        prev_normals_wraparound = np.concatenate([self.prev_normals, self.prev_normals[:1]])
        starts = prev_normals_wraparound[:-1][gap_mask]
        ends = prev_normals_wraparound[1:][gap_mask]
        more_normals = (starts+ends)/2
        #
        
        self.prev_points = np.append(self.prev_points, more_points, axis=0)
        self.prev_normals = np.append(self.prev_normals, more_normals, axis=0)
        
        # process the created points and normals to get them caught up to current step
        points, normals = self.__process_step(self.prev_points, self.prev_normals)
        
        # create sorting array so that the model points are in correct order
        sorting = np.arange(len(points))
        l = len(self.points)
        for index,replace in enumerate(np.arange(l)[gap_mask]):
            replace += index+1
            sorting[replace+1:] = sorting[replace:-1]
            sorting[replace] = l+index
        
        # apply sorting array
        self.points = points[sorting]
        self.normals = normals[sorting]
        self.prev_points = self.prev_points[sorting]
        self.prev_normals = self.prev_normals[sorting]


    def _gap_detection_loops_2D(self):
        point_delta = 0
        for i in range(self.gap_detection_attempts):
            gaps = self._too_sparse_mask_2D()
            if gaps.any():
                point_delta += gaps.sum()
                self._increase_precision_2D(gaps)
            else: break
        
        for i in range(self.gap_detection_attempts):
            nogaps = self._too_dense_mask_2D()
            if nogaps.any():
                point_delta -= nogaps.sum()
                self._remove_items_with_mask(~nogaps)
            else: break

        inside = self._trim_inside_mask()
        if not inside.all() and inside.any():
            point_delta -= inside.sum()
            self._remove_items_with_mask(~inside)
    
    

if __name__ == "__main__":
    from _quick_visuals import *
    
    def test_draw(plot, model, resolution:int):
        drawing = ImageDrawing()
        drawing.plane_depth = 0
        
        drawing.set_color(r=0, g=0, b=0)
        drawing.set_color_bg(r=1, g=1, b=1)
        
        obj = drawing.circles(model.points, model.epsilon/100)
##        obj = drawing.points(model.points)
        obj.set_color(r=1, a=.5)
        obj.set_color_bg(b=1, a=.5)
        
##        obj = drawing.lines(*model.get_boundary_lines())
##        obj.set_color(g=1)
##        obj.set_color_bg(b=1, a=.5)
        
        obj = drawing.lines(*model.get_inner_normals())
        obj.set_color(r=1, a=.9)
        obj.set_color_bg(b=1, a=.1)
##        drawing.lines(*model.get_outer_normals())

##        drawing.set_isometric_rotation()
        drawing.draw_to_plot(plot, resolution)
        
    
    model = Model()
##    model.update_function(x="1-a*x*x+y", y="b*x")
    model.update_function(x="1-a*x*x+y+z", y="b*x", z="z")
    model.epsilon = 0.07
    model.update_constants(a=0.8, b=0.3)
    model.point_density = 10
    model.first_step(0)
    
    resolution = 1000
    timestep = 0
    while 1:
##        model.next_step()
##        input()
        for plot in plotting_grid(1, 1):
            print(model.next_step())
##            print(timestep)
            test_draw(plot, model, resolution)
            plot.set_title(f"step: {timestep}")
            timestep += 1
        plt.show()
    pass




