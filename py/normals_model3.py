from _imports import *


class Model(ModelBase3D): # WIP
    printing = True
    print_func = print
    
    gap_detection_attempts = 10
    
    point_density = 10 # point density (relative to epsilon)
    point_ceiling = 500 # soft ceiling for points (increases allowed distance between points after the ceiling)
    
    hausdorff_dist = 0
    hausdorff_line = None
    
    def __init__(self):
        super().__init__()
        self.reset()

    def __len__(self): return len(self._points)

    def active(self): return self._timestep is not None
    
    def has_points(self): return self._points.size>0
    
    def reset(self):
        self.tij = None
        self._timestep = None
        self._points = np.zeros((0,2))
        self._normals = np.zeros((0,2))
        self._prev_points = np.zeros((0,2))
        self._prev_normals = np.zeros((0,2))
        self.hausdorff_dist = 0
        self.hausdorff_line = None

    def copy(self):
        new = type(self)()
        attrs = [
            "epsilon",
            "start_point",
            "function",
            
            "printing",
            "print_func",
            
            "gap_detection_attempts",

            "tij",
            "_timestep",
            
            "_points",
            "_normals",
            "_prev_points",
            "_prev_normals",

            "point_density",
            "point_ceiling",
            
            "hausdorff_dist",
            "hausdorff_line",
            ]
        for attr in attrs:
            new.copyattr(self, attr)
        return new
    
    def _remove_items_with_mask(self, removal_mask):
        self._points = self._points[~removal_mask]
        self._normals = self._normals[~removal_mask]
        self._prev_points = self._prev_points[~removal_mask]
        self._prev_normals = self._prev_normals[~removal_mask]

    def _min_distance_between_points(self):
        # calculate the minimum distance between points allowed for gap detection
        # -> keeps the point amount from rising impossibly high
        e = self.epsilon
        
        n = len(self._points)
        c = self.point_ceiling
        e *= n/c
        
        e /= self.point_density
        return e
    
    def _too_sparse_mask(self):
        d = self._min_distance_between_points()
        dist = np.linalg.norm(np.diff(self._points, axis=0, append=self._points[:1]), axis=1)
        mask = dist>(d*2)
        return mask

    def _too_dense_mask(self):
        d = self._min_distance_between_points()
        dist = np.linalg.norm(np.diff(self._points, axis=0, append=self._points[:1]), axis=1)
        mask = dist<d
        mask = repeat_mask_ones_until_divisible(mask, 2)
        mask[1::2] = False
        return mask

    
    def _points_inside_the_boundary_mask(self):
        # create a mask that marks every point that is inside the boundary
        # -> when a point is too close to the inside shape (fails when points are far inside the shape)
        d = self.epsilon*.95
        inner_points = self._points-self._normals
        inner_points = np.repeat(np.expand_dims(inner_points, axis=0), len(self._points), axis=0)
        diffs = np.subtract(inner_points, np.expand_dims(self._points, axis=1))
        return np.any(np.linalg.norm(diffs, axis=2)<d, axis=1)
    
    
    def _increase_precision(self, gap_mask):
        # create extra points to fill the model with using previous step's points
        
        # create a average point between two points, that were marked as a gap
        # do the same with normals
        prev_points_wraparound = np.concatenate([self._prev_points, self._prev_points[:1]])
        starts = prev_points_wraparound[:-1][gap_mask]
        ends = prev_points_wraparound[1:][gap_mask]
        more_points = (starts+ends)/2
        
        prev_normals_wraparound = np.concatenate([self._prev_normals, self._prev_normals[:1]])
        starts = prev_normals_wraparound[:-1][gap_mask]
        ends = prev_normals_wraparound[1:][gap_mask]
        more_normals = (starts+ends)/2
        #
        
        points = np.append(self._prev_points, more_points, axis=0)
        normals = np.append(self._prev_normals, more_normals, axis=0)
        prev_points = np.zeros_like(points)
        prev_normals = np.zeros_like(normals)

        # process the created points and normals to get them caught up to current step
        for _ in self._process(points, normals, prev_points, prev_normals, 1): pass

        # create sorting array so that the model points are in correct order
        sorting = np.arange(len(self._points)+len(more_points))
        l = len(self._points)
        for index,replace in enumerate(np.arange(l)[gap_mask]):
            replace += index+1
            sorting[replace+1:] = sorting[replace:-1]
            sorting[replace] = l+index
        
        # apply sorting array
        self._points = points[sorting]
        self._normals = normals[sorting]
        self._prev_points = prev_points[sorting]
        self._prev_normals = prev_normals[sorting]


    def _gap_detection_loops(self):
        point_delta = 0
        for i in range(self.gap_detection_attempts):
            gaps = self._too_sparse_mask()
            if gaps.any():
                point_delta += gaps.sum()
                self._increase_precision(gaps)
            else: break
        
        for i in range(self.gap_detection_attempts):
            nogaps = self._too_dense_mask()
            if nogaps.any():
                point_delta -= nogaps.sum()
                self._remove_items_with_mask(nogaps)
            else: break

        inside = self._points_inside_the_boundary_mask()
        if not inside.all() and inside.any():
            point_delta -= inside.sum()
            self._remove_items_with_mask(inside)

        if self.printing: self.print_func(f"point delta: {point_delta:+d}")

        
    def _start(self, to_timestep:int):
        self.reset()
        self._timestep = 0
        
        # create the initial points & normals
        radians = np.linspace(0, np.pi*2, self.point_density*10)[:-1]
        
        self._points = np.repeat([self.start_point], radians.size, axis=0).astype(np.float64)
        self._prev_points = self._points.copy()
        
        self._normals = radians_to_vectors(radians)
        
        self._normals *= self.epsilon
        self._prev_normals = self._normals.copy()
        #

        # map the points using the function
        self._points[:,0], self._points[:,1] = self.function(x=self._points[:,0], y=self._points[:,1])
        
        self._points += self._normals # add the normals to extend the shape outwards
        
        self._gap_detection_loops()
        self.__hausdorff_update(self._points, self._prev_points)

        # yield integers after every fully processed step
        yield self._timestep
        yield from self._continue(to_timestep=to_timestep)

    def _continue(self, to_timestep:int):
        catch_up_amount = to_timestep - self._timestep
        if catch_up_amount>0:
            # yield integers after every fully processed step
            yield from self._process(self._points, self._normals, self._prev_points, self._prev_normals, catch_up_amount)
    
    def _process(self, points, normals, prev_points, prev_normals, amount:int):
        if points is self._points or self.tij is None:
            # define transposed inverse jacobian matrix to model memory
            # -> wont have to redefine it every precision increase call, only every full step processing
            self.tij = self.function.tij2D()
        
        solve_inputs = self.function.constants
        for i in range(amount):
            prev_points[:] = points
            prev_normals[:] = normals
            
            # solve transposed inverse jacobian matrix for current points
            #   |[a, b]|
            #   |[c, d]|
            solve_inputs["x"] = points[:,0]
            solve_inputs["y"] = points[:,1]
            
            output = solve_matrix(self.tij, **solve_inputs)
            
            # calculate new normals by rotating prev_normals using the solved matrix
            normals[:,0] = (prev_normals[:,0]*output[0][0]+prev_normals[:,1]*output[0][1])
            normals[:,1] = (prev_normals[:,0]*output[1][0]+prev_normals[:,1]*output[1][1])
            
            # scale the normals to unit length
            lengths = np.expand_dims(np.linalg.norm(normals, axis=1), axis=1)
            normals /= lengths
            
            # scale the normals to epsilon length
            normals *= self.epsilon
            
            # do the function mapping and add the normals to the mapped points
            points[:,0], points[:,1] = self.function(x=points[:,0], y=points[:,1])
            
            points += normals
            
            if points is self._points: # every point is being processed
                self._timestep += 1 # increase early so that precision increase can catch up to correct step
                
                self._gap_detection_loops()
                # hausdorff most effective here, where prev and current points exist in full
                self.__hausdorff_update(self._points, prev_points)
                
                # amount of points might have changed -> reload
                points = self._points
                prev_points = self._prev_points
                normals = self._normals
                prev_normals = self._prev_normals
                
                # yield integers after every fully processed step
                yield self._timestep

##    @function_timer
    def __hausdorff_update(self, points1, points2):
        if points1.size>0 and points2.size>0:
##            dist, line = hausdorff_distance4(points1, points2)
##            dist, line = hausdorff_distance5(points1, points2)
##            dist, line = hausdorff_distance6(points1, points2)
            dist, line = hausdorff_distance8(points1, points2)
        else:
            dist, line = 0, None
        self.hausdorff_dist, self.hausdorff_line = dist, line
        

    def process(self, to_timestep:int):
        # progress the model to target timestep, either from the start or continuing forward
        if self.printing:
            self.print_func(f"processing: {self._timestep} -> {to_timestep}")
        if self._timestep is None or self._timestep>to_timestep:
            yield from self._start(to_timestep)
        else:
            yield from self._continue(to_timestep)
    
    def get_boundary_lines(self):
        return point_lines(self._points)
    
    def get_inner_normals(self, length=1):
        return self._points, self._points-self._normals*length
    
    def get_outer_normals(self, length=1):
        return self._points, self._points+self._normals*length

    def get_prev_inner_normals(self, length=1):
        return self._prev_points, self._prev_points-self._prev_normals*length
    


if __name__ == "__main__":
    from _quick_visuals import *
    
    def test_draw(plot, model, resolution:int):
        #
        draw_prev_points = 1
        draw_prev_normals = 0
        draw_boundary_lines = 1
        draw_outer_normals = 0
        draw_inner_normals = 1
        #
        
        drawing = ImageDrawing()
        drawing.set_color(r=0, g=0, b=0)
        drawing.set_color_bg(r=1, g=1, b=1)
        
##        drawing.circles([(0,0)], 1, *red)
        obj = drawing.grid((0,0), .25)
        obj.set_color(b=1, a=0.2)
        
        if draw_boundary_lines:
            obj = drawing.lines(*model.get_boundary_lines())
            obj.set_color(g=1)
        
        if draw_inner_normals:
            drawing.lines(*model.get_inner_normals())
        if draw_outer_normals:
            drawing.lines(*model.get_outer_normals())
        
        if draw_prev_points:
            obj = drawing.points(model._prev_points)
            obj.set_color(b=1)
            if draw_prev_normals:
                drawing.lines(*model.get_prev_inner_normals())
        
        obj = drawing.points(model._points)
        obj.set_color(r=1)

        line = model.hausdorff_line
        obj = drawing.lines([line[0]], [line[1]])
        obj.set_color(r=1, g=.5)
        
##        drawing.circles(model._points[:1], model.epsilon/10, inside=model.epsilon/11, r=1)
        
        drawing.draw_to_plot(plot, resolution)
        
    
    model = Model()
    model.start_point[0] = 0
    model.start_point[1] = 0
    model.epsilon = 0.0625
    model.function.set_constants(a=0.6, b=0.3)

    print(model.function)
    
##    model.function.x.string = "x/2+(1-y)/3*x/4"
##    model.function.y.string = "y/3+x/3"  # 
    
##    model.function.x.string = "x/3+cos(y)**2"
##    model.function.y.string = "y/2+sin(x)**2"
    
##    model.function.x.string = "x/2-y/3"
##    model.function.y.string = "y/2+x/5"
    
##    model.function.x.string = "x/2-y/3"
##    model.function.y.string = "y/2+x/3"
    
##    model.function.x.string = "x/2+1/(y**2+1)"
##    model.function.y.string = "y/(x**2+1)"

##    tij = model.function.transposed_inverse_jacobian()
##    print(tij[0][0])
##    print(tij[0][1])
##    print(tij[1][0])
##    print(tij[1][1])
##    print("")
    
    resolution = 2000
    timestep = 1
    while 1:
        for plot in plotting_grid(2, 2):
            for i in model.process(timestep): pass
            test_draw(plot, model, resolution)
            plot.set_title(f"step: {timestep}\ndist: {model.hausdorff_dist}")
            timestep += 1
        plt.show()
    




