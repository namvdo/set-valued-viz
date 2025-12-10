from _imports import *


def noise_geometry_multipliers(normals, polygon):
    multipliers = np.ones((len(normals),1))
    origin = np.zeros(2)
    for i,normal in enumerate(normals):
        result_dist = 1
        line_dist = 2
        prev_vertex = polygon[-1]
        for vertex in polygon:
            if (dist:=np.linalg.norm((vertex*2+prev_vertex)/2-normal))<line_dist:
                temp = find_intersection((prev_vertex, vertex), [origin, normal])
                if (dist2:=np.linalg.norm(temp))<1:
                    result_dist = dist2
                    line_dist = dist
            prev_vertex = vertex
        if result_dist is not None: multipliers[i] = result_dist
    return multipliers

##def nearest_polygon_vertices(normals, polygon):
##    distances = indexes = None
##    for index,vertex in enumerate(polygon):
##        dist = np.linalg.norm(normals-vertex, axis=1)
##        if distances is None:
##            distances = dist
##            indexes = np.zeros(len(normals), dtype=np.int16)
##        else:
##            closer = dist<distances
##            indexes[closer] = index
##            distances[closer] = dist[closer]
##    return indexes


class ModelConfiguration(ModelBase):
    printing = True
    print_func = print
    
    precision = 10 # point density (relative to epsilon)
    precision_ceiling = 500 # soft ceiling for points (increases allowed distance between points after the ceiling)
    precision_ceiling_hardness = 1 # increase to harshen the ceiling effect
    
    precision_increase_attempts = 10
    precision_decrease_attempts = 10
    
    noise_geometry = None
    noise_geometry_sides = 0
    noise_geometry_rotation = 0
    
    def __init__(self):
        super().__init__()
        self.reset()

    def __len__(self): return len(self._points)

    def reset(self):
        self.tij = None
        self._timestep = None
        self._points = np.zeros((0,2))
        self._normals = np.zeros((0,2))
        self._prev_points = np.zeros((0,2))
        self._prev_normals = np.zeros((0,2))
    
    def _remove_items_with_mask(self, removal_mask):
        self._points = self._points[~removal_mask]
        self._normals = self._normals[~removal_mask]
        self._prev_points = self._prev_points[~removal_mask]
        self._prev_normals = self._prev_normals[~removal_mask]

    def _min_distance_between_points(self):
        n = len(self._points)
        c = self.precision_ceiling
        over_the_ceiling_ratio = max((n-c)/c, 0)
        e = self.epsilon
        e /= self.precision
        if over_the_ceiling_ratio>0:
            over_the_ceiling_ratio *= self.precision_ceiling_hardness
            e *= 1+over_the_ceiling_ratio**2
        return e
    
    def _too_sparse_mask(self):
        d = self._min_distance_between_points()
        dist = np.linalg.norm(np.diff(self._points, axis=0, append=self._points[:1]), axis=1)
        return dist>d*2

    def _too_dense_mask(self):
        d = self._min_distance_between_points()
        dist = np.linalg.norm(np.diff(self._points, axis=0, append=self._points[:1]), axis=1)
        mask = dist<d
        mask = repeat_mask_ones_until_divisible(mask, 2)
        mask[1::2] = False
        return mask

    
    def _points_inside_the_boundary_mask(self):
        allowed_distance = self.epsilon*.98
        if self.noise_geometry is not None:
            allowed_distance *= noise_geometry_multipliers(self._normals, self.noise_geometry).min()
        inner_points = self._points-self._normals
        inner_points = np.repeat(np.expand_dims(inner_points, axis=0), len(self._points), axis=0)
        diffs = np.subtract(inner_points, np.expand_dims(self._points, axis=1))
        return np.any(np.linalg.norm(diffs, axis=2)<allowed_distance, axis=1)

    ##
    def update_noise_geometry(self, sides=None, rotation=None):
        if sides is not None: self.noise_geometry_sides = sides
        if rotation is not None: self.noise_geometry_rotation = rotation
        
        if self.noise_geometry_sides>2:
            self.noise_geometry = even_sided_polygon(self.noise_geometry_sides, self.noise_geometry_rotation)
        else:
            self.noise_geometry = None
        
    def try_to_apply_noise_geometry(self, normals):
        if self.noise_geometry is not None:
            normals *= noise_geometry_multipliers(normals, self.noise_geometry)
    ##

    
    
    def _increase_precision(self, gap_mask):
        # go back to previous step, but with extra points
        prev_normals_wraparound = np.concatenate([self._prev_normals, self._prev_normals[:1]])
        starts = prev_normals_wraparound[:-1][gap_mask]
        ends = prev_normals_wraparound[1:][gap_mask]
        more_normals = (starts+ends)/2
        
        prev_points_wraparound = np.concatenate([self._prev_points, self._prev_points[:1]])
        starts = prev_points_wraparound[:-1][gap_mask]
        ends = prev_points_wraparound[1:][gap_mask]
        more_points = (starts+ends)/2
        
        points = np.append(self._prev_points, more_points, axis=0)
        normals = np.append(self._prev_normals, more_normals, axis=0)
        prev_points = np.zeros_like(points)
        prev_normals = np.zeros_like(normals)
        
        self._process(points, normals, prev_points, prev_normals, 1)
        
        sorting = np.arange(len(self._points)+len(more_points))
        l = len(self._points)
        for index,replace in enumerate(np.arange(l)[gap_mask]):
            replace += index+1
            sorting[replace+1:] = sorting[replace:-1]
            sorting[replace] = l+index
        
        self._points = points[sorting]
        self._normals = normals[sorting]
        self._prev_points = prev_points[sorting]
        self._prev_normals = prev_normals[sorting]


    def _gap_detection_loops(self):
        for i in range(self.precision_increase_attempts):
            gaps = self._too_sparse_mask()
            if gaps.any():
                if self.printing: self.print_func(f"points: +{gaps.sum()}")
                self._increase_precision(gaps)
            else: break
        
        for i in range(self.precision_decrease_attempts):
            nogaps = self._too_dense_mask()
            if nogaps.any():
                if self.printing: self.print_func(f"points: -{nogaps.sum()}")
                self._remove_items_with_mask(nogaps)
            else: break

        inside = self._points_inside_the_boundary_mask()
        if not inside.all() and inside.any():
            self._remove_items_with_mask(inside)

        
    def _start(self, to_timestep:int):
        self.reset()
        self._timestep = 0
        
        radians = np.linspace(0, np.pi*2, 128)[:-1]
        
        self._points = np.repeat([self.start_point.as_tuple()], radians.size, axis=0).astype(np.float64)
        self._prev_points = self._points.copy()
        
        self._normals = radians_to_vectors(radians)
        
        self.try_to_apply_noise_geometry(self._normals)
        
        self._normals *= self.epsilon
        self._prev_normals = self._normals.copy()
        
        self._points[:,0], self._points[:,1] = self.function(self._points[:,0], self._points[:,1])
        self._points += self._normals
        
        self._gap_detection_loops()
        
        self._continue(to_timestep=to_timestep)

    def _continue(self, to_timestep:int):
        catch_up_amount = to_timestep - self._timestep
        if catch_up_amount>0:
            self._process(self._points, self._normals, self._prev_points, self._prev_normals, catch_up_amount)

    def _process(self, points, normals, prev_points, prev_normals, amount:int):
        if points is self._points or self.tij is None:
            self.tij = self.function.transposed_inverse_jacobian()
        
        for i in range(amount):
            prev_points[:] = points
            prev_normals[:] = normals
            # |[a, b]|
            # |[c, d]|
            x = points[:,0]
            y = points[:,1]
            a = self.tij[0][0].solve(x=x, y=y, **self.function.constants)
            b = self.tij[0][1].solve(x=x, y=y, **self.function.constants)
            c = self.tij[1][0].solve(x=x, y=y, **self.function.constants)
            d = self.tij[1][1].solve(x=x, y=y, **self.function.constants)
            
            a = np.nan_to_num(a, nan=0, posinf=1, neginf=-1)
            b = np.nan_to_num(b, nan=0, posinf=1, neginf=-1)
            c = np.nan_to_num(c, nan=0, posinf=1, neginf=-1)
            d = np.nan_to_num(d, nan=0, posinf=1, neginf=-1)
            
            # calculate new normals
            normals[:,0] = (prev_normals[:,0]*a+prev_normals[:,1]*b)
            normals[:,1] = (prev_normals[:,0]*c+prev_normals[:,1]*d)
            
            # scale the normal to epsilon
            lengths = np.linalg.norm(normals, axis=1)
            normals[:,0] /= lengths
            normals[:,1] /= lengths
            
            self.try_to_apply_noise_geometry(normals)
            
            normals *= self.epsilon
            
            # add the normals to the new points
            points[:,0], points[:,1] = self.function(x, y)
            
            points += normals

            # alternative gap detection
            if points is self._points: # every point is being processed
                self._timestep += 1 # increase early so that precision increase can catch up to correct step
                
                self._gap_detection_loops()
                # amount of points might have changed -> reload
                points = self._points
                prev_points = self._prev_points
                normals = self._normals
                prev_normals = self._prev_normals
                    
                if self.printing:
                    self.print_func(f"reached: {self._timestep}")
        

    def process(self, to_timestep:int):
        if self.printing:
            self.print_func(f"processing: {self._timestep} -> {to_timestep}")
        if self._timestep is None or self._timestep>to_timestep: self._start(to_timestep)
        else: self._continue(to_timestep)
    
    def can_draw(self): return self._points.size!=0
    
    def get_boundary_lines(self):
        return point_lines(self._points)
    
    def get_inner_normals(self, length=1):
        return self._points, self._points-self._normals*length
    
    def get_outer_normals(self, length=1):
        return self._points, self._points+self._normals*length

    def get_prev_inner_normals(self, length=1):
        return self._prev_points, self._prev_points-self._prev_normals*length
    
    def draw(self, resolution:int):
        if self.printing: self.print_func(f"\ndrawing: {self._timestep}")
        
        #
        draw_prev_points = 0
        draw_prev_normals = 0
        draw_boundary_lines = 1
        draw_outer_normals = 0
        draw_inner_normals = 1
        
        #
        
        drawing = ImageDrawing(r=1, g=1, b=1)

##        drawing.circles([(0,0)], 1, *red)
        drawing.grid((0,0), .25, b=1, a=0.2)
        
        if draw_boundary_lines:
            drawing.lines(*self.get_boundary_lines(), g=1)
        
        if draw_inner_normals:
            drawing.lines(*self.get_inner_normals())
        if draw_outer_normals:
            drawing.lines(*self.get_outer_normals())
        
        if draw_prev_points:
            drawing.points(self._prev_points, b=1)
            if draw_prev_normals:
                drawing.lines(*self.get_prev_inner_normals())
        
        drawing.points(self._points, r=1)

##        l = len(self)//4
        drawing.circles(self._points[:1], self.epsilon/10, inside=self.epsilon/11, r=1)
##        drawing.circles(self._points[l:l+1], self.epsilon/10, inside=-self.epsilon/20, r=0.75, b=0.25)
##        drawing.circles(self._points[l*2:l*2+1], self.epsilon/10, inside=-self.epsilon/20, r=0.5, b=0.5)
##        drawing.circles(self._points[l*3:l*3+1], self.epsilon/10, inside=-self.epsilon/20, r=0.25, b=0.75)
##        drawing.circles(self._points[-1:], self.epsilon/10, inside=-self.epsilon/20, b=1)
        
        image = drawing.draw(resolution)
        return image, drawing.tl, drawing.br




if __name__ == "__main__":
    config = ModelConfiguration()
    config.start_point.x = 0
    config.start_point.y = 0
    config.epsilon = 0.0625
    config.function.set_constants(a=0.6, b=0.3)

    print(config.function)
    
##    config.function.fx.string = "x/2+(1-y)/3*x/4"
##    config.function.fy.string = "y/3+x/3"  # 
    
##    config.function.fx.string = "x/3+cos(y)**2"
##    config.function.fy.string = "y/2+sin(x)**2"
    
##    config.function.fx.string = "x/2-y/3"
##    config.function.fy.string = "y/2+x/5"
    
##    config.function.fx.string = "x/2-y/3"
##    config.function.fy.string = "y/2+x/3"
    
##    config.function.fx.string = "x/2+1/(y**2+1)"
##    config.function.fy.string = "y/(x**2+1)"

    tij = config.function.transposed_inverse_jacobian()
    print(tij[0][0])
    print(tij[0][1])
    print(tij[1][0])
    print(tij[1][1])
    print("")

    
    resolution = 2000
    timestep = 0
    for ax_target in test_plotting_grid(2, 2, timestep):
        config.process(timestep)
        image,tl,br = config.draw(resolution)
        
        ax_target.imshow(image, extent=(tl[0],br[0],tl[1],br[1]))
        
        timestep += 1





