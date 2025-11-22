from _imports import *


class ModelConfiguration(ModelBase):
    printing = True
    
    precision_increase_attempts = 10
    precision_decrease_attempts = 10
    
    def __init__(self):
        super().__init__()
        self._reset()

    def _reset(self):
        self.tij = None
        self._timestep = None
        self._radians = np.zeros(0)
        self._points = np.zeros((0,2))
        self._prev_points = np.zeros((0,2))
        self._normals = np.zeros((0,2))
        self._prev_normals = np.zeros((0,2))

        self._radian_deadzones = np.zeros((0,2)) # array of ranges deemed offlimits for precision increase
    
    def _remove_items_with_mask(self, removal_mask):
        self._radians = self._radians[~removal_mask]
        self._points = self._points[~removal_mask]
        self._normals = self._normals[~removal_mask]
        self._prev_points = self._prev_points[~removal_mask]
        self._prev_normals = self._prev_normals[~removal_mask]
        
    def _start(self, to_timestep:int):
        self._timestep = 0
        self._radians = np.linspace(0, np.pi*2, 128)[:-1] # first index is same as the last
        self._points, self._prev_points, self._normals, self._prev_normals = self._process_firststep(self._radians, to_timestep=to_timestep)

    def _continue(self, to_timestep:int):
        catch_up_amount = to_timestep - self._timestep
        if catch_up_amount>0:
            self._process(self._points, self._prev_points, self._normals, self._prev_normals, catch_up_amount)

    
    def _too_sparse_mask(self):
        dist = np.linalg.norm(np.diff(self._points, axis=0, append=self._points[:1]), axis=1)
        return dist>self.epsilon/8

    def _too_dense_mask(self):
        dist = np.linalg.norm(np.diff(self._points, axis=0, append=self._points[:1]), axis=1)
        mask = dist<self.epsilon/16
        mask = repeat_mask_ones_until_divisible(mask, 2)
        mask[1::2] = False
        return mask
    
    def _points_inside_the_boundary_mask(self):
        mask = np.zeros(self._points.shape[0], dtype=np.bool_)
        for i,x in enumerate(self._points-self._normals):
            mask |= np.linalg.norm(x-self._points, axis=1)<self.epsilon*.95
        return mask
    
    def _increase_precision(self, gap_mask):
        end = self._radians[-1:]+(self._radians[-1]-self._radians[-2])*2
        radians_wraparound = np.concatenate([self._radians, end])
        radians_from = radians_wraparound[:-1][gap_mask]
        radians_to = radians_wraparound[1:][gap_mask]

        # apply deadzones to radian limits -> prevent increasing precision in areas
        valid = np.ones(radians_from.shape[0], dtype=np.bool_)
        for deadzone in self._radian_deadzones:
            from_over_min = radians_from>deadzone[0]
            to_under_max = radians_to<deadzone[1]
            from_under_max = radians_from<deadzone[1]
            to_over_min = radians_to>deadzone[0]
            
            valid[from_over_min*to_under_max] = False # cut off completely
            radians_from[from_under_max*from_over_min] = deadzone[1]
            radians_to[to_under_max*to_over_min] = deadzone[0]
        #
        
        more_radians = (radians_from+radians_to)/2
        more_radians = more_radians[valid]
        
        if self.printing:
            print("increasing precision by", more_radians.size)
        
        more_points, more_prev_points, more_normals, more_prev_normals = self._process_firststep(more_radians, to_timestep=self._timestep)
        self._radians = np.append(self._radians, more_radians)
        self._points = np.append(self._points, more_points, axis=0)
        self._normals = np.append(self._normals, more_normals, axis=0)
        self._prev_points = np.append(self._prev_points, more_prev_points, axis=0)
        self._prev_normals = np.append(self._prev_normals, more_prev_normals, axis=0)
        
        sorting = self._radians.argsort()
        self._radians = self._radians[sorting]
        self._points = self._points[sorting]
        self._normals = self._normals[sorting]
        self._prev_points = self._prev_points[sorting]
        self._prev_normals = self._prev_normals[sorting]


    def _gap_detection_loops(self):
        for i in range(self.precision_increase_attempts):
            nogaps = self._too_dense_mask()
            if nogaps.any(): self._remove_items_with_mask(nogaps)
            gaps = self._too_sparse_mask()
            if gaps.any(): self._increase_precision(gaps)
            elif not nogaps.any(): break
        
##        for i in range(self.precision_decrease_attempts):
##            nogaps = self._too_dense_mask()
##            if nogaps.any(): self._remove_items_with_mask(nogaps)
##            else: break

        inside = self._points_inside_the_boundary_mask()
        if not inside.all() and inside.any():
            # this can and will make gaps in the radian array
            # -> create radian deadzones
            connect = np.diff(inside.astype(np.int8)) # 1 == left point, -1 == right point
            indexes = np.arange(len(inside))
            left_indexes = indexes[:-1][connect==1]
            right_indexes = indexes[1:][connect==-1]
            if len(left_indexes)>len(right_indexes):
                right_indexes = np.append(right_indexes, [len(indexes)-1])
            elif len(left_indexes)<len(right_indexes):
                left_indexes = np.append(left_indexes, [0])
            deadzones = np.stack([self._radians[left_indexes],self._radians[right_indexes]], axis=1)
            
            if self._radian_deadzones.size:
                was_used = np.zeros(deadzones.shape[0], dtype=np.bool_)
                for i,dz in enumerate(deadzones):
                    over_max_limit = dz[1]>=self._radian_deadzones[:,1]
                    under_min_limit = dz[0]<=self._radian_deadzones[:,0]
                    full_replace = over_max_limit*under_min_limit
                    
                    self._radian_deadzones[over_max_limit*under_min_limit] = dz # fully replace
                    
                    extend_lower_bound = dz[1]==self._radian_deadzones[:,0]
                    self._radian_deadzones[extend_lower_bound,0] = dz[0]
                    
                    extend_higher_bound = dz[0]==self._radian_deadzones[:,1]
                    self._radian_deadzones[extend_higher_bound,1] = dz[1]
                    
                    was_used[i] = (extend_lower_bound|extend_higher_bound|full_replace).any()
                    
                deadzones = deadzones[~was_used]
                
            self._radian_deadzones = np.append(self._radian_deadzones, deadzones, axis=0)
            ##
            
            self._remove_items_with_mask(inside)





        
    def _process_firststep(self, radians, to_timestep:int):
        points = np.repeat([(self.start_point.x,self.start_point.y)], radians.size, axis=0).astype(np.float64)
        prev_points = points.copy()
        
        normals = radians_to_vectors(radians)
        normals *= self.epsilon
        prev_normals = normals.copy()
        
        points[:,0], points[:,1] = self.function(points[:,0], points[:,1])
        points += normals
        
        if radians is self._radians:
            self._points = points
            self._prev_points = prev_points
            self._normals = normals
            self._prev_normals = prev_normals
            self._process(points, prev_points, normals, prev_normals, to_timestep)
            return self._points, self._prev_points, self._normals, self._prev_normals
        
        self._process(points, prev_points, normals, prev_normals, to_timestep)
        return points, prev_points, normals, prev_normals


    def _process(self, points, prev_points, normals, prev_normals, amount:int):
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
            normals *= self.epsilon
            
            # add the normals to the new points
            points[:,0], points[:,1] = self.function(x, y)
            
##            self._ensure_correct_normal_directions(normals, prev_normals)
            
            points += normals

            # alternative gap detection
            if points is self._points: # every point is being processed
                self._timestep += 1 # increase early so that precision increase can catch up to correct step
                
                self._gap_detection_loops()
                # array size changed -> reload
                points = self._points
                prev_points = self._prev_points
                normals = self._normals
                prev_normals = self._prev_normals
                    
                if self.printing:
                    print("reached step", self._timestep)

        

##    def _ensure_correct_normal_directions(self, normals, prev_normals):
##        # sudden normal inversions -> swap normal direction
##        if normals is self._normals: # every point is being processed
##            threshold = .5
##            normal_radians = vectors_to_radians(normals)
##            normal_radians_diff = np.diff(normal_radians, axis=0)
##            mask = (normal_radians_diff>np.pi*threshold)*(normal_radians_diff<np.pi*(2-threshold))
##            mask |= (normal_radians_diff<-np.pi*threshold)*(normal_radians_diff>-np.pi*(2-threshold))
##            mask = switch_mask_to_output_mask(mask)
##            normals[1:][mask] *= -1
##        pass

##    def _check_normal_cohesion(self):
##        normal_radians = vectors_to_radians(self._normals)
##        normal_radians_diff = np.diff(normal_radians, axis=0)
##        mask = (normal_radians_diff>np.pi/2)*(normal_radians_diff<np.pi*3/2)
##        mask |= (normal_radians_diff<-np.pi/2)*(normal_radians_diff>-np.pi*3/2)
##        if mask.any():
##            print("\nNORMAL COHESION FAILED", mask.sum())
####            for index in np.arange(mask.size)[mask]:
####                print(index, normal_radians_diff[index])
####                print("prev_point", self._prev_points[:-1][index], "->", self._prev_points[1:][index])
####                print("point", self._points[:-1][index], "->", self._points[1:][index])
####                print("prev_normal", self._prev_normals[:-1][index], "->", self._prev_normals[1:][index])
####                print("normal", self._normals[:-1][index], "->", self._normals[1:][index])
####                print("")
##            print("")
##            return mask

    def process(self, to_timestep:int):
        if self.printing:
            print("\nprocessing step", self._timestep, "to", to_timestep)
        if self._timestep is None or self._timestep>to_timestep: self._start(to_timestep)
        else: self._continue(to_timestep)
    
    def can_draw(self): return self._points.size!=0
    
    def get_boundary_lines(self):
        return self._points[:-1], self._points[1:]
    
    def get_inner_normals(self, length=1):
        return self._points, self._points-self._normals*length
    
    def get_outer_normals(self, length=1):
        return self._points, self._points+self._normals*length

    def get_prev_inner_normals(self, length=1):
        return self._prev_points, self._prev_points-self._prev_normals*length
    
    def draw(self, resolution:int):
        if self.printing:
            print("\ndrawing step", self._timestep, f"({resolution} px)")
            print("points ->", self._radians.shape[0])

        #
        draw_prev_points = 0
        draw_prev_normals = 0
        draw_boundary_lines = 1
        draw_outer_normals = 0
        draw_inner_normals = 1
        
        radian_bar_height = 10
        #
        
        drawing = ImageDrawing()
        
        black = np.array([0.,0.,0.,1.])
        red = np.array([1.,0.,0.,1.])
        green = np.array([0.,1.,0.,1.])
        blue = np.array([0.,0.,1.,1.])

        drawing.circle((0,0), 1, red)
        drawing.grid((0,.31), .25, black)
        
        if draw_boundary_lines:
            drawing.lines(*self.get_boundary_lines(), green)
        
        if draw_inner_normals:
            drawing.lines(*self.get_inner_normals(), black)
        if draw_outer_normals:
            drawing.lines(*self.get_outer_normals(), black)
        
        if draw_prev_points:
            drawing.points(self._prev_points, blue)
            if draw_prev_normals:
                drawing.lines(*self.get_prev_inner_normals(), black)
        
        drawing.points(self._points, red)
        
        image = drawing.draw(resolution)
        return image, drawing.tl, drawing.br
        
if __name__ == "__main__":
    config = ModelConfiguration()
    config.start_point.x = 0
    config.start_point.y = 0
    config.epsilon = 0.0625
    config.function.set_constants(a=0.6, b=0.3)
    
##    config.function.fx.string = "x/2+(1-y)/3*x/4"
##    config.function.fy.string = "y/3+x/3"  # 
    
    config.function.fx.string = "x/3+cos(y)**2"
    config.function.fy.string = "y/2+sin(x)**2"
    
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

    
    resolution = 500
    timestep = 0
    for ax_target in test_plotting_grid(2, 2, timestep):
        config.process(timestep)
        image,tl,br = config.draw(resolution)
        
        image = image.swapaxes(0, 1)[::-1,:]
        ax_target.imshow(image, extent=(tl[0],br[0],tl[1],br[1]))
        
        timestep += 1





