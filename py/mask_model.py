from _imports import *

class ModelConfiguration():
    alt_visuals = False
    autofill = False
    
    epsilon = 0.01
    precision = 0.005
    border_width = 1
    escape_distance = 10. # distance at which to consider them as escaped (prevents extreme array sizes)
    
    timestep = 0
    image_history = None
    image = None
    
    topleft = (0,0)
    bottomright = (0,0)
    topleft_old = (0,0)
    
    escaped_points = 0
    escaped_points_total = 0
    processed_points = 0
    processed_points_total = 0
    hausdorff_distance = 0
    
    def log_processed_points(self, n):
        self.processed_points = n
        self.processed_points_total += n
        
    def log_escaped_points(self, n):
        self.escaped_points = n
        self.escaped_points_total += n

    def log_hausdorff(self, dist):
        self.hausdorff_distance = dist

    def __init__(self):
        self.start_point = Point2D(0,0)
        self.function = MappingFunction2D()
        
        self.padding_error = np.zeros(2)
        
        # place the starting point on the mask
        self.image = circle_mask(0, border=self.border_width)
        
        tl = -np.divide(self.image.shape, 2).astype(np.int32).astype(np.float64)
        br = np.add(self.image.shape, tl)
        self.topleft = tl*self.precision
        self.bottomright = br*self.precision
        self.topleft = np.add(self.topleft, (self.start_point.x,self.start_point.y))
        self.bottomright = np.add(self.bottomright, (self.start_point.x,self.start_point.y))
        #
        
        self.image_history = self.image.astype(np.uint16)
        self.topleft_old = self.topleft.copy()
        self.timestep = 0

    def epsilon_circle(self):
        return circle_mask(int(self.epsilon/self.precision), border=self.border_width)

    def calc_current_border_points(self):
        border = get_mask_border(self.image)
        indexes = calc_mask_indexes(self.image)
        return indexes[border].astype(np.float64)

    def points_as_units(self, points):
        points -= self.topleft
        points /= self.precision

    def points_as_values(self, points):
        points *= self.precision
        points += self.topleft
    
    def visualization(self):
        extent = (self.topleft[0],self.bottomright[0],self.topleft[1],self.bottomright[1])
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(self.image_history.swapaxes(0, 1)[::-1,:], extent=extent)
        ax[1].imshow(self.image.swapaxes(0, 1)[::-1,:], extent=extent)
        ax[0].set_title(f"Cumulative Timesteps")
        ax[1].set_title(f"Timestep: {self.timestep}")
        plt.show()


    def nextstep(self):
        self.topleft_old[:] = self.topleft

        # create the circle mask & get border points from the image
        eps_circle = self.epsilon_circle()
        points = self.calc_current_border_points()
        #
        
        # translate indexes to their corresponding points
        self.points_as_values(points)

        # process the points using the function
        n = points.shape[0]
        points[:,0], points[:,1] = self.function(points[:,0], points[:,1])
        self.log_processed_points(n)
        
        # check for escapees and delete them
        escaped_points = np.zeros(points.shape[0], dtype=np.bool_)
        escaped_points |= np.abs(points[:,0]-self.start_point.x)>=self.escape_distance
        escaped_points |= np.abs(points[:,1]-self.start_point.y)>=self.escape_distance
        if escaped_points.any():
            points = points[~escaped_points]
            n = escaped_points.sum()
            self.log_escaped_points(n)
            if points.size==0: return
        #
        
        # measure the new topleft and bottomright
        eps_r = eps_circle.shape[0]/2 # space needed around a point for the epsilon circle
        eps_r *= self.precision
        temp_points = np.subtract(points, eps_r)
        self.topleft[0] = min(self.topleft[0], temp_points[:,0].min())
        self.topleft[1] = min(self.topleft[1], temp_points[:,1].min())
        temp_points += eps_r*2
        self.bottomright[0] = max(self.bottomright[0], temp_points[:,0].max())
        self.bottomright[1] = max(self.bottomright[1], temp_points[:,1].max())


        # translate the points to positive integer format
        self.points_as_units(points)
        
        # calculate the new image shape
        image_domain = np.subtract(self.bottomright, self.topleft)
        image_shape = image_domain/self.precision
        image_shape = np.maximum(image_shape.astype(np.int32)+((image_shape%1)!=0), self.image_history.shape)

        # create the new image array
        new_image = np.zeros(image_shape, dtype=np.bool_)
        
        # paint the new image with circles
        eps_r = eps_circle.shape[0]//2
        for x,y in points:
            x = int(x)
            y = int(y)
            x_slice = slice(x-eps_r, x+eps_r+1)
            y_slice = slice(y-eps_r, y+eps_r+1)
            new_image[x_slice, y_slice] |= eps_circle
        
        if self.autofill:
            new_image |= fill_closed_areas(new_image)
        #########
        
        #### PADDING CORRECTION
        # expand the cumulative image array
        pad_l = -(self.topleft-self.topleft_old) / self.precision

        # collect the error from padding misplacement
        # also shift the topleft and bottomright accordingly
        pad_error = pad_l%1
        self.padding_error += pad_error
        self.topleft += pad_error*self.precision
        self.bottomright += pad_error*self.precision

        # consume full integers of the padding error to adjust the image back to place (towards negative)
        # also shift the topleft and bottomright back in to place
        pad_correction = self.padding_error.astype(np.int8)
        if pad_correction.any():
            self.padding_error -= pad_correction
            self.topleft -= pad_correction*self.precision
            self.bottomright -= pad_correction*self.precision
##            print("PAD_CORRECTION:", pad_correction)
        ####
        
        # do the padding
        shape_diff = np.subtract(image_shape, self.image_history.shape)
        pad_r = shape_diff-pad_l.astype(np.int32)
        pad_l += pad_correction
        pad = ((int(pad_l[0]),int(pad_r[0])), (int(pad_l[1]),int(pad_r[1])))
        
        self.image_history = np.pad(self.image_history, pad)
        self.image = np.pad(self.image, pad) # needed for hausdorff
        if pad_correction.any():
            new_image = np.pad(new_image, ((pad_correction[0],0),(pad_correction[1],0)))
        #
        
        # hausdorff
        dist = hausdorff_distance(new_image, self.image)
        dist = max(dist, hausdorff_distance(self.image, new_image))
        dist *= self.precision
        self.log_hausdorff(dist)
        #

        # previous image is overwritten
        self.image = new_image
        #
        
        # add the mask to create a full image
        if self.alt_visuals:
            self.image_history[self.image] = self.image[self.image]*(self.timestep+1)
        else:
            self.image_history += self.image
        #
        
        self.timestep += 1


if __name__ == "__main__":
    pass
    









