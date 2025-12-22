from _imports import *
from _quick_visuals import *

BG_COLOR = np.array([1,1,1])
OBJ_COLOR = np.array([0,0,0])

class ModelConfiguration(ModelBase):
    timestep = None
    drawing_prev = None
    drawing = None
    
    def _firststep(self):
        self.timestep = -1
        self.drawing = ImageDrawing(*BG_COLOR)
        self.drawing.points([self.start_point.as_tuple()], *OBJ_COLOR)
        
    def get_boundary_pixels(self, resolution:int):
        if self.drawing is None: self._firststep()
        image = self.drawing.draw(resolution, False)
        mask = image_color_mask(image, OBJ_COLOR*255)
        border = get_mask_border(mask)
        indexes = calc_mask_indexes(mask)
        return indexes[border].astype(np.float64)
    
    def hausdorff_distance(self, resolution:int):
        if self.drawing_prev is not None:
            # make images the same shape + measure their offset
            tl, br = self.drawing.tl, self.drawing.br
            tl_prev, br_prev = self.drawing_prev.tl, self.drawing_prev.br
            
            # make the images same shapes
            self.drawing.update_tl_br(tl_prev, br_prev)
            self.drawing_prev.update_tl_br(tl, br)
            mask = image_color_mask(self.drawing.draw(resolution, False), OBJ_COLOR*255)
            mask_prev = image_color_mask(self.drawing_prev.draw(resolution, False), OBJ_COLOR*255)
            
            # fill closed areas
            mask |= find_closed_areas(mask)
            mask_prev |= find_closed_areas(mask_prev)
            
            # restore the tl & br of the drawings to what they were
            self.drawing.tl = tl
            self.drawing.br = br
            self.drawing_prev.tl = tl_prev
            self.drawing_prev.br = br_prev

            # calc the distance
            dist = hausdorff_distance(mask_prev, mask)
            dist = max(hausdorff_distance(mask, mask_prev), dist)
            dist /= resolution
            return dist

    def process(self, resolution:int):
        if self.drawing is None: self._firststep()
        # get boundary points from the image
        pixels = self.get_boundary_pixels(resolution)
        
        points = pointify_pixels(pixels, self.drawing.tl, self.drawing.br, resolution)
        #
        
        # process the points using the function
        points[:,0], points[:,1] = self.function(points[:,0], points[:,1])
        
        # place epsilon circles on a new drawing
        self.drawing_prev = self.drawing
        self.drawing = ImageDrawing(*self.drawing.background)
        self.drawing.circles(points, self.epsilon, *OBJ_COLOR)
        
        self.timestep += 1

##    @function_timer
    def draw(self, resolution:int):
        if self.drawing is None: self._firststep()
        image = self.drawing.draw(resolution)
        mask = image_color_mask(image, OBJ_COLOR*255)
        image[find_closed_areas(mask),:3] = OBJ_COLOR*255
        return image, self.drawing.tl, self.drawing.br

    def draw_border(self, resolution:int):
        if self.drawing is None: self._firststep()
        image = self.drawing.draw(resolution)
        mask = image_color_mask(image, OBJ_COLOR*255)
        mask |= find_closed_areas(mask)
        border = get_mask_border(mask)
        image[~border] = self.drawing.background*255
        return image, self.drawing.tl, self.drawing.br


if __name__ == "__main__":
    config = ModelConfiguration()
    config.start_point.x = 0
    config.start_point.y = 0
    config.epsilon = 0.0625
    config.function.set_constants(a=0.6, b=0.3)
    
    resolution = 300
##    for _ in range(10): config.process(resolution)
    while 1:
        for ax_target in plotting_grid(2, 2):
            config.process(resolution)
            image, tl, br = config.draw(resolution)
            
            ax_target.set_title(f"step: {config.timestep}")
            ax_target.imshow(image, extent=(tl[0],br[0],tl[1],br[1]))
            
            print(config.timestep, config.hausdorff_distance(resolution))

        plt.show()







