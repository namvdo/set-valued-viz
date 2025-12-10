from _imports import *


BG_COLOR = np.ones(4)
OBJ_COLOR = np.zeros(3)

def obj_mask(image):
    return (image[:,:,0]==OBJ_COLOR[0])*(image[:,:,1]==OBJ_COLOR[1])*(image[:,:,2]==OBJ_COLOR[2])

class ModelConfiguration(ModelBase):
    timestep = None
    drawing_prev = None
    drawing = None
    
    def _firststep(self):
        self.timestep = 0
        self.drawing = ImageDrawing(*BG_COLOR)
        self.drawing.points([self.start_point.as_tuple()], *OBJ_COLOR)
        
    def get_boundary_pixels(self, resolution:int):
        if self.drawing is None: self._firststep()
        image = self.drawing.draw(resolution, False)
        mask = obj_mask(image)
        border = get_mask_border(mask)
        indexes = calc_mask_indexes(mask)
        return indexes[border].astype(np.float64)

    def hausdorff_distance(self, resolution:int): # WIP
        if self.drawing_prev is not None:
            # make images the same shape + measure their offset
            mask = obj_mask(self.drawing.draw(resolution))
            mask_prev = obj_mask(self.drawing_prev.draw(resolution))
            rect_now = self.drawing.tl, self.drawing.br
            rect_old = self.drawing_prev.tl, self.drawing_prev.br
            # WIP
            dist = hausdorff_distance(mask_prev, mask)
            dist = max(hausdorff_distance(mask, mask_prev), dist)
            return dist

    def process(self, resolution:int):
        if self.drawing is None: self._firststep()
        # get boundary points from the image
        pixels = self.get_boundary_pixels(resolution)
        
        points = pointify_pixels(pixels, self.drawing.tl, self.drawing.br, resolution)
        #
        
        # process the points using the function
        points[:,0], points[:,1] = self.function(points[:,0], points[:,1])
        #
        
        # place epsilon circles on a new drawing
        self.drawing_prev = self.drawing
        self.drawing = ImageDrawing(*self.drawing.background)
        
        self.drawing.circles(points, self.epsilon, *OBJ_COLOR)
        #
        
        self.timestep += 1

    def draw(self, resolution:int):
        if self.drawing is None: self._firststep()
        return self.drawing.draw(resolution), self.drawing.tl, self.drawing.br


if __name__ == "__main__":
    config = ModelConfiguration()
    config.start_point.x = 0
    config.start_point.y = 0
    config.epsilon = 0.0625
    config.function.set_constants(a=0.6, b=0.3)
    
    resolution = 200
    timestep = 0
    for _ in range(timestep): config.process(resolution)
    for ax_target in test_plotting_grid(2, 2, timestep):
        config.process(resolution)
        image, tl, br = config.draw(resolution)
        ax_target.imshow(image, extent=(tl[0],br[0],tl[1],br[1]))
        








