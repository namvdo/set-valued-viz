from _imports import *
SAVE_DIR = os.path.join(WORKDIR, "normals_model_saves")


class ModelConfiguration():
    epsilon = 0.01
    
    def __init__(self):
        self.start_point = Point2D(0,0)
        self.function = MappingFunction2D()
        
    def process(self, radians, timesteps=1):
        tij = self.function.transposed_inverse_jacobian()
        
        normals = radians_to_vectors(radians)*self.epsilon
        prev_normals = normals.copy()
        
        points = np.repeat([(self.start_point.x,self.start_point.y)], radians.size, axis=0).astype(np.float64)
        points[:,0], points[:,1] = self.function(points[:,0], points[:,1])
        points += normals
        
        for i in range(timesteps):
            prev_normals[:] = normals
            new_normals = np.zeros_like(normals)
            
            new_normals[:,0] += tij[0][0].solve(x=points[:,0], y=points[:,1], **self.function.constants)
            new_normals[:,0] += tij[1][0].solve(x=points[:,0], y=points[:,1], **self.function.constants)
            normals[:,0] *= new_normals[:,0]
            
            new_normals[:,1] += tij[0][1].solve(x=points[:,0], y=points[:,1], **self.function.constants)
            new_normals[:,1] += tij[1][1].solve(x=points[:,0], y=points[:,1], **self.function.constants)
            normals[:,1] *= new_normals[:,1]
            
            # unit length
##            print(normals)
            lengths = np.sqrt(normals[:,0]**2+normals[:,1]**2)
            zero_len = lengths==0
            lengths[zero_len] = 1
##            normals[zero_len][:,1] = 1
            
            normals[:,0] /= lengths
            normals[:,1] /= lengths

            # scale to epsilon and add the normals to the points
            points[:,0], points[:,1] = self.function(points[:,0], points[:,1])
            normals *= self.epsilon
            points += normals
        
        return points, normals, prev_normals

    # now every timestep forward use the ever increasingly accurate radian array to create the new boundary
    def draw_timestep_image(self, resolution, timesteps=1):
        radians = np.linspace(0, np.pi*2, int(np.pi*resolution*2))
        points, normals, _ = self.process(radians, timesteps=timesteps)
        
        def pixelize_points(real_points, topleft, bottomright):
            pixelized_points = real_points-topleft
            pixelized_points *= resolution/(bottomright-topleft).max()
            return (pixelized_points+.5).astype(np.int32)
        
        done = False
        max_allowed_pixel_gap = 10
        radian_increase_loops = 10
        loop = 0
        while not done and radian_increase_loops>=loop:
            loop += 1
            # bounding box
            topleft, bottomright = bounding_box(points)
            #
            
            # bring the points to the pixel range
            pixels = points*resolution/(bottomright-topleft).max()
            
            # resulting image might not appear continuous if there are pixel wide differences
            # solution is to expand the radians array in those areas and calculate their values through the stack
            radians_wraparound = np.concatenate([radians[-1:],radians,radians[:1]])
            points_wraparound = np.concatenate([points[-1:],points,points[:1]])
            pixels_wraparound = np.concatenate([pixels[-1:],pixels,pixels[:1]])
            
            diff = np.diff(pixels_wraparound, axis=0)
            pixel_dist = np.sqrt(diff[:,0]**2+diff[:,1]**2)
            gap_mask = pixel_dist>max_allowed_pixel_gap
            
##            if gap_mask.any():
##                print(pixel_dist[gap_mask].min(), pixel_dist[gap_mask].max())
            if gap_mask.any(): #  and radians.size<1e5 # (gap_mask.sum()/gap_mask.size)>.01
                # extend the radians at the gap points
                radians_from = radians_wraparound[:-1][gap_mask]
                radians_to = radians_wraparound[1:][gap_mask]
                
                additional_radians = np.linspace(radians_from, radians_to, (loop+1)*3)[1:-1].flatten()
                radians = np.append(radians, additional_radians)
                print("radians increased to", radians.size)
                
                additional_points, additional_normals, _ = self.process(additional_radians, timesteps=timesteps)
                points = np.append(points, additional_points, axis=0)
                normals = np.append(normals, additional_normals, axis=0)
##                prev_normals = np.append(prev_normals, additional_prev_normals, axis=0)
                
                sorting = radians.argsort()
                radians = radians[sorting]
                points = points[sorting]
                normals = normals[sorting]
##                prev_normals = prev_normals[sorting]
            else:
                done = True
        
        # values are now continuous enough -> ready to draw
        outer_normal_points = points+normals
        inner_normal_points = points-normals
        topleft, bottomright = bounding_box(np.stack([*bounding_box(points),*bounding_box(outer_normal_points),*bounding_box(inner_normal_points)]))

        # bring points to pixel range
        pixels = pixelize_points(points, topleft, bottomright)
        outer_normal_pixels = pixelize_points(outer_normal_points, topleft, bottomright)
        inner_normal_pixels = pixelize_points(inner_normal_points, topleft, bottomright)

        # determine image shape
        max_x = max(outer_normal_pixels[:,0].max(), inner_normal_pixels[:,0].max(), pixels[:,0].max())
        max_y = max(outer_normal_pixels[:,1].max(), inner_normal_pixels[:,1].max(), pixels[:,1].max())
        aspect = max_x/max_y
        shape = (int(resolution*min(aspect, 1))+1, int(resolution*min(1/aspect, 1))+1)
##        shape = (resolution+1, resolution+1)
        image = np.zeros(shape)

##        # failsafe
##        pixels = np.clip(pixels, a_min=0, a_max=np.subtract(image.shape, 1))
##        outer_normal_pixels = np.clip(outer_normal_pixels, a_min=0, a_max=np.subtract(image.shape, 1))
##        inner_normal_pixels = np.clip(inner_normal_pixels, a_min=0, a_max=np.subtract(image.shape, 1))
        
        def draw_line_on_image(start, end, value=1):
            line_mask = mask_line(start, end)
            if line_mask is None: return False
            tl = (min(start[0], end[0]), min(start[1], end[1]))
            tl = np.clip(tl, a_min=0, a_max=np.subtract(image.shape, 1))
            x_slice = slice(tl[0], tl[0]+line_mask.shape[0])
            y_slice = slice(tl[1], tl[1]+line_mask.shape[1])
##            try:
            image[x_slice, y_slice] += line_mask*(image[x_slice,y_slice]<value)*value
##            except ValueError:
##                print("")
##                print(start, end)
##                print(tl)
##                print(image[x_slice, y_slice].shape, line_mask.shape)
##                print(x_slice, y_slice)
##                input()
            return True

        # actual boundary
        image[pixels[:,0],pixels[:,1]] = 10

        # lines between boundary points and the normal lines
        prev = None
        for index,pixel in enumerate(pixels):
            if prev is not None and not draw_line_on_image(prev, pixel, 5): continue
            draw_line_on_image(pixel, inner_normal_pixels[index], value=1)
##            draw_line_on_image(pixel, outer_normal_pixels[index], value=1)
            prev = pixel
        if prev is not None: draw_line_on_image(prev, pixels[0].astype(np.uint16), 5)
        
        return image.astype(np.uint8), topleft, bottomright






if __name__ == "__main__":
    
    config = ModelConfiguration()
    config.start_point.x = 1
    config.start_point.y = 1
    config.epsilon = 0.001
    
##    config.function.x.string = "y/2"
##    config.function.y.string = "x/3"
    
##    config.function.x.string = "x/2+y/3"
##    config.function.y.string = "y/2-x/3"
    
    config.function.x.string = "(x+1)/2-y/3"
    config.function.y.string = "y/2+x/3"
    
##    config.function.x.string = "x/2+1/(y**2+1)"
##    config.function.y.string = "y/(x**2+1)"

##    tij = config.function.transposed_inverse_jacobian()
##    print(tij[0][0])
##    print(tij[0][1])
##    print(tij[1][0])
##    print(tij[1][1])
##    print("")

    resolution = 256*4
    timestep = 0
    while 1:
        fig,ax = plt.subplots(1,4, figsize=(20,5))
        for i in range(len(ax)):
            image,tl,br = config.draw_timestep_image(resolution, timestep)
            print(timestep, f"n:{(image>0).sum()}", br-tl, image.shape)
            
            extent = (tl[0],br[0],-br[1],-tl[1])
            ax[i].imshow(image.swapaxes(0, 1), extent=extent)
            timestep += 1
        
        plt.show()





