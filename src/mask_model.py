
import matplotlib.pyplot as plt
import numpy as np


# from arrayprc

##def around(a):
##    signs = np.sign(a)
##    i = np.int_(a)
##    a *= signs
##    a %= 1
##    a *= 2*signs
##    return i+np.int_(a)

##def distance(a, b):
##    return np.linalg.norm(np.subtract(a, b), axis=-1) # n-dimensional point to point distance

def circle_mask(r, inner=0, border=0):
    x = np.expand_dims(np.arange(-r, r+1), axis=1)
    y = np.expand_dims(np.arange(-r, r+1), axis=0)
    x *= x
    y *= y
    dist_from_center = np.sqrt(x + y)
    mask = dist_from_center<=r
    if inner>0: mask *= dist_from_center>inner
    elif inner<0: mask *= dist_from_center>r+inner
    return np.pad(mask, (border, border))
#




def get_mask_borders(mask):
    borders = mask.copy()
    borders[1:,:] = mask[1:]*~mask[:-1]
    borders[:,1:] |= mask[:,1:]*~mask[:,:-1]
    borders[:-1,:] |= ~mask[1:,]*mask[:-1]
    borders[:,:-1] |= ~mask[:,1:]*mask[:,:-1]
    return borders

def calc_mask_indexes(mask):
    indexes = np.expand_dims(np.arange(mask.shape[0]), axis=1)
    indexes = np.expand_dims(indexes, axis=2)
    indexes = np.repeat(indexes, mask.shape[1], axis=1)
    indexes = np.repeat(indexes, 2, axis=2)
    indexes[:,:,1] = np.arange(mask.shape[1])
    return indexes


def fill_closed_areas(mask): # very crude
    fill_mask_hor = np.zeros(mask.shape, dtype=np.bool_).copy()
    fill_mask_ver = fill_mask_hor.copy()
    
    for r,row in enumerate(mask):
        i = row.argmax() # first one
        while 1:
            j = i+row[i:].argmin() # first zero after that
            ii = j+row[j:].argmax() # closing one
            if j==ii:
                break
            fill_mask_hor[r,j:ii] = True
            i = ii
            
    for c in range(mask.shape[1]):
        col = mask[:,c]
        i = col.argmax() # first one
        while 1:
            j = i+col[i:].argmin() # first zero after that
            ii = j+col[j:].argmax() # closing one
            if j==ii:
                break
            fill_mask_ver[j:ii,c] = True
            i = ii
    
    return fill_mask_hor*fill_mask_ver




def henon_mapping(x, y, a=1.4, b=0.3):
    return (1-a*x*x+y), b*x
    
def henon_test(start=(0,0), steps=10000):
    results = np.zeros((steps, 2))
    for i in range(1, steps):
        results[i] = henon_mapping(results[i-1, 0], results[i-1, 1])
        
    plt.scatter(results[:,0], results[:,1], s=1, c=np.linspace(0, 1, results.shape[0]))
    plt.show()
##
##
##def D1_model(x, map_x, eps_min=-1, eps_max=1, steps=1000, seed=0):
##    rng = np.random.default_rng(seed)
##    
##    results = np.zeros(steps)
##    results[0] = x
##    randomness = np.zeros(results.shape)
##    
##    eps = (min(eps_max, eps_min), max(eps_max, eps_min))
##    eps_range = eps[1]-eps[0]
##
##    for i in range(1, steps):
##        r = rng.random()*eps_range + eps[0]
##        randomness[i] = r
##        results[i] = map_x(results[i-1])
##        results[i] += r
##
##    plt.scatter(np.arange(results.size), results, s=1)
##    plt.plot(np.arange(results.size), randomness, c=(0,0,1,.5))
##    plt.show()
##
##def D2_model_simple(start_point, mapping_function, epsilon=1, steps=1000, resolution=10):
##    results = np.zeros((steps, resolution, 2))
##    results[0,:] = start_point
##
##    resolution_radians = np.linspace(0, np.pi*2, resolution+1)[:-1]
##    eps_offset = np.zeros((resolution, 2))
##    eps_offset[:,0] = np.sin(resolution_radians)
##    eps_offset[:,1] = np.cos(resolution_radians)
##    
##    for i in range(1, steps):
##        for j in range(resolution):
##            results[i, j] = mapping_function(results[i-1, j])
##        results[i] += eps_offset * epsilon
##
##    new_results = results.reshape(-1,2)
##
##    colors = np.random.random((results.shape[0], 3)) * .8
##    colors = np.repeat(colors, resolution, axis=0)
##    
##    plt.scatter(new_results[:,0], new_results[:,1], s=1, c=colors)
##    plt.show()
##    pass


def D2_model_mask(start_point, mapping_function, epsilon=1, border=1, autofill=False, square_aspect=False):
    eps_border = epsilon+border
    eps_circle = circle_mask(epsilon, border=border)

    # place the starting point on the mask
    mask = np.zeros((3,3), dtype=np.bool_)
    mask[1,1] = True
    topleft = np.subtract(start_point, np.divide(mask.shape, 2))
    bottomright = np.add(mask.shape, topleft)
    #

    # start modeling
    while 1:
        yield topleft, mask
        
        borders = get_mask_borders(mask)
        indexes = calc_mask_indexes(mask)

        points_to_process = list(indexes[borders].astype(np.float64)+topleft)
        points_processed = []
        for x,y in points_to_process:
            points_processed.append(mapping_function(x,y))

            topleft = np.minimum(topleft, np.subtract(points_processed[-1], eps_border))
            bottomright = np.maximum(bottomright, np.add(points_processed[-1], eps_border))
        
        new_mask_shape = np.subtract(bottomright, topleft).astype(np.int64)
            
        new_mask = np.zeros(new_mask_shape+1, dtype=np.bool_)
        for x,y in points_processed:
            x = int(x-topleft[0])
            y = int(y-topleft[1])
            x = max(x-eps_border, 0)
            y = max(y-eps_border, 0)
            x_slice = slice(x, x+eps_circle.shape[0])
            y_slice = slice(y, y+eps_circle.shape[1])
            new_mask[x_slice, y_slice] |= eps_circle
        mask = new_mask

        if autofill:
            mask |= fill_closed_areas(mask)



if __name__ == "__main__":
##    henon_test()

    ### PARAMETERS
    # zoom level; -> mapping precision; array indexes have a length of 1/ZOOM units
    ZOOM = 200
    def mapping_wrapper(one_of_the_mapping_functions):
        def wrapper(x,y,*args,**kwargs):
            x /= ZOOM
            y /= ZOOM
            x,y = one_of_the_mapping_functions(x,y,*args,**kwargs)
            x *= ZOOM
            y *= ZOOM
            return x, y
        return wrapper

    # starting point for the first epsilon radius circle; can heavily affect the end shape
    START_POINT = (1,0)
    START_POINT = np.multiply(START_POINT, ZOOM) # make sure the start point is uniform/consistent at every zoom level

    # radius of the created circles; 0== single point; 1==plus-shaped circle; 2+ == actual circles
    EPSILON = 50

    # Alternative visuals
    # True == cumulative addition of masks
    # False == highest value in the image is the most recent timestep
    ALT_VISUALS = False
    
    # crudely fill closed off areas inside each single mask; helps with calculation speed (less edges)
    AUTOFILL = True
    ###
    
    # different functions
    @mapping_wrapper
    def mapping_henon(x, y, a=1.4, b=0.3):
        return (1-a*x*x+y), b*x
    
    @mapping_wrapper
    def mapping_func0(x, y):
        return x/2, y/2
    
    @mapping_wrapper
    def mapping_func1(x, y):
        return x * np.sin(np.pi*y*1.3), y * np.sin(np.pi*y)
    
    @mapping_wrapper
    def mapping_func2(x, y):
        return (x*2 + x*np.sin(y**2))/3, (y + x*2*np.cos(x**2))/3
    
    @mapping_wrapper
    def mapping_func3(x, y):
        return (y*np.cos(y+x**2)), x*np.sin(y**2+x)
    
    # choose a function to use
    FUNCTION = mapping_func2
    #


    ##############
    image = np.zeros((1,1), dtype=np.uint64)
    
    prev_tl = None
    i = 1
    # generate a mask with topleft position per timestep
    for tl,mask in D2_model_mask(START_POINT, FUNCTION, EPSILON, autofill=AUTOFILL):
        if prev_tl is None: prev_tl = tl

        # expand the total image array
        offset = (prev_tl-tl)
        pad_r = np.subtract(mask.shape, image.shape)/2
        offset = (offset-pad_r).astype(np.int64)
        pad_l = pad_r.copy()
        pad_r += pad_r%1>=.5
        pad_r = np.clip(pad_r, a_min=0, a_max=None).astype(np.int64)
        pad_l = np.clip(pad_l, a_min=0, a_max=None).astype(np.int64)
        pad = ((pad_l[0]+offset[0],pad_r[0]-offset[0]), (pad_l[1]+offset[1],pad_r[1]-offset[1]))
        image = np.pad(image, pad)
        #

        # add the mask to create a full image
        if ALT_VISUALS: image[mask] = mask[mask]*i
        else: image += mask
        #
        
        # matplotlib visualization
        print(f"step:{i}", f"scale:{1/ZOOM:.6f}", f"topleft:({tl[0]:.2f}, {tl[1]:.2f}) shape:{mask.shape}")
        fig,ax = plt.subplots(1,2)
        ax[0].imshow(image)
        ax[1].imshow(mask)
        plt.show()
        #

        prev_tl = tl
        i += 1
    ##############
        
    pass











