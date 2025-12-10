from _imports import *


def calc_shape(topleft, bottomright, resolution):
    domain = np.subtract(bottomright, topleft)
    aspect_ratio = np.divide(*domain)
    width = height = resolution
    if aspect_ratio>1: height *= 1/aspect_ratio
    else: width *= aspect_ratio
    shape = (int(width), int(height))
    return shape, domain

def points_in_gridlike_shape(topleft, bottomright, resolution):
    shape, domain = calc_shape(topleft, bottomright, resolution)
    grid = np.zeros((*shape, 2))
    x = np.linspace(topleft[0], bottomright[0], shape[0])
    y = np.linspace(topleft[1], bottomright[1], shape[1])
    y, x = np.meshgrid(y, x)
    grid[:,:,0] = x
    grid[:,:,1] = y
    return grid.reshape(-1, 2)

def probability_array(points, topleft, bottomright, resolution): # resolution ~ density
    points = np.subtract(points, topleft)
    shape, domain = calc_shape(topleft, bottomright, resolution)
    probs = np.zeros(shape)
    size = np.divide(domain, probs.shape)
    
    points = np.divide(points, size)
    indexes = np.clip(points, a_min=0, a_max=np.subtract(probs.shape, 1))
    for i,j in indexes.astype(np.int32): probs[i,j] += 1
    probs /= probs.sum()
    return probs
    

if __name__ == "__main__":
    f = MappingFunction2D("1-a*x*x+y", "b*x")
    f.set_constants(a=0.6, b=0.3)
    
    tl = (-1,-1)
    br = (2,1)
    resolution = 100
    
    points = points_in_gridlike_shape(tl,br, resolution/3)
    
    while 1:
        probs = probability_array(points, tl, br, resolution)
        probs = np.flip(probs.swapaxes(0, 1), axis=0)
        
        plt.imshow(probs, extent=(tl[0], br[0], tl[1], br[1]))
        plt.show()
        points[:,0], points[:,1] = f(points[:,0], points[:,1])
    pass
