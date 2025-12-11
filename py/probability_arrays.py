from _imports import *


def calc_shape(topleft, bottomright, resolution):
    domain = np.subtract(bottomright, topleft)
    aspect_ratio = np.divide(*domain)
    width = height = resolution
    if aspect_ratio>1: height *= 1/aspect_ratio
    else: width *= aspect_ratio
    shape = (int(width), int(height))
    return shape, domain

def points_in_gridlike_shape(topleft, bottomright, shape):
    grid = np.zeros((*shape, 2))
    x = np.linspace(topleft[0], bottomright[0], shape[0]+1)[:-1]
    y = np.linspace(topleft[1], bottomright[1], shape[1]+1)[:-1]
    x += (x[1]-x[0])/2
    y += (y[1]-y[0])/2
    y, x = np.meshgrid(y, x)
    grid[:,:,0] = x
    grid[:,:,1] = y
    return grid

##def probability_array(points, topleft, bottomright, resolution): # resolution ~ density
##    points = np.subtract(points, topleft)
##    shape, domain = calc_shape(topleft, bottomright, resolution)
##    probs = np.zeros(shape)
##    size = np.divide(domain, probs.shape)
##    
##    points = np.divide(points, size)
##    indexes = np.clip(points, a_min=0, a_max=np.subtract(probs.shape, 1))
##    for i,j in indexes.astype(np.int32): probs[i,j] += 1
##    probs /= probs.sum()
##    return probs

def epsilon_circle_points(point, epsilon, amount=4):
    return point+radians_to_vectors(np.linspace(0, np.pi*2, amount))*epsilon

if __name__ == "__main__":
    f = MappingFunction2D("1-a*x*x+y", "b*x")
    f.set_constants(a=0.6, b=0.3)
    eps = 0.06
    
    extent = (-3, 5, -3, 5)
    resolution = 500
    
    points = points_in_gridlike_shape(extent[::2], extent[1::2], (8,8))
    points = points.reshape(-1, 2)
    
    drawing = ImageDrawing(r=1,g=1,b=1)
    while 1:
        drawing.clear()
        drawing.circles(points, eps, a=0.5)
        drawing.points(points, r=1)
        drawing.update_tl_br(extent[::2], extent[1::2])
        image = drawing.draw(resolution)
        
        plt.imshow(image, extent=drawing.get_extent())
        plt.show()
        points[:,0], points[:,1] = f(points[:,0], points[:,1])
    pass
