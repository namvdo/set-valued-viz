from _imports import *

# from arrayprc
def find_intersection(line0, line1):
    def det(a, b): return a[0]*b[1]-a[1]*b[0]
    xdiff = (line0[0][0]-line0[1][0], line1[0][0]-line1[1][0])
    ydiff = (line0[0][1]-line0[1][1], line1[0][1]-line1[1][1])
    if (div:=det(xdiff, ydiff))!=0:
        d = (det(line0[0], line0[1]), det(line1[0],line1[1]))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

##def find_intersections(line, lines, return_valid=False):
##    def det(a, b): return a[0]*b[1]-a[1]*b[0]
##    lines = np.asarray(lines)
##    xdiff0 = line[0][0]-line[1][0]
##    xdiff1 = lines[:,0,0]-lines[:,1,0]
##    ydiff0 = line[0][1]-line[1][1]
##    ydiff1 = lines[:,0,1]-lines[:,1,1]
##    div = xdiff0*ydiff1-xdiff1*ydiff0
##    valid = div!=0
##    d0 = det(*line)
##    d1 = lines[valid][:,0,0]*lines[valid][:,1,1]-lines[valid][:,0,1]*lines[valid][:,1,0]
##    x = d0*xdiff1-d1*xdiff0
##    y = d0*ydiff1-d1*ydiff0
##    x = x[valid]/div[valid]
##    y = y[valid]/div[valid]
##    result = np.concatenate([np.expand_dims(x, axis=1), np.expand_dims(y, axis=1)], axis=1)
##    if return_valid: return result, valid
##    return result
#

def make_polygon_lines(points):
    lines = np.repeat(np.expand_dims(points, axis=0), 2, axis=0)
    lines[1,1:] = lines[0,:-1]
    lines[1,0] = lines[0,-1]
    return lines

def even_sided_polygon(sides, rotation):
    poly_rad = np.linspace(0, np.pi*2, sides+1)[:-1]
    poly_rad += rotation
    return radians_to_vectors(poly_rad)

def apply_noise_geometry(normals, polygon):
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
        if result_dist is not None:
            normals[i] *= result_dist


if __name__ == "__main__":
    resolution = 500
    radians = np.linspace(0, np.pi*2, resolution)[:-1]
    normals = radians_to_vectors(radians)
    
    polygon = even_sided_polygon(4, 0)
##    polygon = radians_to_vectors(np.array([0,0.5,1.])*np.pi+1)
    old_normals = normals.copy()
    apply_noise_geometry(normals, polygon)

    drawing = ImageDrawing(*np.ones(3))
    
    drawing.lines(*make_polygon_lines(polygon), *[1.,0.,0.])
##    drawing.lines(*make_polygon_lines(normals))
    drawing.points(old_normals)
    drawing.points(normals, *[0.,0.,1.])

    image = drawing.draw(resolution)
    image = np.flip(image.swapaxes(0, 1), axis=0)
    
    plt.imshow(image)
    plt.show()
