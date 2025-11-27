import numpy as np

# from arrayprc
def distance(a, b):
    return np.linalg.norm(np.subtract(a, b), axis=-1) # n-dimensional point to point distance

def nearestpoint_i(a, *b):
    return np.argmin([distance(a, x) for x in b])

def nearestvalue_i(array, value):
    return np.argmin(np.abs(np.asarray(array)-value))

def find_intersection(line0, line1):
    def det(a, b): return a[0]*b[1]-a[1]*b[0]
    xdiff = (line0[0][0]-line0[1][0], line1[0][0]-line1[1][0])
    ydiff = (line0[0][1]-line0[1][1], line1[0][1]-line1[1][1])
    if (div:=det(xdiff, ydiff))!=0:
        d = (det(line0[0], line0[1]), det(line1[0],line1[1]))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y
#

def radians_absolute(radians): # -inf...inf -> 0...2*np.pi
    radians = np.mod(radians, np.pi*2)
    radians[radians<0] = radians[radians<0]+np.pi*2
    return radians

def radians_to_vectors(radians):
    vectors = np.repeat(np.expand_dims(radians, axis=1), 2, axis=1)
    vectors[:,0] = np.cos(vectors[:,0])
    vectors[:,1] = np.sin(vectors[:,1])
    return vectors

def vectors_to_radians(vectors):
    return np.arctan2(vectors[:,1], vectors[:,0])

def bounding_box(points):
    return np.min(points, axis=0), np.max(points, axis=0)

def point_normals(points):
    diff = np.diff(points, prepend=points[-1:], axis=0)
    return vectors_to_radians(diff)-np.pi/2

def simple_normals_from_points(points):
    # now outer_points must be expanded outward by the epsilon
    normals = point_normals(points)
    
    candidates0 = radians_to_vectors(normals) # either inside or outside
    candidates1 = radians_to_vectors(normals+np.pi) # either inside or outside
    
    # findout which has the larger bounding box -> the outside
    # and make the outside points the new outer_points
    cand0_topleft, cand0_bottomright = bounding_box(candidates0)
    cand1_topleft, cand1_bottomright = bounding_box(candidates1)
    cand0_area = np.prod(np.subtract(cand0_bottomright, cand0_topleft))
    cand1_area = np.prod(np.subtract(cand1_bottomright, cand1_topleft))
    if cand0_area>cand1_area: return candidates0
    return candidates1


def even_sided_polygon(sides, rotation):
    radians = np.linspace(0, np.pi*2, sides+1)[:-1]
    radians += rotation
    return radians_to_vectors(radians)

def point_lines(points):
    lines = np.repeat(np.expand_dims(points, axis=0), 2, axis=0)
    lines[1,1:] = lines[0,:-1]
    lines[1,0] = lines[0,-1]
    return lines





