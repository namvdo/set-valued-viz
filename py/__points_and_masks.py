import numpy as np
from __containers import *

# from arrayprc
def distance(a, b):
    return np.linalg.norm(np.subtract(a, b), axis=-1) # n-dimensional point to point distance

def nearestpoint_i(point, points):
    return np.argmin(np.linalg.norm(points-point, axis=1))

def farthestpoint_i(point, points):
    return np.argmax(np.linalg.norm(points-point, axis=1))

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

def bounding_box_corners(points):
    tl, br = np.min(points, axis=0), np.max(points, axis=0)
    asd = np.concatenate([tl,br[::-1]], axis=0)
    bl = asd[::2]
    tr = asd[::-2]
    return tl, tr, br, bl

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




# 0100000100 -> 0111111100
# 0110010000 -> 0110011111
def switch_mask_to_output_mask(toggle_mask):
    continuous = np.zeros_like(toggle_mask)
    value = True
    i = j = 0
    while j<toggle_mask.size:
        i = toggle_mask[j:].argmax() # next one
        ii = toggle_mask[j:].argmin()
        if i==ii: break
        j += i
        continuous[j:] = value
        value = not value
        j += 1
    return continuous

# every true value, turn next values true so that length of consecutive trues is divisible by N
# example: 0110010000
#   divisible by 2 -> 0110011000
#   divisible by 3 -> 0111011100
def repeat_mask_ones_until_divisible(mask, divisible_by=2):
    if mask.all() or not mask.any(): return mask
    continuous = np.zeros_like(mask)
    i = j = 0
    while j<mask.size:
        if not mask[j:].any():
            continuous[j:] = False # the rest are false
            break
        i = mask[j:].argmax() # next one
        j += i
        if mask[j:].all():
            continuous[j:] = True # the rest are true
            break
        ones = mask[j:].argmin()
        if ones%divisible_by: ones += divisible_by-(ones%divisible_by)
        continuous[j:j+ones] = True
        j += ones
    return continuous

def get_mask_border(mask):
    border = np.zeros_like(mask)
    border[0,:] = mask[0,:]
    border[:,0] = mask[:,0]
    border[-1,:] = mask[-1,:]
    border[:,-1] = mask[:,-1]
    border[1:,:] |= mask[1:]*~mask[:-1]
    border[:,1:] |= mask[:,1:]*~mask[:,:-1]
    border[:-1,:] |= ~mask[1:,]*mask[:-1]
    border[:,:-1] |= ~mask[:,1:]*mask[:,:-1]
    return border

def calc_mask_indexes(mask):
    indexes = np.expand_dims(np.arange(mask.shape[0]), axis=1)
    indexes = np.expand_dims(indexes, axis=2)
    indexes = np.repeat(indexes, mask.shape[1], axis=1)
    indexes = np.repeat(indexes, 2, axis=2)
    indexes[:,:,1] = np.arange(mask.shape[1])
    return indexes


def simple_hausdorff_distance(source_mask, target_mask):
    # flood source_mask until target_mask is covered
    # source_mask and target_mask must be same shapes
    steps = 0
    flood_mask = np.zeros_like(source_mask)
    while not ((source_mask|target_mask)==source_mask).all():
        flood_mask[:,1:] |= (source_mask[:,1:]!=source_mask[:,:-1])
        flood_mask[:,:-1] |= flood_mask[:,1:]
        source_mask |= flood_mask
        if ((source_mask|target_mask)==source_mask).all():
            return steps+.5
        
        flood_mask[1:,:] |= (source_mask[1:,:]!=source_mask[:-1,:])
        flood_mask[:-1,:] |= flood_mask[1:,:]
        source_mask |= flood_mask
##        print(source_mask.astype(np.int8), end="\n\n")
        steps += 1
    return steps

def hausdorff_distance(source_mask, target_mask): # one way only
    source_floodable = source_mask.copy()
    steps = simple_hausdorff_distance(source_floodable, target_mask)
    if steps==0: return 0 # fully inside
    
    border = get_mask_border(source_floodable)
    border_overlap = border*target_mask
    
    indexes = calc_mask_indexes(source_mask)
    border_overlap_points = indexes[border_overlap]
    source_points = indexes[source_mask]
    
    if source_points.shape[0]>1:
        # reject unimportant points
        rejects = source_points[:,0]<(source_points[:,0].max()-1)
        rejects *= source_points[:,1]<(source_points[:,1].max()-1)
        rejects *= source_points[:,0]>(source_points[:,0].min()+1)
        rejects *= source_points[:,1]>(source_points[:,1].min()+1)
        source_points = source_points[~rejects]
    
    max_distance = 0
    for p in border_overlap_points:
        i = nearestpoint_i(p, source_points)
        max_distance = max(max_distance, distance(p, source_points[i]))
    return max_distance
    

def overlap_box(tl1, br1, tl2, br2): # n-dimensional
    tl = np.max([tl1,tl2], axis=0)
    br = np.min([br1,br2], axis=0)
    if (tl<br).all(): return tl, br

def points_inside_box(points, tl, br): # n-dimensional
    mask = np.ones(points.shape[0], dtype=np.bool_)
    for i in range(points.shape[1]):
        mask *= (points[:,i]>tl[i])*(points[:,i]<br[i])
    return mask



def find_hausdorff_pair(points1, points2): # n-dimensional
    index1 = index2 = 0
    temp_dist = 0
    for i,p1 in enumerate(points1):
        dists = np.linalg.norm(p1-points2, axis=1)
        j = np.argmin(dists)
        if dists[j]>temp_dist:
            temp_dist = dists[j]
            index1 = i
            index2 = j
    return index1, index2



def hausdorff_distance3(points1, points2, float_precision=10000):
    # both ways
    # VERY FAST
    overlap = overlap_box(*bounding_box(points1), *bounding_box(points2))
    if overlap is not None: # there is overlap -> slower method
        # find the nearest point per point in the other object
        temp_dist = 0
        for pp in points1:
            dists = np.linalg.norm(pp-points2, axis=1)
            i = np.argmin(dists)
            p = points2[i]
            if dists[i]>temp_dist:
                temp_dist = dists[i]
                farthest_point1 = pp
                nearest_point2 = p
        temp_dist = 0
        for pp in points2:
            dists = np.linalg.norm(pp-points1, axis=1)
            i = np.argmin(dists)
            p = points1[i]
            if dists[i]>temp_dist:
                temp_dist = dists[i]
                farthest_point2 = pp
                nearest_point1 = p
        
    else: # no overlap -> fast method
        # inaccurate when there is significant overlap between objects
        mp1 = MassPointsFloat(float_precision=float_precision)
        mp2 = MassPointsFloat(float_precision=float_precision)
        for i,p in enumerate(points1): mp1.set(i, p)
        for i,p in enumerate(points2): mp2.set(i, p)

        # find object center
        nearest_point1 = np.add(*bounding_box(points1))/2
        nearest_point2 = np.add(*bounding_box(points2))/2
        
        # find the farthest point from the other nearest point
        farthest1 = mp1.farthest(nearest_point2)
        farthest2 = mp2.farthest(nearest_point1)
        farthest_point1 = points1[list(farthest1)[0]]
        farthest_point2 = points2[list(farthest2)[0]]
        
        # find the nearest point in the other object from the farthest point
        nearest1 = mp1.nearest(farthest_point2)
        nearest2 = mp2.nearest(farthest_point1)
        nearest_point1 = points1[list(nearest1)[0]]
        nearest_point2 = points2[list(nearest2)[0]]
    
    dist1 = distance(nearest_point1, farthest_point2)
    dist2 = distance(nearest_point2, farthest_point1)
    if dist1>dist2: return dist1, (nearest_point1, farthest_point2)
    return dist2, (nearest_point2, farthest_point1)



def hausdorff_distance4(points1, points2): # presumes that points are in order, EVEN FASTER
    def quick_find_pair(ps1, ps2):
        step = max(int(np.sqrt(len(ps1))), 1)
        index1, index2 = find_hausdorff_pair(ps1[::step], ps2)
        if step==1: return ps1[index1], ps2[index2]
        start = max(index1-1, 0)*step
        end = (index1+1)*step+1
        index3, index2 = find_hausdorff_pair(ps1[start:end], ps2)
        index1 = start+index3
        return ps1[index1], ps2[index2]
    
    line1 = quick_find_pair(points1, points2)
    line2 = quick_find_pair(points2, points1)
    
    dist1 = distance(*line1)
    dist2 = distance(*line2)
    if dist1>dist2: return dist1, line1
    return dist2, line2


def hausdorff_distance5(points1, points2): # NOT QUITE AS FAST AS 4, but indifferent to point order; n-dimensional
    distance_matrix = np.zeros((len(points1), len(points2)))
    
    index1 = index2 = 0
    temp_dist = 0
    for i,p1 in enumerate(points1): # go through first points and build the distance matrix
        distance_matrix[i] = np.linalg.norm(p1-points2, axis=1)
        j = np.argmin(distance_matrix[i])
        if distance_matrix[i,j]>temp_dist:
            temp_dist = distance_matrix[i,j]
            index1 = i
            index2 = j
    
    line1 = points1[index1], points2[index2]
    
    temp_dist = 0
    for i,j in enumerate(np.argmin(distance_matrix, axis=0)): # go through the distance matrix columns
        if distance_matrix[j,i]>temp_dist:
            temp_dist = distance_matrix[j,i]
            index1 = i
            index2 = j
    
    line2 = points2[index1], points1[index2]
    
    dist1 = distance(*line1)
    dist2 = distance(*line2)
    if dist1>dist2: return dist1, line1
    return dist2, line2


    
def divide_to_sectors(points): # n-dimensional
    # yield batches of points based on what dimensional corner direction is the closest
    ndim = points.shape[1]
    tl, br = bounding_box(points)
    mask = np.zeros(ndim, dtype=np.bool_)
    
    distance_matrix = np.zeros((ndim*2,len(points)))
    index = 0
    while 1:
        corner = tl.copy()
        corner[mask] = br[mask]
        
        distance_matrix[index] = np.linalg.norm(points-corner, axis=1)
        index += 1
        
        if mask.all(): break
        
        j = np.argmin(mask)
        mask[j] = True
        mask[:j] = False
    
    sectors = np.argmin(distance_matrix, axis=0)
    for i in range(ndim*2):
        yield points[sectors==i]

def hausdorff_distance6(points1, points2): # SUPER FAST, but inaccurate; n-dimensional
    
    def get_sectors_and_centers(points):
        sectors = [ps for ps in divide_to_sectors(points) if ps.size>0]
        centers = [np.mean(ps, axis=0) for ps in sectors]
        return sectors, centers
    
    nearest_points1 = points1
    nearest_points2 = points2
    farthest_points1 = points1
    farthest_points2 = points2
    
    while 1:
        near_sectors1, near_centers1 = get_sectors_and_centers(nearest_points1)
        near_sectors2, near_centers2 = get_sectors_and_centers(nearest_points2)
        far_sectors1, far_centers1 = get_sectors_and_centers(farthest_points1)
        far_sectors2, far_centers2 = get_sectors_and_centers(farthest_points2)
        if len(far_sectors2)<2 and len(far_sectors1)<2 and len(near_sectors1)<2 and len(near_sectors2)<2: break
        
        far1,near2 = find_hausdorff_pair(far_centers1, near_centers2)
        far2,near1 = find_hausdorff_pair(far_centers2, near_centers1)
        
        nearest_points1 = near_sectors1[near1]
        nearest_points2 = near_sectors2[near2]
        farthest_points1 = far_sectors1[far1]
        farthest_points2 = far_sectors2[far2]
    
    # finish
    far1,near2 = find_hausdorff_pair(farthest_points1, nearest_points2)
    line1 = farthest_points1[far1], nearest_points2[near2]
    
    far2,near1 = find_hausdorff_pair(farthest_points2, nearest_points1)
    line2 = farthest_points2[far2], nearest_points1[near1]
    
    dist1 = distance(*line1)
    dist2 = distance(*line2)
    if dist1>dist2: return dist1, line1
    return dist2, line2

def hausdorff_distance7(points1, points2): # BASELINE, checks every point against every point
    far1,near2 = find_hausdorff_pair(points1, points2)
    line1 = points1[far1], points2[near2]
    
    far2,near1 = find_hausdorff_pair(points2, points1)
    line2 = points2[far2], points1[near1]
    
    dist1 = distance(*line1)
    dist2 = distance(*line2)
    if dist1>dist2: return dist1, line1
    return dist2, line2



def hausdorff_distance8(points1, points2): # SUPER FAST and accurate; n-dimensional
    # same as baseline when no overlap at all -> TAKE SHORTCUT
    # twice as fast to baseline when one object is fully overlapping another
    # extremely fast when objects are very close to each other
    box1 = bounding_box(points1)
    box2 = bounding_box(points2)
    
    overlap1 = points_inside_box(points1, *box2)
    overlap2 = points_inside_box(points2, *box1)
    
    if not overlap1.any() and not overlap2.any(): # no overlap at all
        # SHORTCUT
        center1 = np.mean(points1, axis=0)
        near2 = np.argmin(np.linalg.norm(points2-center1, axis=1)) # find point nearest to center1
        far1 = np.argmax(np.linalg.norm(points1-points2[near2], axis=1))
        line1 = points1[far1], points2[near2]
        dist1 = distance(*line1)
        
        center2 = np.mean(points2, axis=0)
        near1 = np.argmin(np.linalg.norm(points1-center2, axis=1)) # find point nearest to center2
        far2 = np.argmax(np.linalg.norm(points2-points1[near1], axis=1))
        line2 = points2[far2], points1[near1]
        dist2 = distance(*line2)
        
    else: # some overlap
        farthest1 = points1[~overlap1] # ignore points that are inside the other shape
        farthest2 = points2[~overlap2] # ignore points that are inside the other shape

        # worst case:
        #   both objects have very little overlap with each other -> near baseline
        # best case:
        #   objects overlap eachother almost fully -> near instant
        
        if farthest1.size>0:
            far1,near2 = find_hausdorff_pair(farthest1, points2)
            line1 = farthest1[far1], points2[near2]
            dist1 = distance(*line1)
        else:
            line1 = points1[0], points1[0]
            dist1 = 0

        if farthest2.size>0:
            far2,near1 = find_hausdorff_pair(farthest2, points1)
            line2 = farthest2[far2], points1[near1]
            dist2 = distance(*line2)
        else:
            line2 = line1
            dist2 = dist1
    
    if dist1>dist2: return dist1, line1
    return dist2, line2


##def find_hausdorff_pair_backandforth(points1, points2): # n-dimensional; inaccurate if there is overlap
##    far_point = np.mean(points1, axis=0)
##    far = near = None
##    while 1:
##        near = np.argmin(np.linalg.norm(points2-far_point, axis=1))
##        i = np.argmax(np.linalg.norm(points1-points2[near], axis=1))
##        if far==i: break # no change -> found the best pair
##        far = i
##    return far, near



def image_shape(resolution, max_x, max_y, *args):
    # resolution -> length of the longest side
    aspect = max_x/max_y if max_y!=0 else 1
    width = int(resolution*min(aspect, 1))
    height = int(resolution*min(1/aspect, 1))
    if width>height: height += 1
    elif width<height: width += 1
    return width, height

def pixelize_points(points, topleft, bottomright, resolution):
    pixels = points-topleft
    scale = (bottomright-topleft).max()
    if scale!=0: pixels /= scale
    else: pixels *= 0
    pixels *= resolution-1
    pixels += .5
    return pixels.astype(np.int32)

def pixelize_distances(values, topleft, bottomright, resolution):
    pixels = np.divide(values, (bottomright-topleft).max())
    pixels *= resolution-1
    return pixels.astype(np.int32)

def pointify_pixels(pixels, topleft, bottomright, resolution):
    points = pixels.astype(np.float64)
    points -= .5
    points /= resolution-1
    points *= (bottomright-topleft).max()
    return points+topleft


def circle_mask(r, inner=0):
    x = np.expand_dims(np.arange(-r, r+1), axis=1)
    y = np.expand_dims(np.arange(-r, r+1), axis=0)
    x *= x
    y *= y
    dist_from_center = np.sqrt(x + y)
    mask = dist_from_center<=r
    if inner>0: mask *= dist_from_center>inner
    elif inner<0: mask *= dist_from_center>r+inner
    return mask

def line_mask(start, end): # mask_line
    offset = np.subtract(end, start).astype(np.float64)
    dist = np.linalg.norm(offset)
    if dist<1: return
    _from = (0 if offset[0]>0 else -offset[0],0 if offset[1]>0 else -offset[1])
    _end = (0 if offset[0]<0 else offset[0],0 if offset[1]<0 else offset[1])
    points = np.linspace(_from, _end, int(dist+((dist%1)>0))).astype(np.int32)
    mask = np.zeros(abs(offset).astype(np.int32)+1, dtype=np.bool_)
    mask[points[:,0],points[:,1]] = True
    return mask

def polygon_mask(points, resolution:int, fill=False):
    # calculate the bounding box
    topleft, bottomright = bounding_box(points)
    # translate points to pixels
    pixels = pixelize_points(points, topleft, bottomright, resolution)
    
    width, height = image_shape(resolution, *np.max(pixels, axis=0))
    image = np.zeros((width, height), dtype=np.bool_)
    
    masks = {}
    def draw_line_on_image(start, end):
        key = (int(start[0]-end[0]), int(start[1]-end[1]))
        alt_key = (-key[0], -key[1])
        if key in masks: mask = masks[key]
        elif alt_key in masks: mask = masks[alt_key]
        else:
            mask = line_mask(start, end)
            if mask is not None:
                masks[key] = mask
        if mask is None: return False
        
        tl = np.min([start,end], axis=0)
        x_slice = slice(tl[0], tl[0]+mask.shape[0])
        y_slice = slice(tl[1], tl[1]+mask.shape[1])
        image[x_slice, y_slice] |= mask
        return True
    
    # draw
    prev = pixels[-1]
    for index,pixel in enumerate(pixels):
        draw_line_on_image(prev, pixel)
        prev = pixel
    
    if fill: image |= horizontally_closed_areas(image)
    return image


def center_rotate_points(points, rotation):
    topleft, bottomright = bounding_box(points)
    center = (bottomright+topleft)/2
    offsets = points-center
    temp = offsets[:,0]
    offsets[:,0] = temp*np.cos(rotation)-offsets[:,1]*np.sin(rotation)
    offsets[:,1] = temp*np.sin(rotation)+offsets[:,1]*np.cos(rotation)
    return center+offsets

def image_color_mask(image, color):
    return (image[:,:,0]==color[0])*(image[:,:,1]==color[1])*(image[:,:,2]==color[2])


##def fill_closed_areas(mask): # very crude
##    fill_mask_hor = np.zeros(mask.shape, dtype=np.bool_)
##    fill_mask_ver = fill_mask_hor.copy()
##    
##    for r,row in enumerate(mask):
##        i = row.argmax() # first one
##        while 1:
##            j = i+row[i:].argmin() # first zero after that
##            ii = j+row[j:].argmax() # closing one
##            if j==ii: break
##            fill_mask_hor[r,j:ii] = True
##            i = ii
##            
##    for c in range(mask.shape[1]):
##        col = mask[:,c]
##        i = col.argmax() # first one
##        while 1:
##            j = i+col[i:].argmin() # first zero after that
##            ii = j+col[j:].argmax() # closing one
##            if j==ii: break
##            fill_mask_ver[j:ii,c] = True
##            i = ii
##    return fill_mask_hor*fill_mask_ver


def horizontally_closed_areas(mask):
    fill_mask_hor = np.zeros_like(mask)
    for r,row in enumerate(mask):
        i = row.argmax() # first one
        while 1:
            j = i+row[i:].argmin() # first zero after that
            ii = j+row[j:].argmax() # closing one
            if j==ii: break
            fill_mask_hor[r,j:ii] = True
            i = ii
    return fill_mask_hor

def vertically_closed_areas(mask):
    fill_mask_ver = np.zeros_like(mask)
    for c in range(mask.shape[1]):
        col = mask[:,c]
        i = col.argmax() # first one
        while 1:
            j = i+col[i:].argmin() # first zero after that
            ii = j+col[j:].argmax() # closing one
            if j==ii: break
            fill_mask_ver[j:ii,c] = True
            i = ii
    return fill_mask_ver

def find_closed_areas(mask): # upgrade to fill_closed_areas
    surrounded = horizontally_closed_areas(mask)
    surrounded *= vertically_closed_areas(mask)
    
    while 1:
        combined = mask|surrounded
        open_upward = surrounded[:-1,:]*~(combined[1:,:])
        if open_upward.any():
            surrounded[:-1,:][open_upward] = False
            continue
        open_downward = surrounded[1:,:]*~(combined[:-1,:])
        if open_downward.any():
            surrounded[1:,:][open_downward] = False
            continue
        open_leftward = surrounded[:,1:]*~(combined[:,:-1])
        if open_leftward.any():
            surrounded[:,1:][open_leftward] = False
            continue
        open_rightward = surrounded[:,:-1]*~(combined[:,1:])
        if open_rightward.any():
            surrounded[:,:-1][open_rightward] = False
            continue
        break
    return surrounded


def lines_intersect(line1, line2, boolean=True):
    linebox1 = bounding_box(line1)
    linebox2 = bounding_box(line2)
    # bounding boxes must have overlap
    overlap = overlap_box(*linebox1, *linebox2)
    if overlap is not None:
        # lines must have an intersection
        intersection = find_intersection(line1, line2)
        if intersection is not None:
            # intersection must be in the overlap
            if points_inside_box(np.expand_dims(intersection, axis=0), *overlap).any():
                if boolean: return True
                return intersection
    if boolean: return False

def iterate_intersections(points):
    l = len(points)
    starts, ends = point_lines(points)
    done = set()
    for i in range(l):
        line1 = starts[i], ends[i]
        for j in range(l):
            if (i==0 and j==(l-1)) or (j==0 and i==(l-1)): continue
            if abs(j-i)<2: continue # lines must be atleast 1 apart
            k1 = (i,j)
            if (j,i) in done: continue
            line2 = starts[j], ends[j]
            intersection = lines_intersect(line1, line2, False)
            if intersection is not None:
                yield intersection, (i-1, i), (j-1, j) # intersection + lines as point indexes
                done.add(k1)

def sort_points_to_polygon(points): # crude and inaccurate
    l = len(points)
    unused = np.ones(l, dtype=np.bool_)
    indexes = np.arange(l)
    sorting = indexes.copy()

    def get_nearest(p):
        dists = np.linalg.norm(points[unused]-p, axis=1)
        i = np.argmin(dists)
        return indexes[unused][i], dists[i]
    
    count_right = count_left = 0
    index_right = index_left = np.argmin(np.linalg.norm(np.diff(points, axis=0), axis=1))
    while 1:
        sorting[count_right] = index_right
        unused[index_right] = False
        sorting[count_left] = index_left
        unused[index_left] = False
        
        if not unused.any(): break
        index1, dist1 = get_nearest(points[index_right])
        index2, dist2 = get_nearest(points[index_left])
        if dist1<dist2:
            index_right = index1
            count_right += 1
        else:
            index_left = index2
            count_left -= 1
    points[:] = points[sorting]



def sort_points_to_polygon3(points): # slow and dumb, but "works"
    l = len(points)
    indexes = np.arange(l)
    
    distances = np.zeros((l,l))
    for i in range(l):
        p = points[i]
        distances[i] = np.linalg.norm(points-p, axis=1)
    
    def single_run(start):
        sorting = indexes.copy()
        valid = np.ones(l, dtype=np.bool_)
        best_index = start
        step = 0
        sorting[step] = best_index
        valid[best_index] = False
        while valid.any():
            step += 1
            order = distances[best_index].argsort()
            next_index = order[valid[order]][0]
            best_index = next_index
            sorting[step] = best_index
            valid[best_index] = False
        return sorting
    
    sorting = single_run(0)
    while 1: # clean up intersections
        points[:] = points[sorting]
        sorting[:] = indexes
        intersections = False
        for x,ii,jj in iterate_intersections(points):
            intersections = True
            sorting[ii[1]] = jj[0]
            sorting[jj[0]] = ii[1]
            break
        if not intersections: break



def small_triangles_from_points(points):
    l = len(points)
    indexes = np.arange(l)
    valid = np.ones(l, dtype=np.bool_)

    def smallest_triangle(index):
        dists = distance(points, points[index])
        dists[index] = dists.max()+1
        
        i = np.argmin(dists)
        dists[i] = dists.max()
        i = indexes[i]
        
        j = np.argmin(dists)
        j = indexes[j]
        
        return i, j

    visited = set()
    for i in range(l):
        if i not in visited:
            j, k = smallest_triangle(i)
            visited |= {i,j,k}
            yield np.array([points[i],points[j],points[k]])
        


def rotate_vectors(vectors, z=0):
    rotated = np.zeros((len(vectors), 2))
    cos_z = np.cos(z)
    sin_z = np.sin(z)
    rotated[:,0] = vectors[:,0]*cos_z-vectors[:,1]*sin_z
    rotated[:,1] = vectors[:,0]*sin_z+vectors[:,1]*cos_z
    return rotated

def rotate_vectors_3d(vectors, x=0, y=0, z=0):
    rotated = np.ones((len(vectors), 3))
    vx = vectors[:,0]
    vy = vectors[:,1]
    vz = vectors[:,2]
    
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    
    cos_y = np.cos(y)
    sin_y = np.sin(y)
    
    cos_z = np.cos(z)
    sin_z = np.sin(z)
    
    rotated[:,0] = vx*cos_y*cos_z + vy*(sin_x*sin_y*cos_z-cos_x*sin_z) + vz*(cos_x*sin_y*cos_z + sin_x*sin_z)
    rotated[:,1] = vx*cos_y*sin_z + vy*(sin_x*sin_y*sin_z + cos_x*cos_z) + vz*(cos_x*sin_y*sin_z - sin_x*cos_z)
    rotated[:,2] = -vx*sin_y + vy*sin_x*cos_y + vz*cos_x*cos_y
    return rotated





def points_in_gridlike_shape(topleft, bottomright, shape):
    grid = np.zeros((*shape, 2))
    x = np.linspace(topleft[0], bottomright[0], int(shape[0]+1))[:-1]
    y = np.linspace(topleft[1], bottomright[1], int(shape[1]+1))[:-1]
    x += (x[1]-x[0])/2
    y += (y[1]-y[0])/2
    y, x = np.meshgrid(y, x)
    grid[:,:,0] = x
    grid[:,:,1] = y
    return grid


class PeriodicPointDetector():
    threshold = 0.1 # atleast this close to be considered the same point
    
    _index = 0
    _last_period = None
    _instability = 0
    
    def __init__(self, size):
        self.size = size
        self.buffer = np.zeros((size,2))

    def __len__(self): return self.size

    def reset(self):
        self._index = 0
        self.buffer = np.zeros((self.size,2))
        self._last_period = None
        self._instability = 0

    def resize(self, size):
        self.size = size
        if size<len(self.buffer): # new size is smaller -> reset
            self.reset()
        else: # new buffer is bigger -> expand buffer
            newbuffer = np.zeros((self.size,2))
            newbuffer[:len(self.buffer)] = self.buffer
            self.buffer = newbuffer
    
    def _check_period(self, point):
        for i in range(-1, -self.size, -1):
            dist = np.linalg.norm(point-self.buffer[(i+self._index)%self.size])
            if dist<=self.threshold: # consider as the same point
                return -i # period distance

    def is_stable(self): # has been current period for atleast period amount of steps
        return self._last_period is not None and self._instability==0
        
    def log(self, point):
        period = self._check_period(point)
        
        if period is not None:
            if period == self._last_period:
                if self._instability>0: self._instability -= 1
            else: self._instability = period # reset instability
        self._last_period = period
        
        self.buffer[self._index] = point
        self._index += 1
        if self._index==self.size: self._index %= self.size
        return period








if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    points = np.random.random((10,3))
    points[:len(points)//2] += 1
    
##    points2 = points.copy()
##    sort_points_to_polygon(points2)
    
##    points3 = points.copy()
##    sort_points_to_polygon3(points3)
    
    plt.plot(points[:,0], points[:,1], color="#F0F")
    center = np.mean(points, axis=0)

    x = np.pi/4
    y = 0#np.pi/4
    z = 0#np.pi/4
    points1 = rotate_vectors_3d(points-center, x, y, z)[:,:2]
    points1 += center[:2]
##    points1[:,1] += 1
    plt.plot(points1[:,0], points1[:,1], color="#F00")
        
    points1 = rotate_vectors_3d(points-center, x*2, y*2, z*2)[:,:2]
    points1 += center[:2]
##    points1[:,1] -= 1
    plt.plot(points1[:,0], points1[:,1], color="#00F") # *point_lines(points1)
    
    plt.show()

    
##    # test mask filling
##    mask = circle_mask(20)
##    mask *= np.random.random(mask.shape)>.5
##    
##    mask1 = mask.astype(np.uint8)+vertically_closed_areas(mask)/2
##    mask2 = mask.astype(np.uint8)+horizontally_closed_areas(mask)/2
##    mask3 = mask.astype(np.uint8)+find_closed_areas(mask)/2
##    
##    fig, ax = plt.subplots(1, 4, figsize=(12,5))
##    ax[0].imshow(mask)
##    ax[1].imshow(mask1)
##    ax[2].imshow(mask2)
##    ax[3].imshow(mask3)
##    plt.show()
    pass

