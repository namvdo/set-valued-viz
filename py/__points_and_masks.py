import numpy as np
from __masspointscontainer import *

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
    

def overlap_box(tl1, br1, tl2, br2):
    tl = np.max([tl1,tl2], axis=0)
    br = np.min([br1,br2], axis=0)
    if (tl<br).all(): return tl, br

def points_inside_box(points, tl, br):
    mask = (points[:,0]>tl[0])*(points[:,0]<br[0])
    mask *= (points[:,1]>tl[1])*(points[:,1]<br[1])
    return mask

def array_slices(array, size):
    i = 0
    while 1:
        x = array[size*i:size*(i+1)]
        if len(x)==0: break
        yield x
        i += 1

def hausdorff_distance3(points1, points2, float_precision=10000):
    # both ways
    # VERY FAST
    overlap = overlap_box(*bounding_box(points1), *bounding_box(points2))
    if overlap is not None: # there is overlap -> slower method
        # find the nearest point per point in the other object
        temp_dist = 0
        for pp in points1:
            p = points2[nearestpoint_i(pp, points2)]
            dist = np.linalg.norm(p-pp)
            if dist>temp_dist:
                temp_dist = dist
                farthest_point1 = pp
                nearest_point2 = p
        temp_dist = 0
        for pp in points2:
            p = points1[nearestpoint_i(pp, points1)]
            dist = np.linalg.norm(p-pp)
            if dist>temp_dist:
                temp_dist = dist
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


##def hausdorff_distance4(points1, points2):
##    nearest_point1 = nearest_point2 = None
##    farthest_point1 = farthest_point2 = None
##    
##    points1_10th = [np.sum(points, axis=0)/10 for points in array_slices(points1, 10)]
##    points2_10th = [np.sum(points, axis=0)/10 for points in array_slices(points2, 10)]
##
##    nearest_points = []
##    for p in points1_10th:
##        nearest_dist = None
##        nearest_pp = None
##        for pp in points2_10th:
##            dist = np.linalg.norm(p-pp)
##            if nearest_dist is None or dist<nearest_dist:
##                nearest_dist = dist
##                nearest_pp = pp
##        nearest_points.append(nearest_pp)
##    
##    dist1 = distance(nearest_point1, farthest_point2)
##    dist2 = distance(nearest_point2, farthest_point1)
##    if dist1>dist2: return dist1, (nearest_point1, farthest_point2)
##    return dist2, (nearest_point2, farthest_point1)
    

def image_shape(resolution, max_x, max_y):
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


if __name__ == "__main__":
    # test mask filling
    mask = circle_mask(20)
    mask *= np.random.random(mask.shape)>.5
    
    mask1 = mask.astype(np.uint8)+vertically_closed_areas(mask)/2
    mask2 = mask.astype(np.uint8)+horizontally_closed_areas(mask)/2
    mask3 = mask.astype(np.uint8)+find_closed_areas(mask)/2
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 4, figsize=(12,5))
    ax[0].imshow(mask)
    ax[1].imshow(mask1)
    ax[2].imshow(mask2)
    ax[3].imshow(mask3)
    plt.show()
    pass

