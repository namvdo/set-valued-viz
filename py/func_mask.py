import numpy as np
from func_vectors import *

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
    border = mask.copy()
    border[1:,:] = mask[1:]*~mask[:-1]
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

##@function_timer
def hausdorff_distance(source_mask, target_mask):
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
##        print(source_points.shape[0])
        rejects = source_points[:,0]<(source_points[:,0].max()-1)
        rejects *= source_points[:,1]<(source_points[:,1].max()-1)
        rejects *= source_points[:,0]>(source_points[:,0].min()+1)
        rejects *= source_points[:,1]>(source_points[:,1].min()+1)
        source_points = source_points[~rejects]
##        print("->", source_points.shape[0])
    
    max_distance = 0
    for p in border_overlap_points:
        i = nearestpoint_i(p, *source_points)
        max_distance = max(max_distance, distance(p, source_points[i]))
    return max_distance




def mask_line(start, end):
    offset = np.subtract(np.array(end, dtype=np.float64), np.array(start, dtype=np.float64))
    dist = np.linalg.norm(offset)
    if dist<1: return
    points = np.linspace((0,0), offset, int(dist+((dist%1)>0)))
    if points[:,0].min()<0: points[:,0] = abs(points[::-1,0])
    if points[:,1].min()<0: points[:,1] = abs(points[::-1,1])
    points += +.5
    points = points.astype(np.int32)
    offset = abs(np.subtract(points[0], points[-1]))
    mask = np.zeros((int(offset[0]+1),int(offset[1]+1)), dtype=np.bool_)
##    print(mask.shape, offset)
    mask[points[:,0],points[:,1]] = True
##    print(offset, mask.shape)
    return mask



if __name__ == "__main__":
    mask = np.ones(10, dtype=np.bool_)
    mask[8] = False
##    mask[4] = True
##    mask[5] = True
    print(mask)
    new_mask = repeat_mask_ones_until_divisible(mask, 2)
    print(new_mask)

    print(mask_line((0,0), (-1,0.1)))
