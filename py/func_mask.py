import numpy as np
from func_vectors import *

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
    fill_mask_hor = np.zeros(mask.shape, dtype=np.bool_)
    fill_mask_ver = fill_mask_hor.copy()
    
    for r,row in enumerate(mask):
        i = row.argmax() # first one
        while 1:
            j = i+row[i:].argmin() # first zero after that
            ii = j+row[j:].argmax() # closing one
            if j==ii: break
            fill_mask_hor[r,j:ii] = True
            i = ii
            
    for c in range(mask.shape[1]):
        col = mask[:,c]
        i = col.argmax() # first one
        while 1:
            j = i+col[i:].argmin() # first zero after that
            ii = j+col[j:].argmax() # closing one
            if j==ii: break
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
    # resolution == longest side
    
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
        valid = image[x_slice,y_slice]==0
        image[x_slice, y_slice][valid] = mask[valid]
        return True
    
    # draw
    prev = pixels[-1]
    for index,pixel in enumerate(pixels):
        draw_line_on_image(prev, pixel)
        prev = pixel
    
    if fill: image |= fill_closed_areas(image)
    return image

def center_rotate_points(points, rotation):
    topleft, bottomright = bounding_box(points)
    center = (bottomright+topleft)/2
    offsets = points-center
    temp = offsets[:,0]
    offsets[:,0] = temp*np.cos(rotation)-offsets[:,1]*np.sin(rotation)
    offsets[:,1] = temp*np.sin(rotation)+offsets[:,1]*np.cos(rotation)
    return center+offsets


if __name__ == "__main__":
    mask = np.ones(10, dtype=np.bool_)
    mask[8] = False
##    mask[4] = True
##    mask[5] = True
    print(mask)
    new_mask = repeat_mask_ones_until_divisible(mask, 2)
    print(new_mask)

    print(mask_line((0,0), (-1,0.1)))
