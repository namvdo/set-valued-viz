import math
import numpy as np


class MassPoints():
    # calculate points in range at a massive scale
    # integer points only
    top = 0
    left = 0
    bottom = 0
    right = 0
    def __init__(self, scaling=2):
        self.scaling = scaling
        self.chunks = [{}] # chunks0 -> keys, chunks1 -> chunks0, chunks2 -> chunks1
        self.points = {} # key -> point

    def distance(self, x1, y1, x2, y2):
        return math.sqrt(math.pow(x1-x2, 2)+math.pow(y1-y2, 2))

    def __len__(self): return len(self.points)

    def shape(self):
        return (self.right-self.left, self.bottom-self.top)

    def set(self, key, point):
        if key in self.points:
            self.delete(key)
        self.points[key] = point
        
        self.top = min(point[1], self.top)
        self.left = min(point[0], self.left)
        self.bottom = max(point[1], self.bottom)
        self.right = max(point[0], self.right)

        # lowest chunk
        if point in self.chunks[0]: self.chunks[0][point].add(key)
        else: self.chunks[0][point] = {key}

        # higher chunks if they exist
        lower_xy = point
        for c in self.chunks[1:]:
            higher_xy = (lower_xy[0]>>self.scaling, lower_xy[1]>>self.scaling)
            if higher_xy in c: c[higher_xy].add(lower_xy)
            else: c[higher_xy] = {lower_xy}
            lower_xy = higher_xy
            
        # check if higher chunks should exist
        if len(self.chunks[-1])>len(self.chunks):
            new_chunk = {}
            for lower_xy in self.chunks[-1]:
                higher_xy = (lower_xy[0]>>self.scaling, lower_xy[1]>>self.scaling)
                if higher_xy in new_chunk: new_chunk[higher_xy].add(lower_xy)
                else: new_chunk[higher_xy] = {lower_xy}
            self.chunks.append(new_chunk)

    def delete(self, key):
        point = self.points[key]
        self.chunks[0][point].remove(key)
        del self.points[key]
        
    def get(self, key):
        return self.points.get(key)

    def distance_between(self, key1, key2):
        return self.distance(*self.points.get(key1), *self.points.get(key2))

    def get_keys(self, point):
        return self.chunks[0].get(point)
    
##    @timepck.function_timer
    def inrange(self, point, radius, as_points=False):
        prev_keys = set(self.chunks[-1].keys())
        for i in range(len(self.chunks)-1, -1, -1):
            keys = set()
            scale = self.scaling*i
            origin = (point[0]>>scale, point[1]>>scale)
            r = radius>>scale
            if scale: r += 1
            
            for target in prev_keys:
                dist = self.distance(*target,*origin)
                if dist<(r+1):
                    if len(self.chunks[i][target]):
                        if i==0 and as_points: keys.add(target)
                        else: keys |= self.chunks[i][target]
            prev_keys = keys
        return keys

##    @timepck.function_timer
    def inrange2(self, point, radius, as_points=False): # faster over large distances
        prev_bypass = set()
        prev_keys = set(self.chunks[-1].keys())
        for i in range(len(self.chunks)-1, -1, -1):
            keys = set()
            bypass = set()
            scale = self.scaling*i
            origin = (point[0]>>scale, point[1]>>scale)
            r = radius>>scale
            if scale: r += 1
            
            for target in prev_bypass:
                if i==0 and as_points:
                    bypass.add(target)
                else:
                    bypass |= self.chunks[i][target]
            
            for target in prev_keys:
                dist = self.distance(*target,*origin)
                if i>0 and dist<max(r-2, 0):
                    # all sub-points here must be in range
                    bypass |= self.chunks[i][target]
                elif dist<(r+1):
                    if len(self.chunks[i][target]):
                        if i==0 and as_points: keys.add(target)
                        else: keys |= self.chunks[i][target]
                        
            prev_bypass = bypass
            prev_keys = keys
        return keys|bypass

##    @timepck.function_timer
    def nearest(self, point, as_points=False, _invert=False):
        prev_keys = set(self.chunks[-1].keys())
        for i in range(len(self.chunks)-1, -1, -1):
            keys = set()
            scale = self.scaling*i
            origin = (point[0]>>scale, point[1]>>scale)
            
            seeked_targets = seeked_dist = None
            for target in prev_keys:
                dist = self.distance(*target,*origin)
                dist = int(dist) # must be an integer (higher chunks introduce error to distances)
                if seeked_targets is None:
                    seeked_targets = [target]
                    seeked_dist = dist
                elif (not _invert and dist<seeked_dist) or (_invert and dist>seeked_dist):
                    seeked_targets = [target]
                    seeked_dist = dist
                elif dist==seeked_dist:
                    seeked_targets.append(target)
            
            if seeked_targets is not None:
                for seeked_target in seeked_targets:
                    if i==0 and as_points: keys.add(seeked_target)
                    else: keys |= self.chunks[i][seeked_target]
            prev_keys = keys
        return keys
    
##    @timepck.function_timer
    def farthest(self, *args, **kwargs):
        return self.nearest(*args, _invert=True, **kwargs)


class MassPointsFloat(MassPoints):
    float_precision = 1000
    def __init__(self, *args, float_precision=1000, **kwargs):
        self.float_precision = float_precision
        super().__init__(*args, **kwargs)

    def float_point_as_int_point(self, x:float, y:float):
        return int(x*self.float_precision), int(y*self.float_precision)
    def int_point_as_float_point(self, x:int, y:int):
        return x/self.float_precision, y/self.float_precision

    def set(self, key, point):
        point = self.float_point_as_int_point(*point)
        point = self.int_point_as_float_point(*point)
        point = self.float_point_as_int_point(*point)
        super().set(key, point)
    
    def get(self, key):
        point = self.points.get(key)
        return self.int_point_as_float_point(*point)
    
    def get_keys(self, point):
        point = self.float_point_as_int_point(*point)
        return self.chunks[0].get(point)
    
    def inrange(self, point, radius, as_points=False):
        point = self.float_point_as_int_point(*point)
        radius *= self.float_precision
        prev_keys = set(self.chunks[-1].keys())
        for i in range(len(self.chunks)-1, -1, -1):
            keys = set()
            scale = self.scaling*i

            origin = (point[0]>>scale, point[1]>>scale)
            r = radius>>scale
            if scale: r += 1
            
            for target in prev_keys:
                dist = self.distance(*target,*origin)
                if dist<(r+1):
                    if len(self.chunks[i][target]):
                        if i==0 and as_points: keys.add(self.int_point_as_float_point(*target))
                        else: keys |= self.chunks[i][target]
            prev_keys = keys
        return keys

##    @timepck.function_timer
    def nearest(self, point, as_points=False, _invert=False):
        point = self.float_point_as_int_point(*point)
        prev_keys = set(self.chunks[-1].keys())
        for i in range(len(self.chunks)-1, -1, -1):
            keys = set()
            scale = self.scaling*i
            
            origin = (point[0]>>scale, point[1]>>scale)
            seeked_targets = seeked_dist = None
            for target in prev_keys:
                dist = self.distance(*target,*origin)
                dist = int(dist) # must be an integer (higher chunks introduce error to distances)
                if seeked_targets is None:
                    seeked_targets = [target]
                    seeked_dist = dist
                elif (not _invert and dist<seeked_dist) or (_invert and dist>seeked_dist):
                    seeked_targets = [target]
                    seeked_dist = dist
                elif (not _invert and dist<(seeked_dist+1)) or (_invert and dist>(seeked_dist-1)):
                    seeked_targets.append(target)
            
            if seeked_targets is not None:
                for seeked_target in seeked_targets:
                    if i==0 and as_points: keys.add(self.int_point_as_float_point(*seeked_target))
                    else: keys |= self.chunks[i][seeked_target]
            prev_keys = keys
        return keys

