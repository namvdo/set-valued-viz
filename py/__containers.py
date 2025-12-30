import math
import numpy as np
from heapq import *
import collections as cls

class Queue2():
    head = 0
    tail = 0
    size = 0
    overwrite = True
    def __init__(self, capacity:int):
        self.list = [None]*capacity

    def resize(self, new_capacity:int):
        old_cap = len(self.list)
        diff = new_capacity-old_cap
        if diff==0: return False
        h = self.head
        t = self.tail
        if diff>0: # bigger
            if h<t:
                self.list = self.list[h:t]
            else:
                self.list = self.list[h:]+self.list[:t]
            self.head = 0
            self.tail = self.size%new_capacity
            self.list.extend([None]*diff)
        else: # smaller
            free_space = old_cap-self.size
            d = max((old_cap-new_capacity)-free_space, 0)
            h += d # move head forward closer to tail
            h %= old_cap
            if h<t:
                self.list = self.list[h:t]
            else:
                self.list = self.list[h:]+self.list[:t]
            self.head = 0
            self.size -= d
            self.tail = self.size%new_capacity
        return True
    
    def add(self, x):
        if not self.overwrite and self.size==len(self.list): return
        if self.size>0 and self.tail == self.head:
            self.head = (self.head+1)%len(self.list)
        self.list[self.tail] = x
        self.tail = (self.tail+1)%len(self.list)
        if self.size<len(self.list): self.size += 1
    
    def take(self):
        if self.size==0: return
        x = self.list[self.head]
        self.head = (self.head+1)%len(self.list)
        self.size -= 1
        return x
    
    def take_all(self):
        while self.size>0: yield self.take()


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
                dist = int(dist)+_invert # must be an integer (higher chunks introduce error to distances); +1 if searching for fartest instead
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













class Graph():
    connections = 0
    class Path:
        distance = 0
        def __init__(self):
            self.vertices = []
            self.distances = []
        def __str__(self):
            string = "Path: "
            if self.vertices:
                string += str(self.vertices)+"\n"
                string += " distances: "+str(self.distances)+"\n"
                string += " total: "+str(self.distance)+"\n"
            else: string += "none"
            return string
        
    def __init__(self):
        self.vertices = set()
        self.edges = {}
        self.weights = {}
    def __len__(self): return len(self.vertices)
    def __str__(self):
        l = len(self.vertices)
        string = f"Graph: {l}\n"
        string += f" connections: {self.connections}\n"
        string += f" deadends: {l-len(self.edges)}\n"
        return string

    def exists(self, source, target=None):
        # return if the vertex exists OR edge exists (if target is not None)
        return source in self.edges and (target is None or target in self.edges[source])

    def get(self, source, target=None, default=None):
        if source in self.edges:
            if target is not None:
                if target in self.edges[source]: return self.weights[(source,target)]
                return default
            return self.edges[source]
        return default

    def remove(self, source):
        self.vertices.remove(source)
        if source in self.edges:
            del self.edges[source]
        
    def connect(self, source, target, weight=1):
        if source==target: return
        key = (source,target)
        if target not in self.vertices:
            self.vertices.add(target)
        if source not in self.edges:
            self.edges[source] = {target}
            self.vertices.add(source)
            if weight is not None: self.weights[key] = weight
            self.connections += 1
            return
        l = self.edges[source]
        if target not in l:
            l.add(target)
            if weight is not None: self.weights[key] = weight
            self.connections += 1
        elif weight is None and key in self.weights: # nullify weight
            del self.weights[key]
        else: self.weights[key] = weight # replace weight

    def disconnect(self, source, target, oneway=False):
        if source in self.edges:
            if target in self.edges[source]:
                self.edges[source].remove(target)
                key = (source,target)
                if key in self.weights:
                    del self.weights[key]
                if not oneway: self.disconnect(target, source, True)
                return True
        return not oneway and self.disconnect(target, source, True)

    def BFS(self, source, target, path=None, visited=None):
        queue = cls.deque()
        queue.append(source)
        if visited is None: visited = {source}
        else: visited.add(source)
        while len(queue):
            active = queue.popleft()
            if path is not None: path.append(active)
            if active==target: return True
            if active in self.edges:
                for t in self.edges[active]:
                    if t not in visited:
                        queue.append(t)
                        visited.add(t)
        return False

    def DFS(self, source, target, path=None, visited=None):
        stack = [source]
        if visited is None: visited = {source}
        else: visited.add(source)
        while len(stack):
            active = stack.pop()
            if path is not None: path.append(active)
            if active==target: return True
            if active in self.edges:
                for t in self.edges[active]:
                    if t not in visited:
                        stack.append(t)
                        visited.add(t)
        return False

    def find_connected(self, source=None):
        if source is None:
            for source in self.vertices: break
        visited = set()
        self.BFS(source, None, None, visited)
        return visited
    
    def find_disconnected(self, source=None):
        return self.vertices-self.find_connected(source)
    
    def is_disconnected(self, source=None):
        return bool(self.find_disconnected(source))

    def get_deadends(self):
        return self.vertices-set(self.edges.keys())
    

    def dijkstra(self, source, target, blacklist=None, whitelist=None):
        # init
        result = self.Path()
        distances = {}
        previous = {}
        priqueue = []
        heapify(priqueue)

        # setup
        heappush(priqueue, (0,source))
        distances[source] = 0

        # loop
        while len(priqueue):
            distance, active = heappop(priqueue)
            if active==target:
                result.vertices.append(active)
                while (prev:=previous.get(active)) is not None:
                    result.distances.append(self.weights.get((prev,active), 0))
                    result.distance += result.distances[-1]
                    active = prev
                    result.vertices.append(active)
                result.vertices = result.vertices[::-1]
                result.distances = result.distances[::-1]
                break
            white = (whitelist is None or active in whitelist)
            if white: black = (blacklist is None or active not in blacklist)
            if active in self.edges and white and black:
                for t in self.edges[active]:
                    d = distance+self.weights.get((active,t), 0)
                    if t in distances:
                        if d<distances[t]:
                            heappush(priqueue, (d,t))
                            distances[t] = d
                            previous[t] = active
                    else:
                        heappush(priqueue, (d,t))
                        distances[t] = d
                        previous[t] = active
        return result













