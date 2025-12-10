import numpy as np


class PeriodicPointBuffer():
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
    buffer = PeriodicPointBuffer(10)
    buffer.threshold = 0.01
    
    points = np.random.random((7,2)) # testable 'repeating' points
    for i in range(len(buffer)*5):
        p = points[i%len(points)]
        p += np.random.random(2)/200 # add randomness
        print(i, "\t", p, buffer.log(p), buffer.is_stable())
