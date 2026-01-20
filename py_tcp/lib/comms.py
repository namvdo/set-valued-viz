import os
import socket
import socketserver
import inspect

import cryptography
import numpy as np

import cryptpck
import timepck



import json
def custom_json(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f'Cannot serialize object of {type(obj)}')

def object_to_bytes(obj:object) -> bytes:
    return json.dumps(obj, default=custom_json).encode("utf8")
def bytes_to_object(b:bytes) -> object:
    return json.loads(b.decode("utf8"))



##from containerpck
import heapq

class FuncTimer(): # extremely fast
    ms = 0
    def __init__(self):
        self.t = {} # for quick getting
        self.f = {}
        self.heap = []
        heapq.heapify(self.heap)
    def __len__(self):
        return len(self.heap)
    
    def exists(self, k):
        return k in self.t
    
    def new(self, k, v):
        self.t[k] = v+self.ms
        heapq.heappush(self.heap, (v+self.ms,k))
        if k in self.f:
            del self.f[k]
        
    def set(self, k, f, *args, **kwargs):
        self.f[k] = (f, args, kwargs)
        
    def replace(self, k, v):
        for i,x in enumerate(self.heap):
            if x[1]==k:
                self.heap.pop(i)
                heapq.heappush(self.heap, (v+self.ms,k))
                self.t[k] = v+self.ms
                break
    def get(self, k, default=None):
        if k in self.t: return self.t[k]-self.ms
        return default
    
    def delete(self, k):
        if k in self.t:
            for i,x in enumerate(self.heap):
                if x[1]==k:
                    self.heap.pop(i)
                    break
            del self.t[k]
            if len(self.heap)==0: self.ms = 0
        if k in self.f:
            del self.f[k]
            
    def __call__(self, ms):
        if len(self.heap):
            self.ms += ms
            while x:=heapq.heappop(self.heap):
                v,k = x
                if v<self.ms:
                    if k in self.t:
                        del self.t[k]
                    if k in self.f:
                        f, args, kwargs = self.f[k]
                        del self.f[k] # unexpected problems if elsewhere
                        f(*args, **kwargs)
                    if len(self.heap)==0:
                        self.ms = 0
                        break
                else:
                    heapq.heappush(self.heap, (v,k))
                    break

    
    def copy(self):
        copy = type(self)()
        copy.f = self.f.copy()
        copy.heap = self.heap.copy()
        copy.ms = self.ms
        return copy

    def __add__(self, another):
        ms_diff = self.ms-another.ms
        for v,k in another.heap:
            heapq.heappush(self.heap, (v+ms_diff,k))
        self.f.update(another.f)
        return self

#

##from datapck
from io import FileIO


def is_array(x): return type(x)==np.ndarray
def is_str(x, array=True): return type(x) in [str,np.str_] or (array and is_array(x) and str(x.dtype)[:2]=="<U")
def is_iter(x): return type(x) in [tuple,set,list,np.ndarray] or hasattr(x, "__iter__")

def filesave(path, x): # save bytes as a file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with FileIO(path, "w") as f:
        f.write(x)

def fileload(path): # load any file as a bytes
    with FileIO(path, "r") as f:
        return f.read()

def strsave(path, s, enc="utf8"):
    filesave(path+".str", bytes(s, enc))
def strload(path, enc="utf8"):
    if os.path.exists(path+".str"):
        return str(fileload(path+".str"), enc)



def index_nearness(element, list_or_string, i=0): # how close is element to the correct index
    offset = len(list_or_string)
    i %= offset
    if list_or_string[i]==element: return 0
    if element in list_or_string[i+1:]: offset = min(offset, list_or_string[i+1:].index(element)+1)
    if element in list_or_string[:i]: offset = min(offset, list_or_string[:i][::-1].index(element)+1)
    return offset

def strloss(x, y):
    loss = 0
    # character count mismatch +count
    char = dict.fromkeys(list(x)+list(y))
    for xx in char:
        y_n = y.count(xx)
        x_n = x.count(xx)
        loss += abs(y_n-x_n)/max(y_n, x_n)
    loss /= len(char)
    # misplaced characters +offset
    t = len(y)*len(x)
    for i,xx in enumerate(x): loss += index_nearness(xx, y, i)/t
    # add multiplied loss for len mismatch
    return loss/2+loss*(abs(len(y)-len(x))/len(y))

def strsearch(string, options, amount=1): # longer string -> faster search
    str_l = len(string)
    options2 = [x for x in options if str_l<=len(x)]
    if len(options2)==len(options):
        options2 = [x for x in options if string in x]
        if len(options2)>amount: options = options2
    if len(options2)>amount: options = options2
    char_match = False
    for i in range(str_l):
        options2 = [x for x in options if i<len(x) and string[i]==x[i]] # min(i+earliest_start, len(x)-1)
        if len(options2)>amount: options = options2
        char_match = char_match or bool(options2)
    for j in range(str_l):
        options2 = [x for x in options if string[j] in x]
        if len(options2)>amount: options = options2
        char_match = char_match or bool(options2)
    if not char_match: return [] # no character matches -> give up
    if len(options)>amount:
        options2 = [x for x in options if str_l==len(x)]
        if len(options2)>amount: options = options2
    return [options[i] for i in np.argsort([strloss(string, x) for x in options], kind="stable")[:amount]]

#

##from textprc
import re

RE_BOOLEANS = r"(True|False)"
RE_STRINGS = r"(?:\"([^\"]+)\")|(?:\'([^\']+)\')|(?:([^\s\d]+\.\S*|\S*\.[^\s\d]+))"
RE_UNKNOWNS = r"([^\s,]+)"
RE_FLOATS = r"(\-?(?:(?:\d+\.\d*)|(?:\d*\.\d+))(?:e[\-\+]\d+)?)"
RE_INTEGERS = r"(\-?\d+)\b" # [\s,$\(\)] (?:[\s,]|\D|[A-Z])? # (?:[\s,$\(\)])
RE_EC = r"(_\{\d+_\d+\})"
RE_ARGS = re.compile(r"(?:,\s?|\s)?(?:"+RE_EC+r")|(?:"+RE_BOOLEANS+r")|(?:"+RE_STRINGS+r")|(?:"+RE_FLOATS+r")|(?:"+RE_INTEGERS+r")|(None)|(?:"+RE_UNKNOWNS+r")(?:\s*$|,\s?|\s)")
RE_STRINGS = re.compile(r"(?:\"[^\"]+\")|(?:\'[^\']+\')")
RE_TUPLES = re.compile(r"(?:[^\(\)]*)(\([^\(\)]*\))(?:[^\(\)]*)")
RE_KWARGS = re.compile(r"(,\s?|\s)?([^\=\s]+\=[^\=]+)(?:\s*$|,\s?|\s)")
RE_FLOATS = re.compile(RE_FLOATS)
RE_INTEGERS = re.compile(r"[^\.\d]?(\-?\d+)[^\.\d]") # [^\.\d]\b

def read_args_kwargs(t, **embedded):
    # solve strings from t
    while 1:
        t_i = len(embedded)
        if not (l:=RE_STRINGS.findall(t)): break # deepest first
        for i,x in enumerate(l):
            if x:
                embedded_code = "_{"+str(t_i)+"_"+str(i)+"}"
                t = t.replace(x, embedded_code, 1)
                embedded[embedded_code] = x[1:-1]
    # solve tuples from t
    while 1:
        t_i = len(embedded)
        if not (l:=RE_TUPLES.findall(t)): break # deepest first
        for i,x in enumerate(l):
            if x:
                embedded_code = "_{"+str(t_i)+"_"+str(i)+"}"
                t = t.replace(x, embedded_code, 1)
                embedded[embedded_code] = read_args_kwargs(x[1:-1], **embedded)[0]
    kwargs = {}
    keys = ''
    values = ''
    for x in RE_KWARGS.findall(t): # (?:\s[^\=\s]+\=.+)
        k,v = x[1].split("=", 1)
        keys += " '"+k+"'"
        values += " "+v
        t = t.replace(x[0]+x[1], "", 1)
    if keys and values:
        keys = read_args_kwargs(keys, **embedded)[0]
        values = read_args_kwargs(values, **embedded)[0]
        for i,k in enumerate(keys): kwargs[k] = values[i]
    args = []
    for ec,b,sss,ss,s,f,i,nan,u in RE_ARGS.findall(t):
        if ec and ec in embedded:
            d = embedded[ec]
            del embedded[ec]
            args.append(d)
        elif b: args.append(b=="True")
        elif f: args.append(float(f))
        elif i: args.append(int(i))
        elif nan: args.append(None)
        elif u: args.append(str(u))
        else: args.append(s+ss+sss)
    return tuple(args), kwargs
#

IDENTIFIER_LEN = 10

class Empty(): pass

def load_key(path):
    key = strload(path)
    if key: return bytes(key, "utf8")
    key = cryptpck.frnt.generate_key()
    strsave(path, str(key, "utf8"))
    return key

def byte_generator(b, buffer=2048):
    while b:
        yield b[:buffer]
        b = b[buffer:]
        
def quicksend(address, b=None, recv=True, buffer=2048, timeout=5):
    # send bytes and recv yields of bytes
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if timeout is not None: s.settimeout(timeout)
        try: s.connect(address)
        except: return # connection refused
        if b: s.sendall(b)
        if recv:
            while 1:
                try:
                    bb = s.recv(buffer)
                    if not bb: break
                except: break
                yield bb
def send(address, byte_buffer, recv=True, copy=False, buffer=2048, timeout=5):
    # send/recv yields of bytes
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if timeout is not None: s.settimeout(timeout)
        try: s.connect(address)
        except: return # connection refused
        b = b""
        for bb in byte_buffer:
            b += bb
            while len(b)>=buffer: # incase generator gave too many bytes
                s.send(b[:buffer])
                b = b[buffer:]
        if b: s.send(b) # the rest
        s.send(b"") # empty for done
        if recv:
            while 1:
                try:
                    bb = s.recv(buffer)
                    if not bb: break
                except: break
                yield bb


def copy(to_address, byte_buffer, buffer=2048, timeout=5):
    # send & copy yields of bytes
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if timeout is not None: s.settimeout(timeout)
        try: s.connect(to_address)
        except: return # connection refused
        b = b""
        for bb in byte_buffer:
            b += bb
            yield bb
            while len(b)>=buffer: # incase generator gave too many bytes
                s.send(b[:buffer])
                b = b[buffer:]
        if b: s.send(b) # the rest
        s.send(b"") # empty for done
        
def recv(listen_address, from_address=None, buffer=2048, timeout=5):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if timeout is not None: s.settimeout(timeout)
        s.bind(listen_address)
        s.listen(1)
        conn, addr = s.accept()
        with conn:
            if from_address is None or addr[0]==from_address:
                while 1:
                    try:
                        b = conn.recv(buffer)
                        if not b: break
                    except: break
                    yield b

def siphon(listen_address, to_address, from_address=None, buffer=2048, timeout=5):
    received_bytes = recv(listen_address, from_address=from_address, buffer=buffer, timeout=timeout)
    yield from copy(to_address, received_bytes, buffer=buffer, timeout=timeout)





if __name__ == "__main__":
    pass
    
