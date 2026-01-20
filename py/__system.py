import os
import math
import pickle

# from timepck
from time import perf_counter_ns as nspec # nanoseconds
def function_timer(func):
    def wrapper(*args, **kwargs):
        print(func)
        t_start = nspec()
        out = func(*args, **kwargs)
        t_stop = nspec()
        print(f"{(t_stop-t_start)/1e6:.3f} ms")
        return out
    return wrapper
#

# from datapck
def load(path, default=None, makedirs_ok=True):
    if makedirs_ok: makedirs(path)
    if os.path.isfile(path):
        with open(path, "rb") as f:
            try: obj = pickle.load(f)
            except TypeError: obj = default
            except pickle.UnpicklingError: obj = default
        return obj
    return default
def save(path, obj):
    makedirs(path)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
#

SI_VALUES_UNIT_PREFIXES_4 = "kMGTPEZY"
def readable_filesize(size):
    p = 0
    if size>0: p += min(int(math.log(size, 1024)), len(SI_VALUES_UNIT_PREFIXES_4))
    if p>0: size /= 1024**p
    return f"{size:.1f} "+(SI_VALUES_UNIT_PREFIXES_4[p-1] if p>0 else "")+"B"





# from ospck
def list_files(path):
    try:
        if os.path.isdir(path):
            for i in os.listdir(path):
                i = os.path.join(path, i)
                if os.path.isfile(i): yield i
    except PermissionError: pass # denied

def list_folders(path):
    try:
        if os.path.isdir(path):
            for f in os.listdir(path):
                f = os.path.join(path, f)
                if os.path.isdir(f): yield f
    except PermissionError: pass # denied

def list_files_recur(path):
    for f in list_folders(path):
        yield from list_files_recur(f)
    yield from list_files(path)

def get_folder(path):
    if "." in os.path.basename(path): path = os.path.dirname(path)
    return path

def makedirs(path):
    if path:=get_folder(path):
        os.makedirs(path, exist_ok=True)

def getsize(path):
    if os.path.isfile(path): return os.path.getsize(path)
    return getsize_folder(path)

def getsize_folder(path):
    size = 0
    for i in list_files_recur(path):
        size += os.path.getsize(i)
    return size

def delete_file(path):
    if os.path.isfile(path):
        os.remove(path)
        return True
    return False

def delete_folder(path):
    if os.path.isdir(path):
        for x in list_files(path): delete_file(x)
        for x in list_folders(path): delete_folder(x)
        os.rmdir(path)
        return True
    return False
#



