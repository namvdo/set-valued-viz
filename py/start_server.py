import sys, os
import numpy as np
sys.path.insert(0, 'lib')
from comms_server_unencrypted import start_server, Server

import normals_model3 as nm

class ArrayHistory():
    updates = 0
    capacity = 10
    def __init__(self, capacity=10):
        self.capacity = capacity
        self.stack = []

    def __len__(self): return len(self.stack)

    def set_capacity(self, capacity):
        self.capacity = capacity
        while len(self.stack)>=self.capacity: self.stack.pop(0)
    
    def update(self, points):
        while len(self.stack)>=self.capacity: self.stack.pop(0)
        self.stack.append(points)
        self.updates += 1

    def clear(self):
        self.stack.clear()
        
    def pop(self):
        return self.stack.pop()

    def copy(self):
        new = type(self)()
        new.updates = self.updates
        new.stack = self.stack.copy()
        new.capacity = self.capacity
        return new
        
class PointsHistory(ArrayHistory):
    def hausdorff(self, points1):
        dists = []
        for i,points2 in enumerate(self.stack[::-1]):
            dist, _ = nm.hausdorff_distance8(points1, points2)
            dists.append(dist)
        return dists

class Checkpoints():
    class Checkpoint:
        def __init__(self, **kwargs):
            for k,v in kwargs.items():
                if hasattr(v, "copy"): v = v.copy()
                setattr(self, k, v)
    
    def __init__(self): self.checkpoints = {}
    def __len__(self): return len(self.checkpoints)
    def clear(self): self.checkpoints.clear()
    def delete(self, key):
        if key in self.checkpoints: del self.checkpoints[key]
    def save(self, key, **kwargs):
        self.checkpoints[key] = self.Checkpoint(**kwargs)
    def load(self, key, default=None):
        return self.checkpoints.get(key, default)



class ModelManager():
    step = None
    model = None
    checkpoints = None
    points_history = None
    
    class ModelManagerConfig:
        hd_min = hd_max = None
    
    def __init__(self, model):
        self.step = 0
        self.model = model
        self.checkpoints = Checkpoints()
        self.points_history = PointsHistory(16)
        self.config = self.ModelManagerConfig()
        
    def _log_points_history(self):
        self.points_history.update(self.model.points.copy())

    def hausdorff_distances(self):
        return self.points_history.hausdorff(self.model.points)
    
    def _hausdorff_distance_ok(self):
        if self.config.hd_min is not None or self.config.hd_max is not None:
            dists = self.hausdorff_distances()
            for dist in dists:
                if self.config.hd_min is not None and dist<self.config.hd_min: return False
                if self.config.hd_max is not None and dist>self.config.hd_max: return False
        return True

    def reset(self):
        ndim = self.model.points.shape[1]
        self.model.first_step(ndim==3)
        self.step = 0

    def recalc(self):
        target_step = self.step
        if self.load("prev"):
            return self._forward(target_step-self.step)
        return True
    
    def goto(self, step=0):
        if step>self.step:
            return self.forward(step-self.step)
        elif step<self.step:
            self.reset()
            if step>0:
                return self.forward(step)
        return True
    
    def _forward(self, steps=1):
        for i in range(steps):
            self._log_points_history()
            if not self.model.next_step(): return False
            if not self._hausdorff_distance_ok(): break
            self.step += 1
        return True

    def forward(self, steps=1):
        self.save("prev")
        return self._forward(steps)

    def save(self, key):
        self.checkpoints.save(key, step=self.step, model=self.model, points_history=self.points_history, config=self.config)
    
    def load(self, key, full=False):
        cp = self.checkpoints.load(key)
        if cp is not None:
            self.step = cp.step
            self.points_history = cp.points_history.copy()
            if full:
                self.config = cp.config
                self.model = cp.model.copy()
            else:
                for attr in ["points","normals"]:
                    self.model.copyattr(cp.model, attr)
            return True
        return False
    

class ModelServer(Server):
    PRINT = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.models = {} # instance -> ModelManager

    def datetime(self): pass
    def hello(self): pass
    
    def alg_solve_string(self, string, instance=None, **inputs):
        return nm.solve(string, **inputs)
    
    def alg_derivate_string(self, string, symbol, instance=None) -> str:
        return nm.derivative(string, symbol)

    def alg_hausdorff_distance(self, points1, points2, instance=None) -> tuple:
        return nm.hausdorff_distance8(np.array(points1), np.array(points2))[1]
    
    def model_expire(self, minutes=1, instance=None) -> bool:
        k = f"model_expire_{instance}"
        ms = minutes*60000 # 10 min
        if self.timer.exists(k):
            self.timer.replace(k, ms)
        else:
            self.timer.new(k, ms)
            self.timer.set(k, self.delete, instance)
        return True
    
    def model_delete(self, instance=None) -> bool:
        if instance in self.models:
            del self.models[instance]
            return True
        return False
    
    def model_start(self, point=(0,0,0), x="x", y="y", z=None, instance=None) -> bool:
        model = nm.Model()
        model.update_start(*point)
        model.update_function(x=x, y=y, z=z)
        model.first_step(z is not None)
        mm = ModelManager(model)
        self.models[instance] = mm
        return True

    def model_set_const(self, instance=None, **constants) -> bool:
        if (mm:=self.models.get(instance)) is None: return False
        mm.model.update_constants(**constants)
        return True
    
    def model_get_const(self, instance=None):
        if (mm:=self.models.get(instance)) is None: return None
        return mm.model.function.constants

    def model_set_config(self, epsilon=None, density=None, hd_min=None, hd_max=None, instance=None) -> bool:
        if (mm:=self.models.get(instance)) is None: return False
        if epsilon is not None: mm.model.epsilon = epsilon
        if density is not None: mm.model.point_density = density
        if hd_max is not None: mm.config.hd_max = hd_max
        if hd_min is not None: mm.config.hd_min = hd_min
        return True
    
    def model_get_config(self, instance=None):
        if (mm:=self.models.get(instance)) is None: return None
        return {
            "epsilon": mm.model.epsilon,
            "density": mm.model.point_density,
            "hd_max": mm.config.hd_max,
            "hd_min": mm.config.hd_min,
            }
    
    def model_step_next(self, steps=1, instance=None) -> bool:
        if (mm:=self.models.get(instance)) is None: return False
        return mm.forward(steps)
    
    def model_step_goto(self, step=0, instance=None) -> bool:
        if (mm:=self.models.get(instance)) is None: return False
        return mm.goto(step)

    def model_step_redo(self, instance=None) -> bool:
        if (mm:=self.models.get(instance)) is None: return False
        return mm.recalc()

    def model_hausdorff(self, instance=None):
        if (mm:=self.models.get(instance)) is None: return None
        return mm.hausdorff_distances()
    
    def model_state(self, instance=None) -> dict:
        if (mm:=self.models.get(instance)) is None: return None
        return {
            "step": mm.step,
            "points": mm.model.points,
            "normals": mm.model.normals,
            }

    def model_save(self, name, instance=None) -> bool:
        if (mm:=self.models.get(instance)) is None: return False
        mm.save(name)
        return True
    
    def model_load(self, name, instance=None) -> bool:
        if (mm:=self.models.get(instance)) is None: return False
        return mm.load(name, full=True)





    

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int)
    args = parser.parse_args()
    port = args.port
    TESTADDRESS = ("localhost", port)
    print(f"Starting the server at port: {port}")
    start_server(ModelServer, TESTADDRESS, "./server", "key")
