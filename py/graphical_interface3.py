from _imports import *
from _gui_matplotlib import *
from _quick_visuals import *

from normals_model2 import Model as NormalsModel

SAVEDIR = os.path.join(WORKDIR, "saves")




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
            dist, _ = hausdorff_distance8(points1, points2)
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





class ModelInstance():
    key = None
    
    model = None
    points_history = None # -> PointsHistory
    
    fig_resolution = None # -> IntegerField
    png_resolution = None # -> IntegerField
    min_x = max_x = None # -> FloatFields
    min_y = max_y = None # -> FloatFields
    min_hausdorff = max_hausdorff = None # -> FloatFields
    colors = None # -> dict

    viewport = None # -> Viewport
    canvas = None # -> Canvas
    
    step = None # -> IntegerField
    printbox = None # -> TextBox
    auto_extend = None # -> BooleanSwitch
    
    yaw = None # -> IntegerField
    tilt = None # -> IntegerField
    pitch = None # -> IntegerField
    
    def __init__(self, on_destroy, **kwargs):
        for k,v in kwargs.items():
            if hasattr(self, k): setattr(self, k, v)

        self.checkpoints = Checkpoints()
        self.points_history = PointsHistory(10)
        
        win = nice_window(self.key, on_destroy=on_destroy, resizeable=True)
        set_padding(win)
        
        self.viewport = Viewport(win, 1280, 720)
        self.canvas = Canvas(self.viewport.get_widget())
        
        k,f = self.canvas.movable_window("Model Controls", (0,0))
        self._init_model_control_panel(f)
        
        k,f = self.canvas.movable_window("Viewport Controls", (600,0))
        self._init_viewport_settings_panel(f)
        
        

    def _init_model_control_panel(self, root):
        frame = nice_frame(root, anchor="nw")
        
        # buttons
        f = padded_frame(frame, anchor="c", side=tk.TOP, fill=tk.BOTH)
        b = nice_button(f, text="Recalc Step", command=self.model_recalc_from_checkpoint)
        ff = nice_frame(f, anchor="c", side=tk.TOP, fill=tk.BOTH)
        self.step = IntegerField(ff, val=1, low=0, high=None, side=tk.RIGHT, width=6, justify="center")
        b = nice_button(ff, text="Reset", side=tk.LEFT, width=6, command=self.model_reset)
        def _press():
            step = self.step.get()
            if step>0: self.model_process(step)
            else: self.model_reset()
        b = nice_button(ff, text="Go to Step", command=_press)
        
##        b = nice_button(f, text="Find Periodic Points (WIP)", command=None)
        #

        #
        f = nice_titled_frame(frame, "model configuration", side=tk.TOP)
        
        f_top = nice_frame(f, side=tk.TOP)
        f_left = nice_frame(f_top, side=tk.LEFT)
        ff = padded_frame(f_left, anchor="nw", side=tk.TOP)
        text = f"fx = {self.model.function.x}"
        text += f"\nfy = {self.model.function.y}"
        text += f"\nstart = {readable_point(self.model.start_point)}"
        label = nice_label(ff, text=text, anchor="nw", justify="left")
        
        #
        width = 4
        fff = nice_titled_frame(f_left, "precision", side=tk.TOP)
        
        def on_update(): self.model.point_density = self.point_density.get()
        self.point_density = IntegerField(fff, val=self.model.point_density, low=1, high=100, width=width, label_text="point density", justify="center", on_update=on_update)
        
        
        fff = nice_titled_frame(f_left, "noise", side=tk.TOP)
        
        def on_update(): self.model.update_noise_geometry(sides=self.noise_vertices.get())
        self.noise_vertices = IntegerField(fff, val=0, low=0, high=None, width=width, label_text="vertices", justify="center", on_update=on_update)
        
        def on_update(): self.model.update_noise_geometry(rotation=self.noise_rotation.get())
        self.noise_rotation = IntegerField(fff, val=0, low=0, mod=360, width=width, label_text="rotation", justify="center", on_update=on_update)
        
        def on_update(): self.model.epsilon = self.noise_distance.get()
        self.noise_distance = FloatField(fff, val=self.model.epsilon, low=0, high=None, scroll=1, width=width*2, label_text="distance", on_update=on_update)
        #
        
        #
        width = 10
        required_constants = self.model.function.required_constants()
        if len(required_constants)>0:
            self.constant_fields = {}
            on_updates = {}
            def on_update(k): self.model.function.constants[k] = self.constant_fields[k].get()
            def new_on_update(k): on_updates[k] = lambda : on_update(k)
            
            ff = nice_titled_frame(f_top, "constants", anchor="nw", side=tk.TOP, fill=tk.NONE)
            
            for k in sorted(required_constants):
                v = self.model.function.constants.get(k)
                new_on_update(k)
                
                obj = FloatField(ff, val=v, low=None, high=None, scroll=1, width=width, label_text=k, on_update=on_updates[k])
                self.constant_fields[k] = obj

                if v is None: self.model.function.constants[k] = obj.get()
        #
        
        
        #
        width = 5
        fff = nice_titled_frame(f, "hausdorff", side=tk.TOP)
##        self.hausdorff_periods = StringList(ff, [], anchor="w", justify="left", visible=3, side=tk.TOP, fill=tk.BOTH)
        
        ffff = nice_frame(fff, anchor="c", side=tk.TOP)
        self.min_hausdorff = FloatField(ffff, val=None, low=0, high=None, can_disable=True, width=width, side=tk.LEFT, justify="center")
        nice_label(ffff, text="<= dist <=", side=tk.LEFT)
        self.max_hausdorff = FloatField(ffff, val=None, low=0, high=None, can_disable=True, width=width, side=tk.LEFT, justify="center")
        #
        
        f = nice_titled_frame(frame, "Print", side=tk.TOP)
        self.printbox = TextBox(f, width=28, height=6, side=tk.TOP, fill=tk.BOTH)
        
        self.refresh_printbox()
        
    def _init_viewport_settings_panel(self, root):
        frame = nice_frame(root, anchor="nw")
        
        # buttons
        f = padded_frame(frame, anchor="c", side=tk.TOP, fill=tk.BOTH)
        ff = nice_frame(f, anchor="c", side=tk.TOP, fill=tk.BOTH)
        b = nice_button(ff, text="Refresh", command=self.refresh_viewport) # Matplotlib Figure
        ff = nice_frame(f, anchor="c", side=tk.TOP, fill=tk.BOTH)
        b = nice_button(ff, text="Save PNG", command=self.save_png)
        #
        
        f = nice_titled_frame(frame, "viewport settings")
        ff = nice_frame(f, side=tk.TOP, fill=tk.BOTH)
        
        #
        width = 8
        fff = nice_titled_frame(ff, "resolution", side=tk.LEFT)
        self.fig_resolution = IntegerField(fff, val=512, low=32, high=None, width=width, label_text="figure", justify="center")
        self.png_resolution = IntegerField(fff, val=5120, low=32, high=None, width=width, label_text="PNG", justify="center")
        #
        
        #
        width = 5
        fff = nice_titled_frame(ff, "extend limits", side=tk.RIGHT)
        
        ffff = nice_frame(fff, anchor="c", side=tk.TOP)
        self.min_x = FloatField(ffff, val=None, low=None, high=None, can_disable=True, side=tk.LEFT, width=width)
        nice_label(ffff, text="<= x <=", side=tk.LEFT)
        self.max_x = FloatField(ffff, val=None, low=None, high=None, can_disable=True, side=tk.LEFT, width=width)
        
        ffff = nice_frame(fff, anchor="c", side=tk.TOP)
        self.min_y = FloatField(ffff, val=None, low=None, high=None, can_disable=True, side=tk.LEFT, width=width)
        nice_label(ffff, text="<= y <=", side=tk.LEFT)
        self.max_y = FloatField(ffff, val=None, low=None, high=None, can_disable=True, side=tk.LEFT, width=width)

        self.auto_extend = BooleanSwitch(fff, text="Auto Extend", side=tk.TOP)
        #

        
        ff = nice_frame(f, side=tk.TOP, anchor="c", fill=tk.BOTH)
        
        #
        width = 8
        fff = nice_titled_frame(ff, "Camera", side=tk.LEFT, anchor="nw", fill=tk.Y)
        ffff = nice_frame(fff, side=tk.LEFT, anchor="c", fill=tk.BOTH)
        
        self.pitch = FloatField(ffff, low=0, mod=360, width=width, label_text="pitch")
        self.yaw = FloatField(ffff, low=0, mod=360, width=width, label_text="yaw")
        self.tilt = FloatField(ffff, low=0, mod=360, width=width, label_text="tilt")
        
        self.camera_distance = FloatField(ffff, val=None, low=0, can_disable=True, width=width, label_text="focal len.")
        
        ffff = nice_frame(fff, side=tk.LEFT, anchor="n")
        def _press():
            rad_deg = 180/np.pi
            self.pitch.set(315)
            self.yaw.set(np.pi/5.1 * rad_deg)
            self.tilt.set(330)
            self.camera_distance.set(None) # disable perspective
            
        b = nice_button(ffff, text="Isometric 1", side=tk.TOP, command=_press)
        
        def _press():
            rad_deg = 180/np.pi
            self.pitch.set(225)
            self.yaw.set(np.pi/5.1 * rad_deg)
            self.tilt.set(210)
            self.camera_distance.set(None) # disable perspective
            
        b = nice_button(ffff, text="Isometric 2", side=tk.TOP, command=_press)
        #
        
        #
        width = 6
        fff = nice_titled_frame(ff, "Grid Settings", side=tk.LEFT, anchor="nw", fill=tk.Y)
        self.grid_x = FloatField(fff, val=0, width=width, label_text="x")
        self.grid_y = FloatField(fff, val=0, width=width, label_text="y")
        self.grid_size = FloatField(fff, val=0.1, low=0, width=width, label_text="size")
        #
        
        #
        ff = nice_titled_frame(f, "colors", side=tk.TOP)
        
        self.colors = {
            "background": [255,255,255,255],
            "grid": [0,127,127,63],
            "points": [255,0,0,255],
            "boundary": [0,255,0,255],
            "normals": [0,0,0,127],
            "prev. points": [0,0,255,0],
            "hausdorff line": [255,127,0,0],
            "axes": [0,127,255,255],
            }
##        longest_name_len = len(max(self.colors.keys(), key=len))
        for name,color in self.colors.items():
            fff = nice_frame(ff, side=tk.TOP, anchor="ne")
            nice_label(fff, text=name, anchor="w", side=tk.LEFT) # , width=longest_name_len
            nice_RGBA_selector(fff, color)
        #

    def draw(self, resolution:int):
        
        drawing = ImageDrawing()
        
        drawing.yaw = self.yaw.get()/180*np.pi
        drawing.tilt = self.tilt.get()/180*np.pi
        drawing.pitch = self.pitch.get()/180*np.pi
        drawing.set_color(*np.divide(self.colors["background"], 255))
        
        color = np.divide(self.colors["boundary"], 255)
        if color[3]>0:
            obj = drawing.lines(*self.model.get_boundary_lines())
            obj.set_color(*color)
        
        color = np.divide(self.colors["normals"], 255)
        if color[3]>0:
            obj = drawing.lines(*self.model.get_inner_normals())
            obj.set_color(*color)
        
        color = np.divide(self.colors["prev. points"], 255)
        if color[3]>0:
            obj = drawing.points(self.model._prev_points)
            obj.set_color(*color)
        
        color = np.divide(self.colors["points"], 255)
        if color[3]>0 and len(self.model)>1:
            obj = drawing.points(self.model._points)
            obj.set_color(*color)
        
        color = np.divide(self.colors["hausdorff line"], 255)
        if color[3]>0:
            line = self.model.hausdorff_line
            obj = drawing.lines([line[0]], [line[1]])
            obj.set_color(*color)
        
        # extend limits
        auto = self.auto_extend.get()
        auto_rounding = 3
        min_x = self.min_x.get()
        max_x = self.max_x.get()
        min_y = self.min_y.get()
        max_y = self.max_y.get()
        if min_x is not None:
            drawing.tl[0] = min(drawing.tl[0], min_x)
            if auto: self.min_x.set(round(drawing.tl[0], auto_rounding))
        if max_x is not None:
            drawing.br[0] = max(drawing.br[0], max_x)
            if auto: self.max_x.set(round(drawing.br[0], auto_rounding))
        if min_y is not None:
            drawing.tl[1] = min(drawing.tl[1], min_y)
            if auto: self.min_y.set(round(drawing.tl[1], auto_rounding))
        if max_y is not None:
            drawing.br[1] = max(drawing.br[1], max_y)
            if auto: self.max_y.set(round(drawing.br[1], auto_rounding))

        def _2d_to_3d():
            # draw 2d as 3d
            tl = np.pad(drawing.tl, (0,1))
            smallest_side = np.argmin(drawing.br-drawing.tl)
            tl[2] = tl[smallest_side]
            br = np.pad(drawing.br, (0,1))
            br[2] = br[smallest_side]
            drawing.update_tl_br(tl, br)
            #

        _2d_to_3d()
        
        color = np.divide(self.colors["axes"], 255)
        if color[3]>0:
            starts, ends = drawing.get_axis_lines()
            for i in range(len(starts)):
                lines = np.linspace(starts[i], ends[i], 10)
                obj = drawing.lines(lines[:-1], lines[1:])
                obj.set_color(r=color[i%3], g=color[(i+1)%3], b=color[(i+2)%3], a=color[3])
        
        color = np.divide(self.colors["grid"], 255)
        size = self.grid_size.get()
        if color[3]>0 and size>0:
            obj = drawing.grid((self.grid_x.get(),self.grid_y.get()), size)
            obj.set_color(*color)
        
        camera_dist = self.camera_distance.get()
        image = drawing.draw(resolution, camera_dist=camera_dist)
        return image, drawing.get_full_extent()
        
    def save_png(self):
        path = os.path.join(SAVEDIR, "test.png")
        makedirs(path)
        image, _ = self.draw(self.png_resolution.get())
        PIL_image_from_array(image).save(path, optimize=True)
    
    def model_reset(self):
        self.checkpoints.clear()
        self.points_history.clear()
        self.model.reset()
        self.step.set(1)
        self.refresh_printbox()
        self.refresh_viewport()

    def model_save_checkpoint(self, key="prev"):
        self.checkpoints.save(key, step=self.step.get(), model=self.model, points_history=self.points_history)

    def model_load_checkpoint(self, key="prev"):
        cp = self.checkpoints.load(key)
        if cp is not None:
            self.step.set(cp.step)
            self.points_history = cp.points_history.copy()
            for attr in ["_points","_normals","_prev_points","_prev_normals","_timestep","tij"]:
                self.model.copyattr(cp.model, attr)
            return True
        return False
        
    def model_recalc_from_checkpoint(self): # recalculate from previous checkpoint with updated settings
        target_step = self.step.get()-1
        if self.model_load_checkpoint("prev"):
            self.model_process(target_step)
    
    def update_points_history(self):
        self.points_history.update(self.model.get_points().copy())
        
    def model_process(self, target_step:int):
        self.model_save_checkpoint("prev")
        self.update_points_history()

        def _hausdorff_distance_is_in_range():
            min_hdr = self.min_hausdorff.get()
            max_hdr = self.max_hausdorff.get()
            if min_hdr is not None or max_hdr is not None:
                dists = self.points_history.hausdorff(self.model.get_points())
                for dist in dists:
                    if min_hdr is not None and dist<min_hdr: return False
                    if max_hdr is not None and dist>max_hdr: return False
            return True

        last_step = (target_step-1) # model's steps are 1 less (starts from 0)
        for model_step in self.model.process(last_step):
            target_step = model_step+1
            
            if not _hausdorff_distance_is_in_range(): break
            
            if model_step!=last_step: self.update_points_history()
        
        self.refresh_printbox(model_step+1)
        self.refresh_viewport()
        self.step.set(target_step+1) # to display next step in the field
    
    def refresh_printbox(self, step:int = 0):
        dists = self.points_history.hausdorff(self.model.get_points())
        self.printbox.clear()
        if dists:
            string = "Hausdorff periods:"
            for i,dist in enumerate(dists):
                v = readable_float(dist, 6)
                string += f"\n {i+1}: {v}"
            
            self.printbox.print(string)
##            write_to_textbox(self.printbox, string, clear=True)
        
        
    def refresh_viewport(self):
        if self.model.has_points():
            image, extent = self.draw(self.fig_resolution.get())
            
            title = f"{len(self.model)} points"
            title += f"\nimage: {image.shape[:2]}"
            self.viewport.update(image, extent, title=title)
        else:
            self.viewport.clear()

class InterfaceMain():
    instances_created = 0
    
    def __init__(self):
        self.model_base = NormalsModel()
        self.model_instances = {}
        self._init_main_window()

    def _init_main_window(self):
        self.window = nice_window("main")
        set_padding(self.window)
        
        leftside = nice_frame(self.window, side=tk.LEFT, fill=tk.BOTH)
        
        # buttons
        f = padded_frame(leftside, side=tk.TOP, fill=tk.BOTH)
        b = nice_button(f, text="New Instance", side=tk.TOP, command=self.start_model_instance)
        #

        #
        width = 8
        ff = nice_titled_frame(leftside, "function", side=tk.TOP, fill=tk.NONE)
        def _update(identifier, string):
            getattr(self.model_base.function, identifier).string = string
        
        for k,f in self.model_base.function:
            field = nice_labeled_field(ff, k, anchor="ne", update_handler=_update)
            field.insert(0, f.string)
        
        ff = nice_titled_frame(leftside, "start point", side=tk.TOP, fill=tk.NONE, anchor="ne")
        
        def on_update_x(): self.model_base.start_point[0] = self.start_x.get()
        def on_update_y(): self.model_base.start_point[1] = self.start_y.get()
        self.start_x = FloatField(ff, val=0, low=None, high=None, on_update=on_update_x, label_text="x", width=width)
        self.start_y = FloatField(ff, val=0, low=None, high=None, on_update=on_update_y, label_text="y", width=width)
        #
        
        f = nice_titled_frame(self.window, "log", side=tk.LEFT)
        self.logbox = TextBox(f, width=36, height=12)


        

##        strings = [str(i)*(i+1) for i in range(10)]
##        self.testlist = StringList(f, strings, anchor="w", justify="left", visible=2, side=tk.TOP, fill=tk.BOTH) # , label_text="test"
##
##        self.testlist.append("asd")
##        self.testlist.pop(0)
##        self.testlist.insert(1, "asdasd")

    

    def start_model_instance(self):
        key = "model_"+str(self.instances_created)
        def on_destroy():
            try:
                del self.model_instances[key]
                self.logbox.print(f"destroyed {key}")
            except: pass
        
        normals_model = self.model_base.copy()
        normals_model.printing = False
        def _print(string):
            self.logbox.print(key, string)
        normals_model.print_func = _print
        
        instance = ModelInstance(on_destroy, model=normals_model, key=key)
        self.model_instances[key] = instance
        self.instances_created += 1
        self.logbox.print(f"created {key}")
        
        
    def start(self):
        self.window.mainloop()


if __name__ == "__main__":
    InterfaceMain().start()
