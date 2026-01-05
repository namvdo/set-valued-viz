from _imports import *
from _gui_matplotlib import *
from _quick_visuals import *

from normals_model2 import Model as NormalsModel

SAVEDIR = os.path.join(WORKDIR, "saves")

class ModelInstance():
    key = None
    model = None
    model_prev = None # copied model
    
    fig_resolution = None # -> IntegerField
    png_resolution = None # -> IntegerField
    min_x = max_x = None # -> FloatFields
    min_y = max_y = None # -> FloatFields
    min_hausdorff = max_hausdorff = None # -> FloatFields
    colors = None # -> dict

    viewport = None # -> Viewport
    
    step = None # -> IntegerField
    step_data_label = None # -> label
    auto_extend = None # -> BooleanSwitch

    yaw = None # -> IntegerField
    tilt = None # -> IntegerField
    pitch = None # -> IntegerField
    
    def __init__(self, on_destroy, **kwargs):
        for k,v in kwargs.items():
            if hasattr(self, k): setattr(self, k, v)
        
        win = nice_window(self.key, on_destroy=on_destroy)
        set_padding(win)
##        top = nice_frame(win, side=tk.TOP, fill=tk.BOTH)
##        nice_label(top, text="asd", anchor="w")
        mid = nice_frame(win, side=tk.TOP)
        self._init_model_control_panel(mid)
        self._init_viewport_panel(mid)
        self._init_viewport_settings_panel(mid)
##        bot = nice_frame(win, side=tk.TOP, fill=tk.BOTH)
        

    def _init_model_control_panel(self, root):
        frame = nice_frame(root, anchor="nw")
        
        # buttons
        f = padded_frame(frame, anchor="c", side=tk.TOP, fill=tk.BOTH)
        b = nice_button(f, text="Recalc Step", command=self.model_recalculate)
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
        width = 8
        ff = nice_titled_frame(frame, "hausdorff distance range", side=tk.TOP)
        fff = nice_frame(ff, anchor="c", side=tk.TOP)
        self.min_hausdorff = FloatField(fff, val=None, low=0, high=None, can_disable=True, width=width, side=tk.LEFT, justify="center")
        nice_label(fff, text="<= dist <=", side=tk.LEFT)
        self.max_hausdorff = FloatField(fff, val=None, low=0, high=None, can_disable=True, width=width, side=tk.LEFT, justify="center")
        #
        
        #
        ff = nice_titled_frame(frame, "step data", anchor="n", side=tk.TOP)
        self.step_data_label = nice_label(ff, width=8, justify="left", anchor="w")
        #

        #
        f = nice_titled_frame(frame, "model configuration", side=tk.TOP)
        ff = padded_frame(f, anchor="nw", side=tk.TOP)
        text = f"fx = {self.model.function.x}"
        text += f"\nfy = {self.model.function.y}"
        text += f"\nstart = {readable_point(self.model.start_point)}"
        label = nice_label(ff, text=text, anchor="nw", justify="left")

##        def _update():
##            print(var.get())
##        var = string_cycler(label, list("abcdef"), on_update=_update)
##        
##        def _update():
##            print(var1.get())
##        var1 = integer_cycler(label, 256, on_update=_update)
        
        #
        ff = nice_frame(f, anchor="nw", side=tk.LEFT)
        width = 4
        fff = nice_titled_frame(ff, "precision", side=tk.TOP)
        
        def on_update(): self.model.point_density = self.point_density.get()
        self.point_density = IntegerField(fff, val=self.model.point_density, low=1, high=100, width=width, label_text="point density", justify="center", on_update=on_update)
        
        
        fff = nice_titled_frame(ff, "noise", side=tk.TOP)
        
        def on_update(): self.model.update_noise_geometry(sides=self.noise_vertices.get())
        self.noise_vertices = IntegerField(fff, val=0, low=0, high=None, width=width, label_text="vertices", justify="center", on_update=on_update)
        
        def on_update(): self.model.update_noise_geometry(rotation=self.noise_rotation.get())
        self.noise_rotation = IntegerField(fff, val=0, low=0, mod=360, width=width, label_text="rotation", justify="center", on_update=on_update)
        
        def on_update(): self.model.epsilon = self.noise_distance.get()
        self.noise_distance = FloatField(fff, val=self.model.epsilon, low=0, high=None, step=1, width=width*2, label_text="distance", on_update=on_update)
        #

        #
        width = 10
        required_constants = self.model.function.required_constants()
        if len(required_constants)>0:
            self.constant_fields = {}
            on_updates = {}
            def on_update(k): self.model.function.constants[k] = self.constant_fields[k].get()
            def new_on_update(k): on_updates[k] = lambda : on_update(k)
            
            ff = nice_titled_frame(f, "constants", anchor="nw", side=tk.LEFT)
            
            for k in sorted(required_constants):
                v = self.model.function.constants.get(k)
                new_on_update(k)
                
                obj = FloatField(ff, val=v, low=None, high=None, step=1, width=width, label_text=k, on_update=on_updates[k])
                self.constant_fields[k] = obj

                if v is None: self.model.function.constants[k] = obj.get()
        
        self.refresh_stepdata()
        

    def _init_viewport_panel(self, root):
        f = nice_frame(root, anchor="nw")
        ff = nice_titled_frame(f, "viewport")
        self.viewport = Viewport(ff)
        
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
        self.min_x = FloatField(ffff, val=None, low=None, high=None, can_disable=True, side=tk.LEFT, justify="center", width=width)
        nice_label(ffff, text="<= x <=", side=tk.LEFT)
        self.max_x = FloatField(ffff, val=None, low=None, high=None, can_disable=True, side=tk.LEFT, justify="center", width=width)
        
        ffff = nice_frame(fff, anchor="c", side=tk.TOP)
        self.min_y = FloatField(ffff, val=None, low=None, high=None, can_disable=True, side=tk.LEFT, justify="center", width=width)
        nice_label(ffff, text="<= y <=", side=tk.LEFT)
        self.max_y = FloatField(ffff, val=None, low=None, high=None, can_disable=True, side=tk.LEFT, justify="center", width=width)

        self.auto_extend = BooleanSwitch(fff, text="Auto Extend", side=tk.TOP)
        #

        #
        width = 8
        ff = nice_titled_frame(f, "Camera", side=tk.TOP, anchor="nw", fill=tk.Y)
        fff = nice_frame(ff, side=tk.LEFT, anchor="c", fill=tk.BOTH)
        
        self.pitch = FloatField(fff, low=0, mod=360, width=width, label_text="pitch", justify="center")
        self.yaw = FloatField(fff, low=0, mod=360, width=width, label_text="yaw", justify="center")
        self.tilt = FloatField(fff, low=0, mod=360, width=width, label_text="tilt", justify="center")
        
        self.camera_distance = FloatField(fff, val=None, low=0, can_disable=True, width=width, label_text="focal length", justify="center")
        
        fff = nice_frame(ff, side=tk.LEFT, anchor="n")
        def _press():
            rad_deg = 180/np.pi
            self.pitch.set(315)
            self.yaw.set(np.pi/5.1 * rad_deg)
            self.tilt.set(330)
            self.camera_distance.set(None) # disable perspective
            
        b = nice_button(fff, text="Isometric 1", side=tk.TOP, command=_press)
        
        def _press():
            rad_deg = 180/np.pi
            self.pitch.set(225)
            self.yaw.set(np.pi/5.1 * rad_deg)
            self.tilt.set(210)
            self.camera_distance.set(None) # disable perspective
            
        b = nice_button(fff, text="Isometric 2", side=tk.TOP, command=_press)

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
        if color[3]>0:
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
        if color[3]>0:
            obj = drawing.grid((0,0), self.model.epsilon)
            obj.set_color(*color)

        camera_dist = self.camera_distance.get()
        image = drawing.draw(resolution, camera_dist=camera_dist)
        return image, drawing.get_extent()#drawing.tl, drawing.br
        
    def save_png(self):
        path = os.path.join(SAVEDIR, "test.png")
        makedirs(path)
        image, _ = self.draw(self.png_resolution.get())
        PIL_image_from_array(image).save(path, optimize=True)
    
    def model_reset(self):
        self.model_prev = None
        self.model.reset()
        self.step.set(1)
        self.refresh_stepdata()
        self.refresh_viewport()

    def model_checkpoint(self):
        self.model_prev = self.model.copy()
        
    def model_recalculate(self): # recalculate current step with changes applied
        if self.model_prev is not None:
            for attr in ["_points","_normals","_prev_points","_prev_normals","_timestep","tij"]:
                self.model.copyattr(self.model_prev, attr)
            self.model_process(self.step.get()-1) # do previous step again

    def model_process(self, target_step:int):
        self.model_checkpoint()
        for model_step in self.model.process(target_step-1): # model's steps are 1 less (starts from 0)
            self.refresh_stepdata(model_step+1)

            # check if hausdorff distance is out of range
            target_step = model_step+1
            min_hdr = self.min_hausdorff.get()
            if min_hdr is not None and self.model.hausdorff_dist<min_hdr: break
            max_hdr = self.max_hausdorff.get()
            if max_hdr is not None and self.model.hausdorff_dist>max_hdr: break
            #
            
        self.refresh_viewport()
        self.step.set(target_step+1) # to display next step in the field
    
    def refresh_stepdata(self, step:int = 0):
        data = {}
        data["step"] = step
        data["hausdorff distance"] = readable_float(self.model.hausdorff_dist, 6)
        
        text = ""
        for k,v in data.items():
            if len(text)>0: text += "\n"
            text += f"{k}: {v}"
        
        self.step_data_label.configure(text=text)
        
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
        self._logbox = nice_textbox(f, 36, 12)

        
        
        
    def log(self, string):
        write_to_textbox(self._logbox, string)
    def log_error(self, string):
        write_to_textbox(self._logbox, f"ERROR: {string}")

    def start_model_instance(self):
        key = "model_"+str(self.instances_created)
        def on_destroy():
            try:
                del self.model_instances[key]
                self.log(f"destroyed {key}")
            except: pass
        
        normals_model = self.model_base.copy()
        normals_model.printing = False
        def _print(string):
            self.log(f"{key} "+string)
        normals_model.print_func = _print
        
        instance = ModelInstance(on_destroy, model=normals_model, key=key)
        self.model_instances[key] = instance
        self.instances_created += 1
        self.log(f"created {key}")
        
        
    def start(self):
        self.window.mainloop()


if __name__ == "__main__":
    InterfaceMain().start()
