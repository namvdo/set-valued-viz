from _imports import *
from __gui import *

from normals_model2 import ModelConfiguration as NormalsModel
from mask_model2 import ModelConfiguration as MaskModel

SAVEDIR = os.path.join(WORKDIR, "saves")


class ModelInstance():
    key = None
    model = None
    step = -1
    
    resolution = 512

    min_x = max_x = None
    min_y = max_y = None
    
    canvas = None
    fig = None
    subplot = None
    
    def __init__(self, on_destroy, **kwargs):
        for k,v in kwargs.items():
            if hasattr(self, k): setattr(self, k, v)
        win = nice_window(self.key, on_destroy=on_destroy)
        set_padding(win)
        self._model_control_panel(win)
        self._figure_viewport_panel(win)
        self._figure_drawing_panel(win)
        

    def _model_control_panel(self, win):
        frame = nice_frame(win, anchor="nw")
        
        # buttons
        f = padded_frame(frame, anchor="c", side=tk.TOP, fill=tk.BOTH)
        
        ff = nice_frame(f, anchor="c", side=tk.TOP, fill=tk.BOTH)
        def _press():
            self.step += 1
            self.model.process(self.step)
            set_field_content(step_field, str(self.step))
            self.refresh_figure()
        b = nice_button(ff, text="Next Step", command=_press)
        
        ff = nice_frame(f, anchor="c", side=tk.TOP, fill=tk.BOTH)
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None:
                self.step = max(int(value), -1)
                
        step_field = nice_field(ff, side=tk.RIGHT, width=6, justify="center", update_handler=_update)
        step_field.insert(0, str(self.step))
        
        def _press():
            self.model.process(self.step)
            set_field_content(step_field, str(self.step))
            self.refresh_figure()
        b = nice_button(ff, text="Go to Step", command=_press)
        
        def _press():
            self.model.reset()
            self.step = -1
            set_field_content(step_field, str(self.step))
            self.refresh_figure()
        b = nice_button(f, text="Reset", command=_press)
        #

        #
        f = nice_titled_frame(frame, "model configuration", side=tk.TOP)
        ff = padded_frame(f, anchor="nw", side=tk.TOP)
        text = f"fx = {self.model.function.fx}"
        text += f"\nfy = {self.model.function.fy}"
        text += f"\nstart = {self.model.start_point}"
        label = nice_label(ff, text=text, anchor="nw", justify="left")

##        def _update():
##            print(var.get())
##        var = string_cycler(label, list("abcdef"), on_update=_update)
##        
##        def _update():
##            print(var1.get())
##        var1 = integer_cycler(label, 256, on_update=_update)
        
        #
        width = 6
        ff = nice_titled_frame(f, "noise", anchor="n", side=tk.TOP)
        
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None: sides = abs(value)
            else: sides = 0
            self.model.update_noise_geometry(sides=sides)
        field = nice_labeled_field(ff, "vertices", width=width, anchor="ne", update_handler=_update)
        
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None: rotation = (value%360)/180*np.pi
            else: rotation = 0
            self.model.update_noise_geometry(rotation=rotation)
        field = nice_labeled_field(ff, "rotation", width=width, anchor="ne", update_handler=_update)
        
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None:
                self.model.epsilon = abs(value)
        field = nice_labeled_field(ff, "distance", width=width, anchor="ne", update_handler=_update)
        field.insert(0, str(self.model.epsilon))
        #

        #
        def on_update():
            pass
        width = 10
        adjusable_parameters = self.model.function.required_constants()
        if len(adjusable_parameters)>0:
            ff = nice_titled_frame(f, "parameters", side=tk.TOP, anchor="n")
            for k in sorted(adjusable_parameters):
                def _update(identifier, string):
                    value = read_number_from_string(string)
                    if value is not None:
                        self.model.function.constants[identifier] = value
                        if on_update is not None: on_update()
                field = nice_labeled_field(ff, k, width=width, anchor="ne", update_handler=_update)
                v = self.model.function.constants.get(k)
                if v is not None: field.insert(0, str(v))
        #

    def _figure_viewport_panel(self, win):
        middle = nice_titled_frame(win, "figure") # , anchor="n"
        def _mouse_handler(event):
##            print(event)
            pass
        self.canvas, self.fig, self.subplot = create_figure(middle, _mouse_handler, 512, 512)
        
        
    def _figure_drawing_panel(self, win):
        frame = nice_frame(win, anchor="nw")
        
        # buttons
        f = padded_frame(frame, anchor="c", side=tk.TOP, fill=tk.BOTH)
        ff = nice_frame(f, anchor="c", side=tk.TOP, fill=tk.BOTH)
        b = nice_button(ff, text="Refresh", command=self.refresh_figure) # Matplotlib Figure
        ff = nice_frame(f, anchor="c", side=tk.TOP, fill=tk.BOTH)
        b = nice_button(ff, text="Save PNG", command=self.save_png)
        #

        #
        f = nice_titled_frame(frame, "figure settings")
        ff = nice_frame(f, side=tk.TOP)
        fff = nice_titled_frame(ff, "resolution", anchor="nw", side=tk.LEFT)
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None: self.resolution = max(int(value), 2)
        field = nice_field(fff, width=7, update_handler=_update)
        field.insert(0, str(self.resolution))
        
        #
        fff = nice_titled_frame(ff, "extend limits", anchor="nw", side=tk.LEFT)
        
        ffff = nice_frame(fff, anchor="c", side=tk.TOP)
        
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None: self.min_x = value
            else: self.min_x = None
        field = nice_field(ffff, side=tk.LEFT, width=5, update_handler=_update)
        
        nice_label(ffff, text="<= x <=", side=tk.LEFT)
        
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None: self.max_x = value
            else: self.max_x = None
        field = nice_field(ffff, side=tk.LEFT, width=5, update_handler=_update)
        
        ffff = nice_frame(fff, anchor="c", side=tk.TOP)
        
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None: self.min_y = value
            else: self.min_y = None
        field = nice_field(ffff, side=tk.LEFT, width=5, update_handler=_update)
        
        nice_label(ffff, text="<= y <=", side=tk.LEFT)
        
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None: self.max_y = value
            else: self.max_y = None
        field = nice_field(ffff, side=tk.LEFT, width=5, update_handler=_update)
        #

        #
        ff = nice_titled_frame(f, "colors", side=tk.TOP, anchor="ne")
        
        self.colors = {
            "background": [255,255,255,255],
            "grid": [0,127,127,63],
            "points": [255,0,0,255],
            "boundary": [0,255,0,255],
            "normals": [0,0,0,127],
            "prev_points": [0,0,255,0],
            }
##        longest_name_len = len(max(self.colors.keys(), key=len))
        for name,color in self.colors.items():
            fff = nice_frame(ff, side=tk.TOP, anchor="ne")
            nice_label(fff, text=name, anchor="w", side=tk.LEFT) # , width=longest_name_len
            nice_RGBA_selector(fff, color)
        #

    def draw(self):
        if not self.model.can_draw(): self.model.process(self.step)
        
        color = np.divide(self.colors["background"], 255)
        drawing = ImageDrawing(*color)
        
        color = np.divide(self.colors["boundary"], 255)
        if color[3]>0: drawing.lines(*self.model.get_boundary_lines(), *color)
        
        color = np.divide(self.colors["normals"], 255)
        if color[3]>0: drawing.lines(*self.model.get_inner_normals(), *color)
        
        color = np.divide(self.colors["prev_points"], 255)
        if color[3]>0: drawing.points(self.model._prev_points, *color)
        
        color = np.divide(self.colors["points"], 255)
        if color[3]>0: drawing.points(self.model._points, *color)
        
        if self.min_x is not None: drawing.tl[0] = min(drawing.tl[0], self.min_x)
        if self.max_x is not None: drawing.br[0] = max(drawing.br[0], self.max_x)
        if self.min_y is not None: drawing.tl[1] = min(drawing.tl[1], self.min_y)
        if self.max_y is not None: drawing.br[1] = max(drawing.br[1], self.max_y)
        
        color = np.divide(self.colors["grid"], 255)
        if color[3]>0: drawing.grid((0,0), self.model.epsilon, *color)
        
        image = drawing.draw(self.resolution)
        return image, drawing.tl, drawing.br
        
    def save_png(self):
        path = os.path.join(SAVEDIR, "test.png")
        makedirs(path)
        image, _, _ = self.draw()
        PIL_image_from_array(image).save(path, optimize=True)
        
    def refresh_figure(self):
        self.subplot.clear()
        if self.step>=0:
            image, tl, br = self.draw()
            self.subplot.imshow(image, extent=(tl[0], br[0], tl[1], br[1]))
            title = f"step: {self.step}, image: {image.shape}"
            title += f"\n{len(self.model)} points"
            self.subplot.set_title(title)
        self.canvas.draw()

class Interface():
    instances_created = 0
    
    def __init__(self):
        self.model_base = ModelBase()
        self.model_instances = {}
        self._init_main_window()

    def _init_main_window(self):
        self.window = nice_window("main")
        set_padding(self.window)
        
        leftside = nice_frame(self.window, side=tk.LEFT, fill=tk.BOTH)
        rightside = nice_titled_frame(self.window, "log", side=tk.LEFT)
        self._logbox = nice_textbox(rightside, 36, 12)
        
        # main window content
        f = padded_frame(leftside, side=tk.TOP, fill=tk.BOTH)
        b = nice_button(f, text="New Instance", side=tk.TOP, command=self.start_model_instance)
        
        ff = nice_titled_frame(leftside, "function", side=tk.TOP, fill=tk.BOTH, anchor="ne")
        def _update(identifier, string):
            getattr(self.model_base.function, identifier).string = string
        field_fx = nice_labeled_field(ff, "fx", update_handler=_update)
        field_fy = nice_labeled_field(ff, "fy", update_handler=_update)
        field_fx.insert(0, self.model_base.function.fx.string)
        field_fy.insert(0, self.model_base.function.fy.string)
        
        ff = nice_titled_frame(leftside, "starting point", side=tk.TOP, anchor="ne")
        def _update_x(identifier, string):
            value = read_number_from_string(string)
            if value is not None: self.model_base.start_point.x = value
        def _update_y(identifier, string):
            value = read_number_from_string(string)
            if value is not None: self.model_base.start_point.y = value
        field_start_x = nice_labeled_field(ff, "x", width=8, update_handler=_update_x)
        field_start_y = nice_labeled_field(ff, "y", width=8, update_handler=_update_y)
        field_start_x.insert(0, str(self.model_base.start_point.x))
        field_start_y.insert(0, str(self.model_base.start_point.y))
        
        
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
        
        normals_model = NormalsModel()
        normals_model.printing = False
        def _print(string):
            self.log(f"{key} "+string)
        normals_model.print_func = _print
        normals_model.copy_attributes_from(self.model_base)
        
        instance = ModelInstance(on_destroy, model=normals_model, key=key)
        self.model_instances[key] = instance
        self.instances_created += 1
        self.log(f"created {key}")
        
        
    def start(self):
        self.window.mainloop()


Interface().start()
