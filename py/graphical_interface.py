from _imports import *
from __gui import *
import PIL

from normals_model2 import ModelConfiguration as NormalsModel
from mask_model2 import ModelConfiguration as MaskModel

SAVEDIR = os.path.join(WORKDIR, "saves")

DEFAULT_PADDING_WIDTH = 8
DEFAULT_PADDING_HEIGHT = 8
DEFAULT_TEXTFIELD_WIDTH = 14

ACTIVE_FG_COLOR = "#FFFFFF"
ACTIVE_BG_COLOR = "#000000"
PASSIVE_FG_COLOR = "#CCCCCC"
PASSIVE_BG_COLOR = "#262626"

def color_as_hex(color):
    return "#"+"".join([hex(i%256)[2:].rjust(2, "0") for i in color[:3]])
def opposite_color_as_hex(color):
    return "#"+"".join([hex(255-i%256)[2:].rjust(2, "0") for i in color[:3]])
##print(color_as_hex([12, 200, 255]))

##def array_to_imagetk(frame, array):
##    return PIL.ImageTk.PhotoImage(PIL.Image.fromarray(array.swapaxes(0, 1)[::-1]), master=frame)

def read_number_from_string(string):
    if m:=re.match(RE_NUMBER, string):
        value = float(m.group(0).replace(",", "."))
        if value==int(value): value = int(value)
        return value
def read_numbers_from_string(string):
    values = []
    for value in re.findall(RE_NUMBER, string):
        value = float(value.replace(",", "."))
        if value==int(value): value = int(value)
        values.append(value)
    return values

def set_colors(obj, fg, bg):
    if isinstance(obj, tk.Frame) or isinstance(obj, tk.Tk) or isinstance(obj, tk.Scrollbar):
        obj.config(bg=bg)
    elif isinstance(obj, tk.Canvas):
        obj.config(bg=fg, highlightbackground=bg)
    elif isinstance(obj, tk.Entry):
        obj.config(fg=fg, bg=bg, insertbackground=PASSIVE_FG_COLOR)
    elif isinstance(obj, tk.Button):
        obj.config(fg=fg, bg=bg, activebackground=PASSIVE_FG_COLOR)
    else:
        obj.config(fg=fg, bg=bg)

def set_padding(obj):
    if isinstance(obj, tk.Entry):
        pass
    else:
        obj.configure(padx=DEFAULT_PADDING_WIDTH, pady=DEFAULT_PADDING_HEIGHT)

def set_active_colors(obj):
    set_colors(obj, ACTIVE_FG_COLOR, ACTIVE_BG_COLOR)
def set_active_colors_inverted(obj):
    set_colors(obj, ACTIVE_BG_COLOR, ACTIVE_FG_COLOR)
def set_passive_colors(obj):
    set_colors(obj, PASSIVE_FG_COLOR, PASSIVE_BG_COLOR)
def set_passive_colors_inverted(obj):
    set_colors(obj, PASSIVE_BG_COLOR, PASSIVE_FG_COLOR)

def set_hover_colors(obj, on_enter=set_active_colors, on_leave=set_passive_colors):
    def _enter(event): on_enter(event.widget)
    def _leave(event): on_leave(event.widget)
    obj.bind("<Enter>", _enter)
    obj.bind("<Leave>", _leave)
    on_leave(obj)

def nice_window(title, configure_handler=None, on_destroy=None):
    win = tk.Tk()
    win.wm_title(title)
    if configure_handler is not None:
        win.bind("<Configure>", configure_handler)
    if on_destroy is not None:
        def _destroy():
            on_destroy()
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _destroy)
    win.resizable(False, False)
    set_passive_colors(win)
    return win

def nice_frame(root, *args, side=tk.LEFT, **kwargs):
    frame = tk.Frame(root)
    frame.pack(*args, side=side, **kwargs)
    set_passive_colors(frame)
    return frame

def nice_textbox(frame, width=80, height=20):
    boxframe = create_frame(frame, tk.TOP)
    scrollbar = tk.Scrollbar(boxframe)
    box = tk.Text(boxframe, height=height, width=width, yscrollcommand=scrollbar.set)
    box.configure(state="disabled")
    box.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbar.configure(command=box.yview)
    set_passive_colors(boxframe)
    set_active_colors(box)
    return box

def nice_canvas(frame, width, height, side=tk.LEFT, fill=tk.BOTH):
    canvas = tk.Canvas(frame, width=width, height=height)
    canvas.pack(side=side, fill=fill)
    set_active_colors(canvas)
    return canvas

def nice_separator(frame, padx=DEFAULT_PADDING_WIDTH, pady=DEFAULT_PADDING_HEIGHT, vertical=False):
    separator = Separator(frame, orient="vertical" if vertical else "horizontal")
    separator.pack(side=tk.LEFT if vertical else tk.TOP, fill=tk.BOTH, padx=padx, pady=pady)
    return separator

def nice_label(*args, side=tk.LEFT, fill=tk.BOTH, **kwargs):
    label = tk.Label(*args, **kwargs)
    label.pack(side=side, fill=fill)
    set_passive_colors(label)
    return label

def nice_field(*args, identifier=None, side=tk.LEFT, update_handler=None, **kwargs):
    field = tk.Entry(*args, **kwargs)
    field.pack(side=side, fill=tk.BOTH)
    if update_handler is not None:
        def _event_handle(event): update_handler(identifier, field.get())
        field.bind("<KeyRelease>", _event_handle)
    set_hover_colors(field, set_active_colors_inverted, set_active_colors)
    return field


def nice_button(frame, *args, side=tk.TOP, **kwargs):
    but = tk.Button(frame, *args, **kwargs)
    but.pack(side=tk.TOP, fill=tk.BOTH)
    set_hover_colors(but)
    return but
    

def nice_labeled_field(frame, text, width=DEFAULT_TEXTFIELD_WIDTH, side=tk.TOP, update_handler=None):
    subframe = nice_frame(frame, side=side)
    if text:
        label = nice_label(subframe, text=text, width=len(text), anchor="w")
    field = nice_field(subframe, identifier=text, width=width, update_handler=update_handler)
    return field

def nice_RGB_selector(frame, color, on_update=None):
    f = nice_frame(frame, anchor="e")
    set_passive_colors_inverted(f)
    
    ff = nice_frame(f, side=tk.LEFT, padx=2)
    preview = nice_label(ff, width=4)
    def _update_preview_color():
        set_colors(preview, opposite_color_as_hex(color), color_as_hex(color))
    _update_preview_color()
    
    def _update(identifier, string):
        value = read_number_from_string(string)
        if value is not None: color[0] = int(value)%256
        else: color[0] = 0
        if on_update is not None: on_update()
        _update_preview_color()
    
    field = nice_field(f, width=3, update_handler=_update)
    field.insert(0, str(color[0]))
    
    def _update(identifier, string):
        value = read_number_from_string(string)
        if value is not None: color[1] = int(value)%256
        else: color[1] = 0
        if on_update is not None: on_update()
        _update_preview_color()
        
    field = nice_field(f, width=3, update_handler=_update)
    field.insert(0, str(color[1]))
    
    def _update(identifier, string):
        value = read_number_from_string(string)
        if value is not None: color[2] = int(value)%256
        else: color[2] = 0
        if on_update is not None: on_update()
        _update_preview_color()
        
    field = nice_field(f, width=3, update_handler=_update)
    field.insert(0, str(color[2]))
    
    return f

def nice_RGBA_selector(frame, color, on_update=None):
    f = nice_RGB_selector(frame, color, on_update)
    preview = f.slaves()[0].slaves()[0]
    preview.configure(text=f"{int((color[3]/255)*100)}%")
    def _update(identifier, string):
        value = read_number_from_string(string)
        if value is not None: color[3] = int(value)%256
        else: color[3] = 0
        if on_update is not None: on_update()
        preview.configure(text=f"{int((color[3]/255)*100)}%")
    field = nice_field(f, width=3, update_handler=_update)
    field.insert(0, str(color[3]))
    
    return color


##class HybridModel(ModelBase):
##    step = None
##    def __init__(self, model_base):
##        self.copy_attributes_from(model_base)
##        
##        self.normals = NormalsModel()
##        self.mask = MaskModel()
##
##        # share start_points
##        self.normals.start_point = self.start_point
##        self.mask.start_point = self.start_point
##        
##        # share functions
##        self.normals.function = self.function
##        self.mask.function = self.function

class ModelInstance():
    key = None
    model = None
    step = 0
    
    resolution = 512

    min_x = max_x = None
    min_y = max_y = None
    
    canvas = None
    fig = None
    subplot = None
    
    def __init__(self, on_destroy, **kwargs):
        for k,v in kwargs.items():
            if hasattr(self, k): setattr(self, k, v)
        
        win = nice_window(f"model_{self.key}", on_destroy=on_destroy)
        set_padding(win)
        
        self.__init_model_control_panel(win)
        middle = nice_frame(win) # , anchor="n"
        self.__init_figure_drawing_panel(win)
        
        def _mouse_handler(event):
##            print(event)
            pass
        self.canvas, self.fig, self.subplot = create_figure(middle, _mouse_handler, 512, 512)
        

    def __init_model_control_panel(self, win):
        frame = nice_frame(win, anchor="nw")
        
        nice_label(frame, text="configuration", side=tk.TOP, anchor="c", justify="center")
        f = nice_frame(frame, anchor="nw", side=tk.TOP)
        set_padding(f)
        text = f"fx = {self.model.function.fx}"
        text += f"\nfy = {self.model.function.fy}"
        text += f"\nstart = {self.model.start_point}"
        nice_label(f, text=text, side=tk.TOP, anchor="nw", justify="left")

        def _update():
##            print(self.model.function)
            pass
        self.__init_parameters_frame(frame, on_update=_update)
        
        f = nice_frame(frame, anchor="nw", side=tk.TOP)
        set_padding(f)
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None:
                self.model.epsilon = abs(value)
        field = nice_labeled_field(f, "epsilon", width=8, update_handler=_update)
        field.insert(0, str(self.model.epsilon))
        
        # buttons
        f = nice_frame(frame, anchor="c", side=tk.TOP, fill=tk.BOTH)
        set_padding(f)
        #
        ff = nice_frame(f, anchor="c", side=tk.TOP, fill=tk.BOTH)
        def _press():
            self.step += 1
            self.model.process(self.step)
            field.delete(0, tk.END)
            field.insert(0, str(self.step))
        b = nice_button(ff, text="Next Step", command=_press)
        #
        ff = nice_frame(f, anchor="c", side=tk.TOP, fill=tk.BOTH)
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None:
                self.step = max(int(value), 0)
        field = nice_field(ff, side=tk.RIGHT, width=6, update_handler=_update)
        field.insert(0, str(self.step))
        def _press():
            self.model.process(self.step)
            field.delete(0, tk.END)
            field.insert(0, str(self.step))
        b = nice_button(ff, text="Go to Step", command=_press)
        #
        
    def __init_figure_drawing_panel(self, win):
        frame = nice_frame(win, anchor="nw")
        #
        
        nice_label(frame, text="extend figure limits", anchor="c", side=tk.TOP)
        f = nice_frame(frame, anchor="c", side=tk.TOP)
        set_padding(f)
        
        ff = nice_frame(f, anchor="ne", side=tk.TOP)
        nice_label(ff, text="horizontal", side=tk.LEFT)
        
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None: self.min_x = value
            else: self.min_x = None
        field = nice_field(ff, side=tk.LEFT, width=5, update_handler=_update)
        
        nice_label(ff, text=" ... ", side=tk.LEFT)
        
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None: self.max_x = value
            else: self.max_x = None
        field = nice_field(ff, side=tk.LEFT, width=5, update_handler=_update)
        
        ff = nice_frame(f, anchor="ne", side=tk.TOP)
        nice_label(ff, text="vertical", side=tk.LEFT)
        
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None: self.min_y = value
            else: self.min_y = None
        field = nice_field(ff, side=tk.LEFT, width=5, update_handler=_update)
        
        nice_label(ff, text=" ... ", side=tk.LEFT)
        
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None: self.max_y = value
            else: self.max_y = None
        field = nice_field(ff, side=tk.LEFT, width=5, update_handler=_update)
        
        
        f = nice_frame(frame, anchor="nw", side=tk.TOP)
        set_padding(f)
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None: self.resolution = max(int(value), 2)
        field = nice_labeled_field(f, "image resolution", width=8, update_handler=_update)
        field.insert(0, str(self.resolution))
        
        # buttons
        f = nice_frame(frame, anchor="c", side=tk.TOP, fill=tk.BOTH)
        set_padding(f)
        ff = nice_frame(f, anchor="c", side=tk.TOP, fill=tk.BOTH)
        b = nice_button(ff, text="Matplotlib Figure", command=self.refresh_figure)
        ff = nice_frame(f, anchor="c", side=tk.TOP, fill=tk.BOTH)
        b = nice_button(ff, text="Save PNG", command=self.save_png)
        #

        self.__init_color_settings_frame(frame)

    def __init_parameters_frame(self, frame, on_update=None):
        adjusable_parameters = self.model.function.required_constants()
        if len(adjusable_parameters)>0:
            nice_label(frame, text="parameters", side=tk.TOP, anchor="c", justify="center")
            f = nice_frame(frame, side=tk.TOP, anchor="ne")
            set_padding(f)
            for k in sorted(adjusable_parameters):
                def _update(identifier, string):
                    value = read_number_from_string(string)
                    if value is not None:
                        self.model.function.constants[identifier] = value
                        if on_update is not None: on_update()
                field = nice_labeled_field(f, k, update_handler=_update)
                v = self.model.function.constants.get(k)
                if v is not None: field.insert(0, str(v))

    def __init_color_settings_frame(self, frame, on_update=None):
        nice_label(frame, text="figure RGBA colors", side=tk.TOP, anchor="c", justify="center")
        f = nice_frame(frame, side=tk.TOP, anchor="ne")
        set_padding(f)
        
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
            ff = nice_frame(f, side=tk.TOP, anchor="ne")
##            set_padding(ff)
            nice_label(ff, text=name, anchor="w", side=tk.LEFT) # , width=longest_name_len
            nice_RGBA_selector(ff, color)

    def draw(self):
        if not self.model.can_draw():
            self.model.process(self.step)
        
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
        if color[3]>0:
##            dist = np.linalg.norm(drawing.br-drawing.tl)/10
            drawing.grid((0,0), self.model.epsilon, *color)
        
        image = drawing.draw(self.resolution)
        return np.flip(image.swapaxes(0, 1), axis=0), drawing.tl, drawing.br
        
    def save_png(self):
        path = os.path.join(SAVEDIR, "test.png")
        makedirs(path)
        image, _, _ = self.draw()
        PIL.Image.fromarray(image).save(path, optimize=True)
        
    def refresh_figure(self):
        image, tl, br = self.draw()
        
        self.subplot.clear()
        self.subplot.imshow(image, extent=(tl[0], br[0], tl[1], br[1]))
        self.subplot.set_title(f"step: {self.step}, shape: {image.shape}")
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
        
        leftside = nice_frame(self.window, anchor="nw")
        set_padding(leftside)
        rightside = nice_frame(self.window, anchor="nw")
        self._logbox = nice_textbox(rightside, 36, 12)
        
        # main window content
        b1 = nice_button(leftside, text="New Instance", command=self.start_model_instance)
        
        f = nice_frame(leftside, side=tk.TOP)
        set_padding(f)
        nice_label(f, text="function", side=tk.TOP, anchor="sw")
        def _update(identifier, string): getattr(self.model_base.function, identifier).string = string
        field_fx = nice_labeled_field(f, "fx", update_handler=_update)
        field_fy = nice_labeled_field(f, "fy", update_handler=_update)
        field_fx.insert(0, self.model_base.function.fx.string)
        field_fy.insert(0, self.model_base.function.fy.string)
        
        f = nice_frame(leftside, side=tk.TOP)
        set_padding(f)
        label = nice_label(f, text="starting point", side=tk.TOP, anchor="sw")
        def _update_x(identifier, string):
            value = read_number_from_string(string)
            if value is not None: self.model_base.start_point.x = value
        def _update_y(identifier, string):
            value = read_number_from_string(string)
            if value is not None: self.model_base.start_point.y = value
        field_start_x = nice_labeled_field(f, "x", update_handler=_update_x)
        field_start_y = nice_labeled_field(f, "y", update_handler=_update_y)
        field_start_x.insert(0, str(self.model_base.start_point.x))
        field_start_y.insert(0, str(self.model_base.start_point.y))
        
        
    def log(self, string):
        write_to_textbox(self._logbox, string)
    def log_error(self, string):
        write_to_textbox(self._logbox, f"ERROR: {string}")

    def start_model_instance(self):
        key = self.instances_created
        def on_destroy():
            try:
                del self.model_instances[key]
                self.log(f"destroyed instance {key}")
            except: pass
        
        normals_model = NormalsModel()
        normals_model.copy_attributes_from(self.model_base)
        
        instance = ModelInstance(on_destroy, model=normals_model, key=key)
        self.model_instances[key] = instance
        self.instances_created += 1
        self.log(f"created instance {key}")
        
        
    def start(self):
        self.window.mainloop()


Interface().start()
