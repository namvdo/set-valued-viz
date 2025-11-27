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



def scroll_delta_translate(delta):
    if os.name=="nt": return delta//120 # on windows must divide by 120
    return delta

def integer_cycler(obj, mod, on_update=None, button=True, scroll=True):
    var = tk.IntVar(value=0)
    def button_handler(event):
        match event.num:
            case 1: var.set((var.get()+1)%mod)
            case 3: var.set((var.get()-1)%mod)
            case _: return
        if on_update is not None: on_update()
    def wheel_handler(event):
        var.set((var.get()+scroll_delta_translate(event.delta))%mod)
        if on_update is not None: on_update()
    if button: obj.bind("<Button>", button_handler, add="+")
    if scroll: obj.bind("<MouseWheel>", wheel_handler, add="+")
    return var

def string_cycler(obj, options, on_update=None, button=True, scroll=True):
    var = tk.StringVar(value=options[0])
    var.option_index = 0
    def button_handler(event):
        match event.num:
            case 1:
                var.option_index += 1
                var.option_index %= len(options)
                var.set(options[var.option_index])
            case 3:
                var.option_index -= 1
                var.option_index %= len(options)
                var.set(options[var.option_index])
            case _: return
        if on_update is not None: on_update()
    def wheel_handler(event):
        var.option_index += scroll_delta_translate(event.delta)
        var.option_index %= len(options)
        var.set(options[var.option_index])
        if on_update is not None: on_update()
    if button: obj.bind("<Button>", button_handler, add="+")
    if scroll: obj.bind("<MouseWheel>", wheel_handler, add="+")
    return var









def set_colors(obj, fg, bg):
    if isinstance(obj, tk.Frame) or isinstance(obj, tk.Tk) or isinstance(obj, tk.Scrollbar):
        obj.config(bg=bg)
    elif isinstance(obj, tk.ttk.Combobox):
        pass
    elif isinstance(obj, tk.OptionMenu):
        obj.config(fg=fg, bg=bg, highlightbackground=PASSIVE_BG_COLOR)
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

def nice_textbox(frame, width=80, height=20, side=tk.TOP, **kwargs):
    boxframe = nice_frame(frame, side=side, **kwargs)
    scrollbar = tk.Scrollbar(boxframe)
    box = tk.Text(boxframe, height=height, width=width, yscrollcommand=scrollbar.set)
    box.configure(state="disabled")
    box.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbar.configure(command=box.yview)
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

def nice_label(*args, side=tk.TOP, fill=tk.BOTH, **kwargs):
    label = tk.Label(*args, **kwargs)
    label.pack(side=side, fill=fill)
    set_passive_colors(label)
    return label

def nice_title(frame, text, *args, side=tk.TOP, fill=tk.BOTH, **kwargs):
    label = tk.Label(frame, text=text.title(), *args, **kwargs)
    label.pack(side=side, fill=fill)
    set_passive_colors_inverted(label)
    return label

def nice_field(*args, identifier=None, side=tk.LEFT, fill=tk.BOTH, update_handler=None, **kwargs):
    field = tk.Entry(*args, **kwargs)
    field.pack(side=side, fill=fill)
    if update_handler is not None:
        def _event_handle(event): update_handler(identifier, field.get())
        field.bind("<KeyRelease>", _event_handle)
    set_hover_colors(field, set_active_colors_inverted, set_active_colors)
    return field

def set_field_content(field, string):
    field.delete(0, tk.END)
    field.insert(0, string)

def nice_labeled_field(frame, text, width=DEFAULT_TEXTFIELD_WIDTH, side=tk.TOP, fill=tk.BOTH, update_handler=None, **kwargs):
    subframe = nice_frame(frame, side=side, **kwargs)
    if text:
        label = nice_label(subframe, text=text, side=tk.LEFT, anchor="c") # , width=len(text)
    field = nice_field(subframe, identifier=text, width=width, side=tk.LEFT, update_handler=update_handler)
    return field

def nice_button(frame, *args, side=tk.TOP, **kwargs):
    but = tk.Button(frame, *args, **kwargs)
    but.pack(side=tk.TOP, fill=tk.BOTH)
    set_hover_colors(but)
    return but

def nice_RGB_selector(frame, color, on_update=None):
    f = nice_frame(frame, anchor="e")
    set_passive_colors_inverted(f)
    
    ff = nice_frame(f, side=tk.LEFT, padx=2)
    preview = nice_label(ff, width=4, side=tk.LEFT)
    def _update_preview_color():
        set_colors(preview, opposite_color_as_hex(color), color_as_hex(color))
    _update_preview_color()

    # Red
    def _update(identifier, string):
        value = read_number_from_string(string)
        if value is not None: color[0] = int(value)%256
        else: color[0] = 0
        if on_update is not None: on_update()
        _update_preview_color()
    
    field0 = nice_field(f, width=3, update_handler=_update)
    field0.insert(0, str(color[0]))
    
    def _var0_update():
        color[0] = var0.get()
        if on_update is not None: on_update()
        _update_preview_color()
        set_field_content(field0, str(color[0]))
    var0 = integer_cycler(field0, 256, on_update=_var0_update, button=False)
    var0.set(color[0])
    #

    # Green
    def _update(identifier, string):
        value = read_number_from_string(string)
        if value is not None: color[1] = int(value)%256
        else: color[1] = 0
        if on_update is not None: on_update()
        _update_preview_color()
        
    field1 = nice_field(f, width=3, update_handler=_update)
    field1.insert(0, str(color[1]))
    
    def _var1_update():
        color[1] = var1.get()
        if on_update is not None: on_update()
        _update_preview_color()
        set_field_content(field1, str(color[1]))
    var1 = integer_cycler(field1, 256, on_update=_var1_update, button=False)
    var1.set(color[1])
    #

    # Blue
    def _update(identifier, string):
        value = read_number_from_string(string)
        if value is not None: color[2] = int(value)%256
        else: color[2] = 0
        if on_update is not None: on_update()
        _update_preview_color()
        
    field2 = nice_field(f, width=3, update_handler=_update)
    field2.insert(0, str(color[2]))
    
    def _var2_update():
        color[2] = var2.get()
        if on_update is not None: on_update()
        _update_preview_color()
        set_field_content(field2, str(color[2]))
    var2 = integer_cycler(field2, 256, on_update=_var2_update, button=False)
    var2.set(color[2])
    #
    
    return f

def nice_RGBA_selector(frame, color, on_update=None):
    f = nice_RGB_selector(frame, color, on_update)
    preview = f.slaves()[0].slaves()[0]
    
    def _preview_update():
        preview.configure(text=f"{int((color[3]/255)*100)}%")

    _preview_update()
    
    def _update(identifier, string):
        value = read_number_from_string(string)
        if value is not None: color[3] = int(value)%256
        else: color[3] = 0
        if on_update is not None: on_update()
        _preview_update()
    field = nice_field(f, width=3, update_handler=_update)
    field.insert(0, str(color[3]))
    
    def _update():
        color[3] = var.get()
        if on_update is not None: on_update()
        _preview_update()
        set_field_content(field, str(color[3]))
    var = integer_cycler(field, 256, on_update=_update, button=False)
    var.set(color[3])
    
    return color

def padded_frame(frame, *args, **kwargs):
    f = nice_frame(frame, *args, **kwargs)
    set_padding(f)
    return f

def raised_frame(frame, *args, border=1, **kwargs):
    f = nice_frame(frame, *args, **kwargs)
    set_passive_colors_inverted(f)
    return nice_frame(f, padx=border, pady=border, fill=tk.BOTH)

def padded_and_raised_frame(frame, *args, border=1, **kwargs):
    f = padded_frame(frame, *args, **kwargs)
    return raised_frame(f, border=border, fill=tk.BOTH)

def nice_titled_frame(frame, title, **kwargs):
    f = padded_and_raised_frame(frame, **kwargs)
    nice_title(f, title)
    ff = nice_frame(f, fill=tk.BOTH)
    set_padding(ff)
    return ff




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
        
        win = nice_window(self.key, on_destroy=on_destroy)
        set_padding(win)
        
        self.__init_model_control_panel(win)
        middle = nice_titled_frame(win, "figure") # , anchor="n"
        self.__init_figure_drawing_panel(win)
        
        def _mouse_handler(event):
##            print(event)
            pass
        self.canvas, self.fig, self.subplot = create_figure(middle, _mouse_handler, 512, 512)
        

    def __init_model_control_panel(self, win):
        frame = nice_frame(win, anchor="nw")

        # buttons
        f = padded_frame(frame, anchor="c", side=tk.TOP, fill=tk.BOTH)
        
        ff = nice_frame(f, anchor="c", side=tk.TOP, fill=tk.BOTH)
        def _press():
            self.step += 1
            self.model.process(self.step)
            set_field_content(step_field, str(self.step))
        b = nice_button(ff, text="Next Step", command=_press)
        
        ff = nice_frame(f, anchor="c", side=tk.TOP, fill=tk.BOTH)
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None:
                self.step = max(int(value), 0)
                
        step_field = nice_field(ff, side=tk.RIGHT, width=6, justify="center", update_handler=_update)
        step_field.insert(0, str(self.step))
        
        def _press():
            self.model.process(self.step)
            set_field_content(step_field, str(self.step))
        b = nice_button(ff, text="Go to Step", command=_press)
        
        def _press():
            self.model.reset()
            self.step = 0
            set_field_content(step_field, str(self.step))
        b = nice_button(f, text="Reset", command=_press)
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
        ff = nice_titled_frame(f, "noise", anchor="n", side=tk.TOP)
##        label = nice_label(ff, text="shape?")
        
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None: sides = abs(value)
            else: sides = 0
            self.model.update_noise_geometry(sides=sides)
        field = nice_labeled_field(ff, "sides", width=8, anchor="ne", update_handler=_update)
        
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None: rotation = (value%360)/180*np.pi
            else: rotation = 0
            self.model.update_noise_geometry(rotation=rotation)
        field = nice_labeled_field(ff, "rotation", width=8, anchor="ne", update_handler=_update)
        
        
        def _update(identifier, string):
            value = read_number_from_string(string)
            if value is not None:
                self.model.epsilon = abs(value)
        field = nice_labeled_field(ff, "epsilon", width=8, anchor="ne", update_handler=_update)
        field.insert(0, str(self.model.epsilon))
        #

        def _update():
##            print(self.model.function)
            pass
        self.__init_parameters_frame(f, on_update=_update)
        
        

    def __init_parameters_frame(self, frame, on_update=None):
        adjusable_parameters = self.model.function.required_constants()
        if len(adjusable_parameters)>0:
            f = nice_titled_frame(frame, "parameters", side=tk.TOP, anchor="n")
            for k in sorted(adjusable_parameters):
                def _update(identifier, string):
                    value = read_number_from_string(string)
                    if value is not None:
                        self.model.function.constants[identifier] = value
                        if on_update is not None: on_update()
                field = nice_labeled_field(f, k, anchor="ne", update_handler=_update)
                v = self.model.function.constants.get(k)
                if v is not None: field.insert(0, str(v))
        
    def __init_figure_drawing_panel(self, win):
        frame = nice_frame(win, anchor="nw")
        
        # buttons
        f = padded_frame(frame, anchor="c", side=tk.TOP, fill=tk.BOTH)
        ff = nice_frame(f, anchor="c", side=tk.TOP, fill=tk.BOTH)
        b = nice_button(ff, text="Matplotlib Figure", command=self.refresh_figure)
        ff = nice_frame(f, anchor="c", side=tk.TOP, fill=tk.BOTH)
        b = nice_button(ff, text="Save PNG", command=self.save_png)
        #

        self.__init_figure_settings_frame(frame)

    def __init_figure_settings_frame(self, frame, on_update=None):
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
