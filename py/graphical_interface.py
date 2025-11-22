from _imports import *
from __gui import *
import PIL

from normals_model import ModelConfiguration as NormalsModel

SAVEDIR = os.path.join(WORKDIR, "saves")

DEFAULT_PADDING_WIDTH = 8
DEFAULT_PADDING_HEIGHT = 8
DEFAULT_TEXTFIELD_WIDTH = 14

ACTIVE_FG_COLOR = "#FFFFFF"
ACTIVE_BG_COLOR = "#000000"
PASSIVE_FG_COLOR = "#CCCCCC"
PASSIVE_BG_COLOR = "#262626"

def array_to_image(frame, array):
    return PIL.ImageTk.PhotoImage(PIL.Image.fromarray(array.swapaxes(0, 1)[::-1]), master=frame)

def read_number_from_string(string):
    if m:=re.match(RE_NUMBER, string):
        value = float(m.group(0).replace(",", "."))
        if value==int(value): value = int(value)
        return value

def set_colors(obj, fg, bg):
    if isinstance(obj, tk.Frame) or isinstance(obj, tk.Tk) or isinstance(obj, tk.Scrollbar):
        obj.config(bg=bg)
    elif isinstance(obj, tk.Canvas):
        obj.config(bg=fg, highlightbackground=bg)
    elif isinstance(obj, tk.Entry):
        obj.config(fg=fg, bg=bg, insertbackground=PASSIVE_FG_COLOR)
    else:
        obj.config(fg=fg, bg=bg)

def set_padding(obj):
    if isinstance(obj, tk.Entry):
        pass
    else:
        obj.configure(padx=DEFAULT_PADDING_WIDTH, pady=DEFAULT_PADDING_HEIGHT)

def set_active_colors(obj):
    set_colors(obj, ACTIVE_FG_COLOR, ACTIVE_BG_COLOR)
def set_active_colors_inverse(obj):
    set_colors(obj, ACTIVE_BG_COLOR, ACTIVE_FG_COLOR)
def set_passive_colors(obj):
    set_colors(obj, PASSIVE_FG_COLOR, PASSIVE_BG_COLOR)
def set_passive_colors_inverse(obj):
    set_colors(obj, PASSIVE_BG_COLOR, PASSIVE_FG_COLOR)



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
##    set_active_colors(scrollbar)
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

def nice_field(*args, side=tk.LEFT, **kwargs):
    field = tk.Entry(*args, **kwargs)
    field.pack(side=side, fill=tk.BOTH)
    set_active_colors(field)
    return field


def nice_button(frame, *args, side=tk.TOP, **kwargs):
##    subframe = nice_frame(frame, side=side)
    but = tk.Button(frame, *args, **kwargs)
    but.pack(side=tk.TOP, fill=tk.BOTH)
    but.config(fg=PASSIVE_FG_COLOR, bg=PASSIVE_BG_COLOR, activebackground=PASSIVE_BG_COLOR)
    def _event_handle0(event):
        event.widget["fg"] = ACTIVE_FG_COLOR
        event.widget["bg"] = ACTIVE_BG_COLOR
    def _event_handle1(event):
        event.widget["fg"] = PASSIVE_FG_COLOR
        event.widget["bg"] = PASSIVE_BG_COLOR
    but.bind("<Enter>", _event_handle0)
    but.bind("<Leave>", _event_handle1)
    return but

def nice_labeled_field(frame, text, width=DEFAULT_TEXTFIELD_WIDTH, update_handler=None):
    subframe = nice_frame(frame, side=tk.TOP)
    if text:
        label = nice_label(subframe, text=text, width=len(text), anchor="w") # 
    field = nice_field(subframe, width=width)
    if update_handler is not None:
        def _event_handle(event): update_handler(text, field.get())
        field.bind("<KeyRelease>", _event_handle)
    return field


class ModelInstance():
    key = None
    step = 0
    model = None
    resolution = 512
    
    def __init__(self, on_destroy, **kwargs):
        for k,v in kwargs.items():
            if hasattr(self, k): setattr(self, k, v)
        
        win = nice_window(f"model_{self.key}", on_destroy=on_destroy)
        set_padding(win)
        
        left_frame = nice_frame(win, anchor="nw")
        set_padding(left_frame)

        info_label = None # tk label obj
        def _update_info_label():
            text = f"step = {self.step}"
            text += f"\nfunc = {self.model.function}"
            info_label.configure(text=text)
        
        right_frame = nice_frame(win, anchor="nw")
        info_label = nice_label(right_frame, text="", side=tk.TOP, anchor="nw", justify="left")
        _update_info_label()
        
        # buttons
        f = left_frame
        def _next():
            self.step += 1
            self.model.process(self.step)
            _update_info_label()
        b = nice_button(f, text="Next Step", command=_next)
        
        ####
        def _mouse_handler(event):
            print(event)
            pass
        canvas, self.fig, subplot = create_figure(right_frame, _mouse_handler, 512, 512)
        def _draw_with_matplotlib():
            if not self.model.can_draw(): self.model.process(self.step)
            array, tl, br = self.model.draw(self.resolution)
            array = np.flip(array.swapaxes(0, 1), axis=0)
            subplot.clear()
            subplot.imshow(array, extent=(tl[0], br[0], tl[1], br[1]))
            canvas.draw()
        b = nice_button(f, text="Matplotlib Figure", command=_draw_with_matplotlib)
        
        def _update(labeltext, string):
            value = read_number_from_string(string)
            if value is not None: self.resolution = max(int(value), 2)
        field = nice_labeled_field(f, "resolution", width=8, update_handler=_update)
        field.insert(0, str(self.resolution))
        ####
        
        f = nice_frame(left_frame, side=tk.TOP)
        set_padding(f)
        nice_label(f, text="CONFIGURATION", side=tk.TOP, anchor="n", justify="center")
        text = f"fx = {self.model.function.fx}"
        text += f"\nfy = {self.model.function.fy}"
        text += f"\nstart = {self.model.start_point}"
        nice_label(f, text=text, side=tk.TOP, anchor="nw", justify="left")
        
        def _update(labeltext, string):
            value = read_number_from_string(string)
            if value is not None and value>=0:
                self.model.epsilon = value
        field = nice_labeled_field(f, "epsilon", width=8, update_handler=_update)
        field.insert(0, str(self.model.epsilon))
    
        adjusable_parameters = self.model.function.required_constants()
        if len(adjusable_parameters)>0:
            f = nice_frame(left_frame, side=tk.TOP)
            set_padding(f)
            nice_label(f, text="PARAMETERS", side=tk.TOP, anchor="n", justify="center")
            for k in sorted(adjusable_parameters):
                def _update(labeltext, string):
                    value = read_number_from_string(string)
                    if value is not None:
                        self.model.function.constants[labeltext] = value
                        _update_info_label()
                field = nice_labeled_field(f, k, update_handler=_update)
                v = self.model.function.constants.get(k)
                if v is not None: field.insert(0, str(v))
    

class Interface():
    instances_created = 0
    
    def __init__(self):
        self.normals_model = NormalsModel()
        self.model_instances = {}
        self._init_main_window()

    def _init_main_window(self):
        self.window = nice_window("main")
        set_padding(self.window)
        
        left_frame = nice_frame(self.window, anchor="nw")
        set_padding(left_frame)
        right_frame = nice_frame(self.window, anchor="nw")
        self._logbox = nice_textbox(right_frame, 36, 12)
        
        # main window content
        b1 = nice_button(left_frame, text="New Instance", command=self.start_model_instance)
        
        f = nice_frame(left_frame, side=tk.TOP)
        set_padding(f)
        nice_label(f, text="function", side=tk.TOP, anchor="sw")
        def _update(labeltext, string): getattr(self.normals_model.function, labeltext).string = string
        field_fx = nice_labeled_field(f, "fx", update_handler=_update)
        field_fy = nice_labeled_field(f, "fy", update_handler=_update)
        field_fx.insert(0, self.normals_model.function.fx.string)
        field_fy.insert(0, self.normals_model.function.fy.string)
        
        f = nice_frame(left_frame, side=tk.TOP)
        set_padding(f)
        label = nice_label(f, text="starting point", side=tk.TOP, anchor="sw")
        def _update_x(labeltext, string):
            value = read_number_from_string(string)
            if value is not None: self.normals_model.start_point.x = value
        def _update_y(labeltext, string):
            value = read_number_from_string(string)
            if value is not None: self.normals_model.start_point.y = value
        field_start_x = nice_labeled_field(f, "x", update_handler=_update_x)
        field_start_y = nice_labeled_field(f, "y", update_handler=_update_y)
        field_start_x.insert(0, str(self.normals_model.start_point.x))
        field_start_y.insert(0, str(self.normals_model.start_point.y))
        
        
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
        instance = ModelInstance(on_destroy, model=self.normals_model.copy(), key=key)
        self.model_instances[key] = instance
        self.instances_created += 1
        self.log(f"created instance {key}")
        
        
    def start(self):
        self.window.mainloop()


Interface().start()
