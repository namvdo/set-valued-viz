from _imports import *
from __gui import *
import PIL

from normals_model import ModelConfiguration as NormalsModel

save_directory = os.path.join(WORKDIR, "saves")

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
    if isinstance(obj, tk.Frame) or isinstance(obj, tk.Tk) or isinstance(obj, tk.Canvas):
        obj.config(bg=bg)
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
    set_padding(win)
    return win

def nice_frame(root, *args, side=tk.LEFT, **kwargs):
    frame = tk.Frame(root)
    frame.pack(*args, side=side, **kwargs)
    set_passive_colors(frame)
    set_padding(frame)
    return frame

def nice_canvas(frame, width, height):
    canvas = tk.Canvas(frame, width=width, height=height)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH)
    set_active_colors_inverse(canvas)
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



class Interface():
    instances_created = 0
    
    class SubInstance():
        key = None
        step = 0
        model = None
        image = None
        resolution = 512
        
    def __init__(self):
        self.normals_model = NormalsModel()
        self.subinstances = {}
        self._init_main_window()

    def _init_main_window(self):
        self.window = nice_window("main")
        left_frame = nice_frame(self.window)
        right_frame = nice_frame(self.window)
        self._logbox = create_textbox(right_frame, 48, 12)
        set_active_colors(self._logbox)
        
        # main window content
        b1 = nice_button(left_frame, text="Start", command=self.start_model_instance)
        
        
        nice_label(left_frame, text="\nfunction", side=tk.TOP, anchor="sw")
        def _update(labeltext, string): getattr(self.normals_model.function, labeltext).string = string
        field_fx = nice_labeled_field(left_frame, "fx", update_handler=_update)
        field_fy = nice_labeled_field(left_frame, "fy", update_handler=_update)
        field_fx.insert(0, self.normals_model.function.fx.string)
        field_fy.insert(0, self.normals_model.function.fy.string)
        
        
        label = nice_label(left_frame, text="\nstarting point", side=tk.TOP, anchor="sw")
        def _update_x(labeltext, string):
            value = read_number_from_string(string)
            if value is not None: self.normals_model.start_point.x = value
        def _update_y(labeltext, string):
            value = read_number_from_string(string)
            if value is not None: self.normals_model.start_point.y = value
        field_start_x = nice_labeled_field(left_frame, "x", update_handler=_update_x)
        field_start_y = nice_labeled_field(left_frame, "y", update_handler=_update_y)
        field_start_x.insert(0, str(self.normals_model.start_point.x))
        field_start_y.insert(0, str(self.normals_model.start_point.y))
        
        
    def log(self, string):
        write_to_textbox(self._logbox, string)
    def log_error(self, string):
        write_to_textbox(self._logbox, f"ERROR: {string}")

    def start_model_instance(self):
        instance = self.SubInstance()
        instance.model = self.normals_model.copy()
        
        instance.key = self.instances_created
        self.instances_created += 1
        self.subinstances[instance.key] = instance

        log_id = f"MODEL_{instance.key}"
        def _destroy():
            del self.subinstances[instance.key]
        win = nice_window(log_id, on_destroy=_destroy)
        
        left_frame = nice_frame(win)
        right_frame = nice_frame(win)
        canvas = nice_canvas(right_frame, 0, 0)
        
        nice_label(left_frame, text="CONFIGURATION", side=tk.TOP, anchor="n", justify="center")
        nice_label(left_frame, text=f"fx = {instance.model.function.fx}", side=tk.TOP, anchor="nw", justify="left")
        nice_label(left_frame, text=f"fy = {instance.model.function.fy}", side=tk.TOP, anchor="nw", justify="left")
        nice_label(left_frame, text=f"s. point = {instance.model.start_point}", side=tk.TOP, anchor="nw", justify="left")
        
        # parameters
        adjusable_parameters = instance.model.function.required_constants()
        nice_label(left_frame, text="\nPARAMETERS", side=tk.TOP, anchor="n", justify="center")
        for k in sorted(adjusable_parameters):
            def _update(labeltext, string):
                value = read_number_from_string(string)
                if value is not None:
                    instance.model.function.constants[labeltext] = value
##                    label.configure(text=str(instance.model.function))
            field = nice_labeled_field(left_frame, k, update_handler=_update)
            v = instance.model.function.constants.get(k)
            if v is not None: field.insert(0, str(v))
        
        def _update(labeltext, string):
            value = read_number_from_string(string)
            if value is not None:
                instance.resolution = max(int(value), 2)
##        nice_label(left_frame, text="\n", side=tk.TOP, anchor="sw", justify="left")
        field = nice_labeled_field(left_frame, "resolution", width=6, update_handler=_update)
        field.insert(0, str(instance.resolution))
        
        def _draw():
            if not instance.model.can_draw(): instance.model.process(instance.step)
            array, tl, br = instance.model.draw(instance.resolution)
            instance.image = array_to_image(right_frame, array)
            shape = (instance.image.width(), instance.image.height())
            canvas.configure(width=shape[0], height=shape[1])
            canvas.create_image(2,2, anchor="nw", image=instance.image)
            self.log(f"{log_id} draw: {shape}")
        b = nice_button(left_frame, text="Draw", command=_draw)
        
        def _next():
            instance.step += 1
            instance.model.process(instance.step)
            self.log(f"{log_id} step: {instance.step}")
        b = nice_button(left_frame, text="Next Step", command=_next)
        
    def start(self):
        self.window.mainloop()


Interface().start()
