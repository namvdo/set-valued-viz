import os, re
import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np

from PIL.Image import fromarray as PIL_image_from_array

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
matplotlib.use("TkAgg")

from __equation import RE_NUMBER

re_number_reader = re.compile(RE_NUMBER)


DEFAULT_PADDING_WIDTH = 8
DEFAULT_PADDING_HEIGHT = 8
DEFAULT_TEXTFIELD_WIDTH = 14

ACTIVE_FG_COLOR = "#FFFFFF"
ACTIVE_BG_COLOR = "#000000"
PASSIVE_FG_COLOR = "#CCCCCC"
PASSIVE_BG_COLOR = "#262626"


def read_number_from_string(string):
    if m:=re_number_reader.match(string):
        value = float(m.group(0).replace(",", "."))
        if value==int(value): value = int(value)
        return value

def readable_float(value:float, rounding=6):
    if value==int(value): return str(int(value))
    rounded_value = round(value, rounding)
    return "~"*int(rounded_value!=value)+str(rounded_value)

def color_as_hex(color):
    return "#"+"".join([hex(i%256)[2:].rjust(2, "0") for i in color[:3]])
def opposite_color_as_hex(color):
    return "#"+"".join([hex(255-i%256)[2:].rjust(2, "0") for i in color[:3]])


def set_colors(obj, fg, bg):
    if isinstance(obj, tk.Frame) or isinstance(obj, tk.Tk) or isinstance(obj, tk.Scrollbar):
        obj.config(bg=bg)
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

def set_padding(obj, x=DEFAULT_PADDING_WIDTH, y=DEFAULT_PADDING_HEIGHT):
    if isinstance(obj, tk.Entry): pass
    else: obj.configure(padx=x, pady=y)
    
def set_random_colors(obj): # for debugging
    foreground = color_as_hex(np.random.randint(0, 255, 3))
    background = color_as_hex(np.random.randint(0, 255, 3))
    set_colors(obj, foreground, background)

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













##def array_to_imagetk(frame, array):
##    return PIL.ImageTk.PhotoImage(PIL.Image.fromarray(array.swapaxes(0, 1)[::-1]), master=frame)

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








class IntVar(tk.IntVar):
    can_disable = True
    disabled = False # True -> returns None with .get()
    def __init__(self, value, can_disable=True, **kwargs):
        self.can_disable = can_disable
        super().__init__(value=0, **kwargs)
        self.set(value)
    def get(self):
        if self.disabled: return None
        return super().get()
    def set(self, value):
        if value is None:
            if self.can_disable: self.disabled = True
        else:
            if self.disabled: self.disabled = False
            super().set(value)

def float_precision2(f:float, maximum:int=12):
    c = 0
    while not f.is_integer():
        f = round(f*10, maximum)
        c += 1
    return c

class FloatVar(IntVar):
    precision = 0
    def get(self):
        val = super().get()
        if val is not None:
            f = float(val)
            f /= 10**self.precision
            return f
        
    def set(self, f, set_precision=True):
        if f is None:
            if self.can_disable: self.disabled = True
        else:
            if set_precision: self.precision = float_precision2(f)
            return super().set(round(f*10**self.precision))


def integer_selector(obj, start=0, low=None, high=None, mod=None, step=1, on_update=None, button=True, scroll=True, can_disable=False):
    var = IntVar(value=start, can_disable=can_disable)
    def button_handler(event):
        if var.disabled: return
        match event.num:
            case 1:
                val = var.get()+step
                if mod is not None: val %= mod
                if high is not None: val = min(val, high)
                if low is not None: val = max(val, low)
                var.set(val)
            case 3:
                val = var.get()-step
                if mod is not None: val %= mod
                if high is not None: val = min(val, high)
                if low is not None: val = max(val, low)
                var.set(val)
            case _: return
        if on_update is not None: on_update()
    def wheel_handler(event):
        if var.disabled: return
        i = scroll_delta_translate(event.delta)
        i *= step
        val = var.get()+i
        if mod is not None: val %= mod
        if high is not None: val = min(val, high)
        if low is not None: val = max(val, low)
        var.set(val)
        if on_update is not None: on_update()
    if button: obj.bind("<Button>", button_handler, add="+")
    if scroll: obj.bind("<MouseWheel>", wheel_handler, add="+")
    return var




def float_selector(obj, start=0., low=None, high=None, mod=None, step=1, on_update=None, button=True, scroll=True, can_disable=False):
    # step increments are relative to current float precision
    # and float precision in the var cannot be changed by just incremental step adjustments
    var = FloatVar(value=start, can_disable=can_disable)
    def button_handler(event):
        if var.disabled: return
        match event.num:
            case 1:
                val = var.get()+step/10**var.precision
                if mod is not None: val %= mod
                if high is not None: val = min(val, high)
                if low is not None: val = max(val, low)
                var.set(val, False)
            case 3:
                val = var.get()-step/10**var.precision
                if mod is not None: val %= mod
                if high is not None: val = min(val, high)
                if low is not None: val = max(val, low)
                var.set(val, False)
            case _: return
        if on_update is not None: on_update()
    def wheel_handler(event):
        if var.disabled: return
        i = scroll_delta_translate(event.delta)
        i *= step/10**var.precision
        val = var.get()+i
        if mod is not None: val %= mod
        if high is not None: val = min(val, high)
        if low is not None: val = max(val, low)
        var.set(val, False)
        if on_update is not None: on_update()
    if button: obj.bind("<Button>", button_handler, add="+")
    if scroll: obj.bind("<MouseWheel>", wheel_handler, add="+")
    return var




def create_figure(frame, mouse_handler, width, height):
    figure = Figure(figsize=(width / 100, height / 100), dpi=100)
    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas.get_tk_widget().pack(side=tk.TOP)
    canvas.mpl_connect("button_press_event", mouse_handler)
    subplot = figure.add_subplot()
    return canvas, figure, subplot

def open_msg_window(title, message, error=False):
    if error: messagebox.showerror(title, message)
    else: messagebox.showinfo(title, message)

def open_folder_dialog(title, initial="."):
    return filedialog.askdirectory(title=title, mustexist=True, initialdir=initial)

def open_file_dialog(title, initial="."):
    return filedialog.askopenfilename(title=title, initialdir=initial)

def open_save_dialog(title, initial="."):
    return filedialog.asksaveasfilename(title=title, initialdir=initial)

def write_to_textbox(box, content, clear=False):
    box.configure(state="normal")
    if clear:
        try:
            box.delete(1.0, tk.END)
        except tk.TclError:
            pass
    box.insert(tk.END, content + "\n")
    box.configure(state="disabled")










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

def nice_frame(root, *args, side=tk.LEFT, pack=True, **kwargs):
    frame = tk.Frame(root)
    if pack:
        frame.pack(*args, side=side, **kwargs)
    set_passive_colors(frame)
##    set_random_colors(frame) # DEBUG
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

def nice_field(frame, *args, identifier=None, side=tk.LEFT, fill=tk.BOTH, update_handler=None, **kwargs):
    field = tk.Entry(frame, *args, **kwargs)
    field.pack(side=side, fill=fill)
    if update_handler is not None:
        def _event_handle(event): update_handler(identifier, field.get())
        field.bind("<KeyRelease>", _event_handle)
    set_hover_colors(field, set_active_colors_inverted, set_active_colors)
    return field

def set_field_content(field, string):
    field.delete(0, tk.END)
    field.insert(0, string)

def nice_labeled_field(frame, text, width=DEFAULT_TEXTFIELD_WIDTH, side=tk.TOP, update_handler=None, justify="left", **kwargs):
    subframe = nice_frame(frame, side=side, **kwargs)
    if text:
        label = nice_label(subframe, text=text, side=tk.LEFT, anchor="c") # , width=len(text)
    field = nice_field(subframe, identifier=text, width=width, side=tk.LEFT, justify=justify, update_handler=update_handler)
    return field

def nice_button(frame, *args, side=tk.TOP, fill=tk.BOTH, **kwargs):
    but = tk.Button(frame, *args, **kwargs)
    but.pack(side=side, fill=fill)
    set_hover_colors(but)
    return but






def padded_frame(frame, *args, **kwargs):
    f = nice_frame(frame, *args, **kwargs)
    set_padding(f)
    return f

def raised_frame(frame, *args, border=1, **kwargs):
    f = nice_frame(frame, *args, **kwargs)
    set_passive_colors_inverted(f)
    set_padding(f, border, border)
    return nice_frame(f, side=tk.TOP, fill=tk.BOTH)

def nice_titled_frame(frame, title, *args, side=tk.TOP, border=1, fill=tk.BOTH, **kwargs):
    f = nice_frame(frame, *args, side=side, fill=fill, **kwargs)
    set_padding(f)
    nice_title(f, title, side=tk.TOP, fill=tk.BOTH)
    ff = raised_frame(f, side=tk.TOP, fill=tk.BOTH, border=border)
    set_padding(ff)
##    set_random_colors(f)
    return ff








def nice_RGBA_selector(frame, color, on_update=None): # both RGBA and RGB work fine
    f = nice_frame(frame, anchor="e")
    set_passive_colors_inverted(f)
    
    ff = nice_frame(f, side=tk.LEFT, padx=2)
    preview = nice_label(ff, width=4, side=tk.LEFT)
    def _update_preview_color(index):
        if index==3: preview.configure(text=f"{int((color[index]/255)*100)}%")
        else: set_colors(preview, opposite_color_as_hex(color), color_as_hex(color))
    _update_preview_color(0)
    if len(color)>3: _update_preview_color(3)
    
    _vars = []
    _var_updates = []
    _fields = []
    
    def _update(identifier, string):
        value = read_number_from_string(string)
        if value is not None: color[identifier] = int(value)%256
        else: color[identifier] = 0
        if on_update is not None: on_update()
        _update_preview_color(identifier)
        _vars[identifier].set(color[identifier])
        
    def _var_update(index):
        color[index] = _vars[index].get()
        if on_update is not None: on_update()
        _update_preview_color(index)
        set_field_content(_fields[index], str(color[index]))

    def __new_variable_update_lambda(index):
        _var_updates.append(lambda : _var_update(index))
        
    for i in range(len(color)):
        field = nice_field(f, identifier=i, width=4, justify="center", update_handler=_update)
        field.insert(0, str(color[i]))
        _fields.append(field)
        
        __new_variable_update_lambda(i)
        var = integer_cycler(_fields[i], 256, on_update=_var_updates[i], button=False)
        var.set(color[i])
        _vars.append(var)
        
    return f





def nice_left_label(root, text):
    f = nice_frame(root, side=tk.TOP, anchor="ne")
    nice_label(f, text=text, side=tk.LEFT)
    return f

class IntegerField:
    var = None
    field = None
    on_update = None
    def __init__(self, root, val=0, low=None, high=None, mod=None, step=1, can_disable=False, on_update=None, label_text=None, **kwargs):
        def update_handler(identifier, string):
            if len(string)==0 and not can_disable:
                # reset because in this case there must be a value in the field
                if low is not None: val = low
                elif mod is not None: val = mod
                elif high is not None: val = high
                else: val = self.get()
                self.set(val)
            else:
                val = read_number_from_string(string)
                if val is not None:
                    if mod is not None: val %= mod
                    if high is not None: val = min(val, high)
                    if low is not None: val = max(val, low)
                self.set(val, False)
                if self.on_update is not None: self.on_update()
        if label_text is not None: root = nice_left_label(root, label_text)
        self.field = nice_field(root, update_handler=update_handler, **kwargs)
        self.init_var(val, low, high, mod, step, can_disable=can_disable)
        self.refresh()
        self.on_update = on_update

    def init_var(self, *args, **kwargs):
        self.var = integer_selector(self.field, *args, on_update=self.refresh, button=False, **kwargs)
        
    def __str__(self): return self.field.get()
    
    def refresh(self):
        val = self.get()
        if val is not None: set_field_content(self.field, str(val))
        if self.on_update is not None: self.on_update()
    def get(self): return self.var.get()
    def set(self, value, refresh=True):
        self.var.set(value)
        if refresh: self.refresh()


class FloatField(IntegerField):
    def init_var(self, *args, **kwargs):
        self.var = float_selector(self.field, *args, on_update=self.refresh, button=False, **kwargs)








