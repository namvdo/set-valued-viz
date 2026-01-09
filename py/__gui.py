import os, re
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog

from PIL.Image import fromarray as PIL_image_from_array
import PIL.ImageTk

def array_to_imagetk(root, array):
    return PIL.ImageTk.PhotoImage(PIL.Image.fromarray(array), master=root.master)

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

def readable_point(point):
    string = "("
    for i,v in enumerate(point):
        if i!=0: string += ", "
        string += str(v)
    string += ")"
    return string

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











def scroll_delta_translate(delta):
    if os.name=="nt": return delta//120 # on windows must divide by 120
    return delta

def integer_cycler(obj, mod, on_update=None, button=1, scroll=1, var=None):
    if var is None:
        var = tk.IntVar(value=0)
    def button_handler(event):
        match event.num:
            case 1: var.set((var.get()+int(button))%mod)
            case 3: var.set((var.get()-int(button))%mod)
            case _: return
        if on_update is not None: on_update()
    def wheel_handler(event):
        var.set((var.get()+int(scroll*scroll_delta_translate(event.delta)))%mod)
        if on_update is not None: on_update()
    if button!=0: obj.bind("<Button>", button_handler, add="+")
    if scroll!=0: obj.bind("<MouseWheel>", wheel_handler, add="+")
    return var

def index_cycler(obj, options, on_update=None, button=1, scroll=1, var=None):
    if var is None:
        var = tk.IntVar(value=0)
    def button_handler(event):
        if len(options)==0: return
        match event.num:
            case 1: var.set((var.get()+int(button))%len(options))
            case 3: var.set((var.get()-int(button))%len(options))
            case _: return
        if on_update is not None: on_update()
    def wheel_handler(event):
        if len(options)==0: return
        var.set((var.get()+int(scroll*scroll_delta_translate(event.delta)))%len(options))
        if on_update is not None: on_update()
    if button!=0: obj.bind("<Button>", button_handler, add="+")
    if scroll!=0: obj.bind("<MouseWheel>", wheel_handler, add="+")
    return var

def string_cycler(obj, options, on_update=None, button=1, scroll=1, var=None):
    if var is None:
        var = tk.StringVar(value=options[0])
        var.option_index = 0
    def button_handler(event):
        if len(options)==0: return
        match event.num:
            case 1:
                var.option_index += int(button)
                var.option_index %= len(options)
                var.set(options[var.option_index])
            case 3:
                var.option_index -= int(button)
                var.option_index %= len(options)
                var.set(options[var.option_index])
            case _: return
        if on_update is not None: on_update()
    def wheel_handler(event):
        if len(options)==0: return
        var.option_index += int(scroll*scroll_delta_translate(event.delta))
        var.option_index %= len(options)
        var.set(options[var.option_index])
        if on_update is not None: on_update()
    if button!=0: obj.bind("<Button>", button_handler, add="+")
    if scroll!=0: obj.bind("<MouseWheel>", wheel_handler, add="+")
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
            if set_precision: self.precision = float_precision2(float(f))
            return super().set(round(f*10**self.precision))


def integer_selector(obj, start=0, low=None, high=None, mod=None, on_update=None, button=1, scroll=1, can_disable=False, var=None):
    if var is None:
        var = IntVar(value=start, can_disable=can_disable)
    def button_handler(event):
        if var.disabled: return
        match event.num:
            case 1:
                val = var.get()+int(button)
                if mod is not None: val %= mod
                if high is not None: val = min(val, high)
                if low is not None: val = max(val, low)
                var.set(val)
            case 3:
                val = var.get()-int(button)
                if mod is not None: val %= mod
                if high is not None: val = min(val, high)
                if low is not None: val = max(val, low)
                var.set(val)
            case _: return
        if on_update is not None: on_update()
    def wheel_handler(event):
        if var.disabled: return
        i = int(scroll*scroll_delta_translate(event.delta))
        val = var.get()+i
        if mod is not None: val %= mod
        if high is not None: val = min(val, high)
        if low is not None: val = max(val, low)
        var.set(val)
        if on_update is not None: on_update()
    if button!=0: obj.bind("<Button>", button_handler, add="+")
    if scroll!=0: obj.bind("<MouseWheel>", wheel_handler, add="+")
    return var




def float_selector(obj, start=0., low=None, high=None, mod=None, on_update=None, button=1, scroll=1, can_disable=False, var=None):
    # step increments are relative to current float precision
    # and float precision in the var cannot be changed by just incremental step adjustments
    if var is None:
        var = FloatVar(value=start, can_disable=can_disable)
    def button_handler(event):
        if var.disabled: return
        match event.num:
            case 1:
                val = var.get()+button/10**var.precision
                if mod is not None: val %= mod
                if high is not None: val = min(val, high)
                if low is not None: val = max(val, low)
                var.set(val, False)
            case 3:
                val = var.get()-button/10**var.precision
                if mod is not None: val %= mod
                if high is not None: val = min(val, high)
                if low is not None: val = max(val, low)
                var.set(val, False)
            case _: return
        if on_update is not None: on_update()
    def wheel_handler(event):
        if var.disabled: return
        i = scroll_delta_translate(event.delta)
        i *= scroll/10**var.precision
        val = var.get()+i
        if mod is not None: val %= mod
        if high is not None: val = min(val, high)
        if low is not None: val = max(val, low)
        var.set(val, False)
        if on_update is not None: on_update()
    if button!=0: obj.bind("<Button>", button_handler, add="+")
    if scroll!=0: obj.bind("<MouseWheel>", wheel_handler, add="+")
    return var




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










def nice_window(title, configure_handler=None, on_destroy=None, resizeable=False):
    win = tk.Tk()
    win.wm_title(title)
    if configure_handler is not None:
        win.bind("<Configure>", configure_handler)
    if on_destroy is not None:
        def _destroy():
            on_destroy()
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _destroy)
    win.resizable(resizeable, resizeable)
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

def nice_canvas(frame, width, height, side=tk.LEFT, fill=tk.BOTH, expand=True, **kwargs):
    canvas = tk.Canvas(frame, width=width, height=height, **kwargs)
    canvas.pack(side=side, fill=fill, expand=expand)
    return canvas

def nice_label(*args, side=tk.TOP, fill=tk.BOTH, expand=False, **kwargs):
    label = tk.Label(*args, **kwargs)
    label.pack(side=side, fill=fill, expand=expand)
    set_passive_colors(label)
    return label

def nice_title(frame, text, *args, side=tk.TOP, fill=tk.BOTH, expand=False, **kwargs):
    label = tk.Label(frame, text=text.title(), *args, **kwargs)
    label.pack(side=side, fill=fill, expand=expand)
    set_passive_colors_inverted(label)
    return label

def nice_field(frame, *args, identifier=None, side=tk.LEFT, fill=tk.BOTH, expand=False, update_handler=None, **kwargs):
    field = tk.Entry(frame, *args, **kwargs)
    field.pack(side=side, fill=fill, expand=expand)
    if update_handler is not None:
        def _event_handle(event): update_handler(identifier, field.get())
        field.bind("<KeyRelease>", _event_handle)
    set_hover_colors(field, set_active_colors_inverted, set_active_colors)
    return field

def clear_field_content(field):
    field.delete(0, tk.END)

def set_field_content(field, string):
    field.delete(0, tk.END)
    field.insert(0, string)

def nice_labeled_field(frame, text, width=DEFAULT_TEXTFIELD_WIDTH, side=tk.TOP, update_handler=None, justify="left", **kwargs):
    subframe = nice_frame(frame, side=side, **kwargs)
    if text:
        label = nice_label(subframe, text=text, side=tk.LEFT, anchor="c") # , width=len(text)
    field = nice_field(subframe, identifier=text, width=width, side=tk.LEFT, justify=justify, update_handler=update_handler)
    return field

def nice_button(frame, *args, side=tk.TOP, fill=tk.BOTH, expand=False, **kwargs):
    but = tk.Button(frame, *args, **kwargs)
    but.pack(side=side, fill=fill, expand=expand)
    set_hover_colors(but)
    return but






def padded_frame(frame, *args, **kwargs):
    f = nice_frame(frame, *args, **kwargs)
    set_padding(f)
    return f

def raised_frame(frame, *args, border=1, expand=True, **kwargs):
    f = nice_frame(frame, *args, expand=expand, **kwargs)
    set_passive_colors_inverted(f)
    set_padding(f, border, border)
    return nice_frame(f, side=tk.TOP, fill=tk.BOTH, expand=expand)

def nice_titled_frame(frame, title, *args, side=tk.TOP, border=1, fill=tk.BOTH, **kwargs):
    f = nice_frame(frame, *args, side=side, expand=True, fill=fill, **kwargs)
    set_padding(f)
    nice_title(f, title, side=tk.TOP, fill=tk.BOTH)
    ff = raised_frame(f, side=tk.TOP, expand=True, fill=tk.BOTH, border=border)
    set_padding(ff)
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





def nice_left_label(root, text, side=tk.TOP, fill=tk.BOTH):
    f = nice_frame(root, side=side, anchor="ne", fill=fill)
    nice_label(f, text=text, anchor="w", justify="left", side=tk.LEFT, fill=tk.BOTH)
    return f

class IntegerField:
    var = None
    field = None
    on_update = None
    def __init__(self, root, val=0, low=None, high=None, mod=None, scroll=1, can_disable=False, on_update=None, label_text=None, **kwargs):
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
        if label_text is not None:
            root = nice_left_label(root, label_text, side=kwargs.get("side", tk.TOP), fill=kwargs.get("fill", tk.BOTH))
            kwargs["side"] = tk.RIGHT
            kwargs["fill"] = tk.BOTH
        self.field = nice_field(root, update_handler=update_handler, **kwargs)
        self.init_var(val, low, high, mod, scroll=scroll, can_disable=can_disable)
        self.refresh()
        self.on_update = on_update

    def init_var(self, *args, **kwargs):
        self.var = integer_selector(self.field, *args, on_update=self.refresh, button=0, **kwargs)
        
    def __str__(self): return self.field.get()
    
    def refresh(self):
        val = self.get()
        if val is not None: set_field_content(self.field, str(val))
        else: clear_field_content(self.field)
        if self.on_update is not None: self.on_update()
    def get(self): return self.var.get()
    def set(self, value, refresh=True):
        self.var.set(value)
        if refresh: self.refresh()


class FloatField(IntegerField):
    def init_var(self, *args, **kwargs):
        self.var = float_selector(self.field, *args, on_update=self.refresh, button=0, **kwargs)



class BooleanSwitch():
    value = False
    text = None
    button = None
    def __init__(self, root, value=False, text=None, on_update=None, label_text=None, **kwargs):
        if label_text is not None:
            root = nice_left_label(root, label_text, side=kwargs.get("side", tk.TOP), fill=kwargs.get("fill", tk.BOTH))
            kwargs["side"] = tk.TOP
            kwargs["fill"] = tk.BOTH
        def on_press():
            self.set(not self.get())
            if on_update is not None: on_update()
        kwargs["width"] = 4
        if text is not None: kwargs["width"] += len(text)
        self.button = nice_button(root, command=on_press, **kwargs)
        self.value = value
        self.text = text
        self.update_visuals()

    def update_visuals(self):
        if self.value:
            set_hover_colors(self.button, set_active_colors, set_active_colors_inverted)
            self.button.configure(text=f"{self.text} - ON" if self.text else "ON")
        else:
            set_hover_colors(self.button, set_passive_colors_inverted, set_passive_colors)
            self.button.configure(text=f"{self.text} - OFF" if self.text else "OFF")

    def get(self): return self.value
    
    def set(self, value:bool):
        if value!=self.value:
            self.value = value
            self.update_visuals()





class StringList():
    strings = None
    labels = None
    var = None

    prev_index = 0
    def __init__(self, root, strings, visible=1, on_update=None, label_text=None, anchor="c", justify="center", **kwargs):
        self.strings = strings
        if label_text is not None:
            root = nice_left_label(root, label_text, side=kwargs.get("side", tk.TOP), fill=kwargs.get("fill", tk.BOTH))
            kwargs["side"] = tk.TOP
            kwargs["fill"] = tk.BOTH
        
        kwargs["anchor"] = anchor
        f = nice_frame(root, **kwargs)
        
        ff = nice_frame(f, side=tk.RIGHT, fill=tk.Y)
        set_passive_colors_inverted(ff)
        l = nice_label(ff, text="▲", side=tk.TOP)
        set_passive_colors_inverted(l)
        l = nice_label(ff, text="▼", side=tk.BOTTOM)
        set_passive_colors_inverted(l)

        ff = nice_frame(f, side=tk.TOP, fill=tk.BOTH)
        self.init_labels(ff, visible, anchor=anchor, justify=justify, button=0, scroll=-1, on_update=self.refresh) # , low=0, high=len(self.strings)-1
        

    def init_labels(self, f, visible, anchor="c", justify="center", **kwargs):
##        self.var = integer_selector(f, **kwargs)
        self.var = index_cycler(f, self.strings, **kwargs)
        
        self.labels = []
        for i in range(visible):
            obj = nice_label(f, anchor=anchor, justify=justify)
            set_passive_colors_inverted(obj)
            index_cycler(obj, self.strings, **kwargs, var=self.var)
##            integer_selector(obj, **kwargs, var=self.var)
            self.labels.append(obj)
        self.refresh()
        
##    def refresh(self):
##        # wraparound
##        index = self.var.get()
##        l = len(self.labels)
##        ll = len(self.strings)
##        for i in range(len(self.labels)):
##            if i+index < ll or ll>=l: text = self.strings[(i+index)%ll]
##            else: text = ""
##            self.labels[i].configure(text=text)
            
    def refresh(self):
        # no wraparound
        index = self.var.get()
        l = len(self.labels)
        ll = len(self.strings)
        diff = self.prev_index-index
        if ll>l and not (self.prev_index==0 and index==ll-1):
            over = max((index+l)-ll, 0)
            if over>0: index -= over
        else: index = 0
        self.var.set(index)
        self.prev_index = index
        for i in range(len(self.labels)):
            if i+index < ll or ll>=l: text = self.strings[(i+index)%ll]
            else: text = ""
            self.labels[i].configure(text=text)

    def __len__(self): return len(self.strings)

    def clear(self):
        self.strings.clear()

    def insert(self, index, string):
        self.strings.insert(index, string)
        
    def append(self, string):
        self.strings.append(string)
    
    def pop(self, index=-1):
        return self.strings.pop(index)
    
    def swap(self, index, string):
        self.strings[index], string = string, self.strings[index]
        return string



class TextBox():
    box = None
    def __init__(self, root, *args, **kwargs):
        self.box = nice_textbox(root, *args, **kwargs)
    
    def write(self, string):
        self.box.configure(state="normal")
        self.box.insert(tk.END, string)
        self.box.configure(state="disabled")
    
    def clear(self):
        self.box.configure(state="normal")
        self.box.delete(1.0, tk.END)
        self.box.configure(state="disabled")
    
    def print(self, *args, end="\n"):
        string = " ".join([str(x) for x in args])
        string += end
        self.write(string)



class Canvas():
    configures_to_ignore = 0
    canvas = None
    def __init__(self, root, width, height, side=tk.TOP, fill=tk.BOTH, anchor="nw", **kwargs):
        self.canvas = nice_canvas(root, width, height, **kwargs)
        self.canvas.config(highlightthickness=0) # stop endless resize loop
        self.canvas.pack_propagate(0)
        
        # auto check widgets
        def configure_handler(event):
            if self.configures_to_ignore>=0: self.configures_to_ignore -= 1
            else: self.check_that_every_widget_is_inside()
        self.canvas.bind("<Configure>", configure_handler, add="+")
        #
        
        self.images = set() # keep images from being garbage collected
        self.drag_start_pos = {} # key -> pos
        
        self.widget_keys = {} # widget -> key
        self.widget_visible = {} # widget -> boolean
        
        self._canvas_panning_init()

    def get_inside_shape(self):
        return (self.canvas.winfo_width(), self.canvas.winfo_height())
    def get_canvas_shape(self):
        padx = self.canvas.master["padx"]
        pady = self.canvas.master["pady"]
        if not isinstance(padx, int): padx = int(str(padx))
        if not isinstance(pady, int): pady = int(str(pady))
        return (self.canvas.master.winfo_width()-padx*2, self.canvas.master.winfo_height()-pady*2)


    def _canvas_panning_init(self):
        self.pannable = {} # key -> panning offset
        
        def drag_handler(event):
            prev = self.drag_start_pos[self.canvas]
            now = np.array((event.x, event.y))
            self.drag_start_pos[self.canvas] = now
            self.panning_move(now-prev)
        
        def button_handler(event):
            if self.canvas in self.drag_start_pos: del self.drag_start_pos[self.canvas]
            self.drag_start_pos[self.canvas] = np.array((event.x, event.y))

        def reset_panning(event):
            for k,offset in self.pannable.items():
                pos = self.coords(key)
                self.coords(key, np.subtract(pos, offset))
                offset[:] = 0
        
        self.canvas.bind("<Button-1>", button_handler, add="+")
        self.canvas.bind("<Button-2>", self.panning_reset, add="+")
        self.canvas.bind("<B1-Motion>", drag_handler, add="+")
    
    def panning_reset(self, event=None):
        for k,offset in self.pannable.items():
            pos = self.coords(k)
            self.coords(k, np.subtract(pos, offset))
            offset[:] = 0

    def panning_move(self, move):
        for k,offset in self.pannable.items():
            offset += move
            self.move(k, move)

    def move(self, key, offset): # move by offset amount
        self.canvas.move(key, *offset)
        
    def coords(self, key, pos=None): # set new position or get current position
        if pos is None: return self.canvas.coords(key)
        return self.canvas.coords(key, *pos)


    # natives
    def image(self, image, pos=(0,0), anchor="nw", pan=True, **kwargs):
        if type(image)==np.ndarray: image = array_to_imagetk(self.canvas, image)
        self.images.add(image)
        key = self.canvas.create_image(*pos, image=image, anchor=anchor, **kwargs)
        return self._native_canvas_obj(key, pan=pan)

    def text(self, text, pos=(0,0), anchor="nw", pan=True, **kwargs):
        key = self.canvas.create_text(*pos, text=text, anchor=anchor, **kwargs)
        return self._native_canvas_obj(key, pan=pan)
    
    def _native_canvas_obj(self, key, pan=True):
        if pan: self.pannable[key] = np.zeros(2)
        return key
    #

    def get_topmost(self, obj):
        topmost = obj
        while topmost.master is not self.canvas: topmost = topmost.master
        return topmost
    
    def window(self, obj, pos=(0,0), anchor="nw"):
        topmost = self.get_topmost(obj)
        key = self.canvas.create_window(*pos, anchor=anchor, window=topmost)
        self.widget_keys[topmost] = key
        self.configures_to_ignore += 1
        return key

    def _keep_obj_inside(self, key, obj):
        coords = self.coords(key)
        canvas_shape = np.array(self.get_inside_shape())
        obj_shape = np.array((obj.winfo_width(), obj.winfo_height()))
        new_coords = np.clip(coords, a_min=0, a_max=abs(canvas_shape-obj_shape))
        if (coords!=new_coords).any(): self.coords(key, new_coords)
    
    def check_that_every_widget_is_inside(self):
        for obj,key in self.widget_keys.items():
            self._keep_obj_inside(key, obj)
        
    def _movable_obj(self, key, obj):
        # make object draggable
        topmost = self.get_topmost(obj)
        
        def drag_handler(event):
            prev = self.drag_start_pos[key]
            self.move(key, np.array((event.x, event.y))-prev)
            self._keep_obj_inside(key, topmost)
            
        def button_handler(event):
            if key in self.drag_start_pos: del self.drag_start_pos[key]
            self.drag_start_pos[key] = np.array((event.x, event.y))
        
        obj.bind("<Button-1>", button_handler, add="+")
        obj.bind("<B1-Motion>", drag_handler, add="+")
    
    def _liftable_obj(self, obj):
        # when clicked -> raise to top
        topmost = self.get_topmost(obj)
        def button_handler(event):
            topmost.lift()
        obj.bind("<Button-1>", button_handler, add="+")
        
    
    def movable_window(self, title, pos=(0,0)):
        f = nice_titled_frame(self.canvas, title)
        topmost = self.get_topmost(f)
        set_padding(topmost, 1, 1)
        key = self.window(topmost, pos)
        
        for obj in topmost.slaves():
            self._movable_obj(key, obj)
            self._liftable_obj(obj)
            self.hide_trigger(obj, f.master)
        self._liftable_obj(f)
        return key, f

    def hide_trigger(self, trigger, target):
        self.widget_visible[target] = self.widget_visible.get(target, True)
        def button_handler(event):
            if self.widget_visible[target]: target.pack_forget()
            else: target.pack()
            self.widget_visible[target] = not self.widget_visible[target]
        trigger.bind("<Button-2>", button_handler, add="+")
        trigger.bind("<Button-3>", button_handler, add="+")

if __name__ == "__main__":
    win = nice_window("test", resizeable=True)
    set_padding(win)
##    win.configure(width=500, height=500)
##    win.pack_propagate(0) # do not let children trigger resizing
    
    topframe = nice_titled_frame(win, "TOPFRAME", fill=tk.BOTH)
    canvas = Canvas(topframe, 500, 100)
    
    f = nice_button(canvas.canvas, text="button")
    canvas.window(f, (0,0))
    
    k,f1 = canvas.movable_window("title")
    canvas.coords(k, (100,50))
    b1 = nice_button(f1, text="button")
    
    k,f2 = canvas.movable_window("title2", (100,100))
    b2 = nice_button(f2, text="button")
    
##    canvas.hide_trigger(b1, f2.master)
##    canvas.hide_trigger(b2, f1.master)
    
    key = canvas.image(np.random.random((100,100))*255, (200,50))
    canvas.move(key, (-200,0))
    canvas.coords(key, (0,10))

    k = canvas.text("asd", (400,100))
    
    win.mainloop()






