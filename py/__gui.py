import os, re
import tkinter as tk
from tkinter import messagebox, filedialog

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








