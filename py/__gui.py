import tkinter as tk
from tkinter.ttk import Separator
from tkinter import messagebox, filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
matplotlib.use("TkAgg")

def apply_font(obj, fontname=None, fontsize=None):
    if fontname is not None or fontsize is not None:
        obj.configure(font=(fontname, fontsize))

def create_window(title):
    global window
    window = tk.Tk()
    window.wm_title(title)
    return window

def create_window2(title):
    window = tk.Tk()
    window.wm_title(title)
    return window

def create_frame(host, side=tk.LEFT):
    frame = tk.Frame(host)
    frame.pack(side=side, anchor="n")
    return frame

def create_button(frame, label, handler):
    button = tk.Button(frame, text=label, command=handler)
    button.pack(side=tk.TOP, fill=tk.BOTH)
    return button

def create_figure(frame, mouse_handler, width, height):
    figure = Figure(figsize=(width / 100, height / 100), dpi=100)
    canvas = FigureCanvasTkAgg(figure, master=frame)
    canvas.get_tk_widget().pack(side=tk.TOP)
    canvas.mpl_connect("button_press_event", mouse_handler)
    subplot = figure.add_subplot()
    return canvas, figure, subplot

def create_textbox(frame, width=80, height=20):
    boxframe = create_frame(frame, tk.TOP)
    scrollbar = tk.Scrollbar(boxframe)
    box = tk.Text(boxframe, height=height, width=width, yscrollcommand=scrollbar.set)
    box.configure(state="disabled")
    box.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbar.configure(command=box.yview)
    return box

def write_to_textbox(box, content, clear=False):
    box.configure(state="normal")
    if clear:
        try:
            box.delete(1.0, tk.END)
        except tk.TclError:
            pass
    box.insert(tk.END, content + "\n")
    box.configure(state="disabled")

def create_listbox(frame, width=80, height=20):
    boxframe = create_frame(frame, tk.TOP)
    scrollbar = tk.Scrollbar(boxframe)
    box = tk.Listbox(boxframe,
        height=height,
        width=width,
        yscrollcommand=scrollbar.set
    )
    box.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbar.configure(command=box.yview)
    return box

def add_list_row(box, content, place=tk.END):
    box.insert(place, content)

def remove_list_row(box, index):
    box.delete(index)

def read_selected(box):
    selected = box.curselection()
    if selected:
        content = box.get(selected)
        return selected[0], content
    return None, None

def create_label(frame, text, side=tk.TOP, fill=tk.BOTH):
    label = tk.Label(frame, text=text)
    label.pack(side=side, fill=fill)
    return label

def update_label(label, text):
    label.configure(text=text)

def create_textfield(frame, side=tk.TOP, fill=tk.BOTH):
    field = tk.Entry(frame, width=200)
    field.pack(side=side, fill=fill)
    return field

def read_field(field):
    return field.get()

def clear_field(field):
    field.delete(0, len(field.get()))

def write_field(field, content):
    field.insert(0, content)

def create_horiz_separator(frame, margin=2):
    separator = Separator(frame, orient="horizontal")
    separator.pack(side=tk.TOP, fill=tk.BOTH, pady=margin)

def create_vert_separator(frame, margin=2):
    separator = Separator(frame, orient="vertical")
    separator.pack(side=tk.TOP, fill=tk.BOTH, pady=margin)

def open_msg_window(title, message, error=False):
    if error: messagebox.showerror(title, message)
    else: messagebox.showinfo(title, message)

def open_folder_dialog(title, initial="."):
    return filedialog.askdirectory(title=title, mustexist=True, initialdir=initial)

def open_file_dialog(title, initial="."):
    return filedialog.askopenfilename(title=title, initialdir=initial)

def open_save_dialog(title, initial="."):
    return filedialog.asksaveasfilename(title=title, initialdir=initial)

def remove_component(component):
    try: component.destroy()
    except AttributeError: component.get_tk_widget().destroy()

def create_subwindow(title):
    sub = tk.Toplevel()
    sub.title(title)
    sub.protocol("WM_DELETE_WINDOW", sub.withdraw)
    return sub

def show_subwindow(sub, title=None):
    if title:
        sub.title(title)
    sub.deiconify()

def hide_subwindow(sub):
    sub.withdraw()

def start():
    window.mainloop()

def quit():
    window.destroy()


def create_vertical_separator(frame, padx=0, pady=0):
    separator = Separator(frame, orient="vertical")
    separator.pack(side=tk.LEFT, fill=tk.BOTH, padx=padx, pady=pady)
def create_horizontal_separator(frame, padx=0, pady=0):
    separator = Separator(frame, orient="horizontal")
    separator.pack(side=tk.TOP, fill=tk.BOTH, padx=padx, pady=pady)

