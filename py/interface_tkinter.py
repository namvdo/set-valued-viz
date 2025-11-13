from _imports import *
from func_tkinter import *
import PIL

from normals_model import ModelConfiguration as NormalsModel

save_directory = os.path.join(WORKDIR, "saves")


def nice_button(frame, text, handler, color_text="black", color_bg="white", hover_color_text="white", hover_color_bg="black"):
    but = create_button(frame, text, handler)
    but.config(fg=color_text, bg=color_bg, activebackground=color_bg)
    def _event_handle0(event):
        event.widget["fg"] = hover_color_text
        event.widget["bg"] = hover_color_bg
    def _event_handle1(event):
        event.widget["fg"] = color_text
        event.widget["bg"] = color_bg
    but.bind("<Enter>", _event_handle0)
    but.bind("<Leave>", _event_handle1)
    return but

def create_labeled_field(frame, label, width, update_handler=None):
    subframe = create_frame(frame, tk.TOP)
    label = tk.Label(subframe, text=label, width=len(label)+1, anchor="n")
    label.pack(side=tk.LEFT, fill=tk.BOTH)
    field = tk.Entry(subframe, width=width)
    field.pack(side=tk.LEFT, fill=tk.BOTH)
    if update_handler is not None:
        def _event_handle(event): update_handler(field.get())
        field.bind("<KeyRelease>", _event_handle)
    return field

class Interface():
    def __init__(self):
        self.start_point = Point2D(0,0)
        self.function = MappingFunction2D()
        self.window = create_window2("main")
        self._setup_main_window()

    def _setup_main_window(self):
        win_frame0 = create_frame(self.window)
        create_vertical_separator(win_frame0, 4, 4)
        
        win_frame1 = create_frame(win_frame0)
        create_vertical_separator(win_frame0, 4, 4)
        
        self._logbox = create_textbox(win_frame0, 48, 12)
        
        # main window content
        create_horizontal_separator(win_frame1, 0, 4)
        
        b1 = nice_button(win_frame1, "Start", self.start_model_instance)
        
        create_horizontal_separator(win_frame1, 0, 6)
        
        field_width = 14
        create_label(win_frame1, "function")
        def _update_fx(string): self.function.x.string = string
        def _update_fy(string): self.function.y.string = string
        field_fx = create_labeled_field(win_frame1, "fx", field_width, _update_fx)
        field_fy = create_labeled_field(win_frame1, "fy", field_width, _update_fy)
        field_fx.insert(0, self.function.x.string)
        field_fy.insert(0, self.function.y.string)
        
        create_horizontal_separator(win_frame1, 0, 6)
        
        label = create_label(win_frame1, "starting point")
        def _update_x(string):
            self.start_point.x = float(string)
            label.configure(text=str(self.start_point))
        def _update_y(string):
            self.start_point.y = float(string)
            label.configure(text=str(self.start_point))
        field_start_x = create_labeled_field(win_frame1, " x", field_width, _update_x)
        field_start_y = create_labeled_field(win_frame1, " y", field_width, _update_y)
        field_start_x.insert(0, str(self.start_point.x))
        field_start_y.insert(0, str(self.start_point.y))
        
        create_horizontal_separator(win_frame1, 0, 4)
        
    def log(self, string):
        write_to_textbox(self._logbox, string)
    def log_error(self, string):
        write_to_textbox(self._logbox, f"ERROR: {string}")

    def start_model_instance(self):
        normals_model = NormalsModel()
        normals_model.start_point.x = self.start_point.x
        normals_model.start_point.y = self.start_point.y
        normals_model.function = self.function.copy()
        
        win = create_window2("model")
        win_frame0 = create_frame(win, tk.LEFT)
        create_vertical_separator(win_frame0, 4, 4)
        win_frame1 = create_frame(win_frame0, tk.LEFT)
        create_vertical_separator(win_frame0, 4, 4)
        
        canvas = tk.Canvas(win_frame0,width=512,height=512)
        canvas.pack()
        
        create_horizontal_separator(win_frame1, 0, 4)
        create_label(win_frame1, str(normals_model.function))
        create_horizontal_separator(win_frame1, 0, 6)
        b = nice_button(win_frame1, "Next", None)
        create_horizontal_separator(win_frame1, 0, 6)
        self.img = None
        def _draw():
            normals_model.process(2)
            array, tl, br = normals_model.draw(512)
            self.img = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(array.swapaxes(0, 1)[::-1]), master=win_frame0)
            shape = (self.img.width()+10, self.img.height()+10)
            canvas.configure(width=shape[0], height=shape[1])
            canvas.create_image(10,10, anchor="nw", image=self.img)
            
        b = nice_button(win_frame1, "Draw", _draw)
        create_horizontal_separator(win_frame1, 0, 4)
        
    def start(self):
        self.window.mainloop()


Interface().start()
