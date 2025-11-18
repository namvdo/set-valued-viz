from _imports import *
from __gui import *
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
    instances_created = 0
    class SubInstance():
        key = None
        step = 0
        model = None
        image = None
        
    def __init__(self):
        self.normals_model = NormalsModel()
        self.subinstances = {}
        self._init_main_window()

    def _init_main_window(self):
        self.window = create_window2("main")
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
        def _update_fx(string): self.normals_model.function.fx.string = string
        def _update_fy(string): self.normals_model.function.fy.string = string
        field_fx = create_labeled_field(win_frame1, "fx", field_width, _update_fx)
        field_fy = create_labeled_field(win_frame1, "fy", field_width, _update_fy)
        field_fx.insert(0, self.normals_model.function.fx.string)
        field_fy.insert(0, self.normals_model.function.fy.string)
        
        create_horizontal_separator(win_frame1, 0, 6)
        
        label = create_label(win_frame1, "starting point")
        def _update_x(string):
            self.normals_model.start_point.x = float(string)
            label.configure(text=str(self.normals_model.start_point))
        def _update_y(string):
            self.normals_model.start_point.y = float(string)
            label.configure(text=str(self.normals_model.start_point))
        field_start_x = create_labeled_field(win_frame1, " x", field_width, _update_x)
        field_start_y = create_labeled_field(win_frame1, " y", field_width, _update_y)
        field_start_x.insert(0, str(self.normals_model.start_point.x))
        field_start_y.insert(0, str(self.normals_model.start_point.y))
        
        create_horizontal_separator(win_frame1, 0, 4)
        
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
        
        win = create_window2("model")
        win_frame0 = create_frame(win, tk.LEFT)
        create_vertical_separator(win_frame0, 4, 4)
        win_frame1 = create_frame(win_frame0, tk.LEFT)
        create_vertical_separator(win_frame0, 4, 4)
        
        canvas = tk.Canvas(win_frame0,width=512,height=512)
        canvas.pack()
        
        create_horizontal_separator(win_frame1, 0, 4)
        create_label(win_frame1, str(instance.model.function))
        create_horizontal_separator(win_frame1, 0, 6)
        
        def _next():
            instance.step += 1
            instance.model.process(instance.step)
        b = nice_button(win_frame1, "Next", _next)
        create_horizontal_separator(win_frame1, 0, 6)
        
        def _draw():
            if not instance.model.can_draw(): instance.model.process(instance.step)
            array, tl, br = instance.model.draw(512)
            instance.image = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(array.swapaxes(0, 1)[::-1]), master=win_frame0)
            shape = (instance.image.width()+10, instance.image.height()+10)
            canvas.configure(width=shape[0], height=shape[1])
            canvas.create_image(10,10, anchor="nw", image=instance.image)
            
        b = nice_button(win_frame1, "Draw", _draw)
        create_horizontal_separator(win_frame1, 0, 4)
        
    def start(self):
        self.window.mainloop()


Interface().start()
