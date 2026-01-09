from __gui import *

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
matplotlib.use("TkAgg")

from __system import function_timer

class Viewport():
    figure = None
    subplot = None
    canvas = None
    
    def __init__(self, root, width=512, height=512):
        def mousebutton_handler(event):
            pass
        self.figure = Figure(figsize=(width / 100, height / 100), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(side=tk.TOP)
        self.canvas.mpl_connect("button_press_event", mousebutton_handler)
        self.subplot = self.figure.add_subplot()
        
    def clear(self):
        self.subplot.clear()
        self.canvas.draw()
    
    def update(self, image, extent, title=None):
        self.subplot.clear()
        self.subplot.imshow(image, extent=extent[:4])
        self.subplot.set_title(title)
        self.canvas.draw()

    def get_widget(self):
        return self.canvas.get_tk_widget()



if __name__ == "__main__":
##    win = nice_window("test")
##    vp = Viewport3D(win, width=512, height=512)
##    vp.get_widget().bind("t", vp.test, add="+")
##    
##    win.mainloop()
    pass
