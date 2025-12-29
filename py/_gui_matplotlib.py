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
    
    def update(self, image, tl, br, title=None):
        self.subplot.clear()
        self.subplot.imshow(image, extent=(tl[0], br[0], tl[1], br[1]))
        self.subplot.set_title(title)
        self.canvas.draw()

    def get_widget(self):
        return self.canvas.get_tk_widget()


##class Viewport3D(Viewport):
##    camera_pos = None
##    
##    camera_yaw = 0 # yaw
##    z_rotation = 0 # tilt
##    x_rotation = 0 # pitch
##
##    points = None
##    normals = None
##    
##    def __init__(self, *args, **kwargs):
##        super().__init__(*args, **kwargs)
##        self.camera_pos = np.zeros(3)
##        self.objects = []
##    
##    def start_plotting(self):
##        self.subplot.clear()
##        
##    def plot_points(self, points, size=None, color=None):
##        self.subplot.scatter(points[:,0], points[:,1], alpha=1-points[:,2])
##
##    def finish_plotting(self, title=None):
##        self.subplot.set_title(title)
##        self.canvas.draw()
##
##    def test(self, event):
##        self.start_plotting()
##
##        center = 
##
##        points = np.random.random((10,3))
##        self.plot_points(points, size=0.2)
##        
##        self.finish_plotting()


if __name__ == "__main__":
##    win = nice_window("test")
##    vp = Viewport3D(win, width=512, height=512)
##    vp.get_widget().bind("t", vp.test, add="+")
##    
##    win.mainloop()
    pass
