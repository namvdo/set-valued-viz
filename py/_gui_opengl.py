from __gui import *

import OpenGL.GL as GL
##import OpenGL.GLUT as GLUT

from pyopengltk import OpenGLFrame

##def display():
##    GL.glClear(GL.GL_COLOR_BUFFER_BIT)
##    GL.glMatrixMode(GL.GL_MODELVIEW)
##    GL.glLoadIdentity()
##    # Draw your primitives here
##    GL.glFlush()

##def reshape(width, height):
##    GL.glViewport(0, 0, width, height)
##    GL.glMatrixMode(GL.GL_PROJECTION)
##    GL.glLoadIdentity()
##    GL.glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)

##def main():
##    GLUT.glutInit()
##    GLUT.glutInitDisplayMode(GLUT.GLUT_SINGLE | GLUT.GLUT_RGB)
##    GLUT.glutInitWindowSize(800, 600)
##    GLUT.glutCreateWindow(b"OpenGL with Python")
##    GL.glClearColor(0.0, 0.0, 0.0, 1.0)
##    GLUT.glutDisplayFunc(display)
##    GLUT.glutReshapeFunc(reshape)
##    GLUT.glutMainLoop()

class AppOgl(OpenGLFrame):

    pov_zoom = 1
    pov_pos = np.zeros(3)
    pov_rot = np.zeros(3)
    
    def initgl(self):
        """Initalize gl states when the frame is created"""
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClearColor(0.0, 1.0, 0.0, 0.0)
    
    def redraw(self):
        """Render a single frame"""
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        # Draw your primitives here
        
        self.test_draw()
        
        #
        GL.glFlush()


    def primitive_points(self, vertices, size=None, color=None):
        if size is not None: GL.glPointSize(size)
        if color is not None: GL.glColor3f(*color)
        GL.glBegin(GL.GL_POINTS)
        for v in vertices: GL.glVertex3f(*v)
        GL.glEnd()
    
    def primitive_polygon(self, vertices, size=None, color=None):
        if size is not None: GL.glLineWidth(size)
        if color is not None: GL.glColor3f(*color)
        GL.glBegin(GL.GL_LINE_LOOP)
        for v in vertices: GL.glVertex3f(*v)
        GL.glEnd()

    def test_draw(self):
        vertices = np.random.random((20,3))
        vertices -= .5
        vertices *= 2
        self.primitive_points(vertices, size=10, color=(1,0,0))
        self.primitive_polygon(vertices, size=5, color=(0,0,1))

class Viewport():
    frame = None
    def __init__(self, root, side=tk.TOP, fill=tk.BOTH, expand=True, **kwargs):
        self.frame = AppOgl(root, **kwargs)
        self.frame.pack(side=side, fill=fill, expand=expand)
        

if __name__ == "__main__":
    win = nice_window("test", resizeable=True)
    vp = Viewport(win, width=512, height=200)

    win.mainloop()
    pass
