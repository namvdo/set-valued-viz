##from OpenGLContext import testingcontext
##BaseContext = testingcontext.getInteractive()
import numpy as np
from OpenGL.GL import *
from OpenGL.arrays import vbo
from OpenGL.GL import shaders
import OpenGL.GLUT as GLUT

##class TestContext( BaseContext ):
##    """Creates a simple vertex shader..."""
##    def OnInit( self ):
##        VERTEX_SHADER = shaders.compileShader("""#version 120
##        void main() {
##            gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
##        }""", GL_VERTEX_SHADER)
##
##        FRAGMENT_SHADER = shaders.compileShader("""#version 120
##        void main() {
##            gl_FragColor = vec4( 0, 1, 0, 1 );
##        }""", GL_FRAGMENT_SHADER)
##        
##        self.shader = shaders.compileProgram(VERTEX_SHADER,FRAGMENT_SHADER)
##        
##        self.vbo = vbo.VBO(
##            array( [
##                [  0, 1, 0 ],
##                [ -1,-1, 0 ],
##                [  1,-1, 0 ],
##                [  2,-1, 0 ],
##                [  4,-1, 0 ],
##                [  4, 1, 0 ],
##                [  2,-1, 0 ],
##                [  4, 1, 0 ],
##                [  2, 1, 0 ],
##            ],'f')
##        )
##
##    def Render( self, mode):
##        """Render the geometry for the scene."""
##        shaders.glUseProgram(self.shader)
##        try:
##            self.vbo.bind()
##            try:
##                glEnableClientState(GL_VERTEX_ARRAY);
##                glVertexPointerf( self.vbo )
##                glDrawArrays(GL_TRIANGLES, 0, 9)
##            finally:
##                self.vbo.unbind()
##                glDisableClientState(GL_VERTEX_ARRAY);
##        finally:
##            shaders.glUseProgram( 0 )



test_shader = None
test_vbo = None
def after_init():
    global test_shader, test_vbo
    VERTEX_SHADER = shaders.compileShader("""#version 120
    void main() {
        gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
    }""", GL_VERTEX_SHADER)
    
    FRAGMENT_SHADER = shaders.compileShader("""#version 120
    void main() {
        gl_FragColor = vec4( 0, 1, .5, 1 );
    }""", GL_FRAGMENT_SHADER)
    
    test_shader = shaders.compileProgram(VERTEX_SHADER,FRAGMENT_SHADER)
    
    test_vbo = vbo.VBO(
        np.array( [
            [  0, 1, 1 ],
            [ -1,-1, 0 ],
            [  1,-1, 0 ],
            [  2,-1, 0 ],
            [  4,-1, 0 ],
            [  4, 1, 0 ],
            [  2,-1, 0 ],
            [  4, 1, 0 ],
            [  2, 1, 0 ],
        ], dtype=np.float64)
    )

def render():
    """Render the geometry for the scene."""
    shaders.glUseProgram(test_shader)
    try:
        test_vbo.bind()
        try:
            glEnableClientState(GL_VERTEX_ARRAY);
            glVertexPointerf( test_vbo )
            glDrawArrays(GL_TRIANGLES, 0, 9)
        finally:
            test_vbo.unbind()
            glDisableClientState(GL_VERTEX_ARRAY);
    finally:
        shaders.glUseProgram( 0 )


extent = np.ones(6)
extent[::2] *= -1

def display_ortho():
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(*extent)

def display():
    glClear(GL_COLOR_BUFFER_BIT)
    # Draw your primitives here
    render()
    glFlush()

def reshape(width, height):
    r = width/height
    glViewport(0, 0, width, height)
    extent[:2] = r
    extent[2:4] = 1/r
    extent[:4] += 3
    extent[:4:2] *= -1
    display_ortho()

def main():
    GLUT.glutInit()
    GLUT.glutInitDisplayMode(GLUT.GLUT_SINGLE | GLUT.GLUT_RGB)
    GLUT.glutInitWindowSize(800, 600)
    GLUT.glutCreateWindow(b"OpenGL with Python")
    after_init()
    glClearColor(0.0, 0.0, 0.0, 1.0)
    GLUT.glutDisplayFunc(display)
    GLUT.glutReshapeFunc(reshape)
    GLUT.glutMainLoop()

if __name__ == "__main__":
    main()
##    TestContext.ContextMainLoop()
