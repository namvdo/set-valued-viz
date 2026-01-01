from _imports import *
from _quick_visuals import *


@function_timer
def test1(*args, **kwargs):
    return hausdorff_distance7(*args, **kwargs)

@function_timer
def test2(*args, **kwargs):
    return hausdorff_distance8(*args, **kwargs)

##@function_timer
##def test3(*args, **kwargs):
##    return hausdorff_distance9(*args, **kwargs)

def hausdorff_distance_testing():
    # hausdorff distance function testing
    n = 1280
    for i in range(10):
        print("\n\n", n)
        print("\nvery close")
        points1 = np.random.random((n,2))
        points2 = points1+0.001
        print(test1(points1, points2))
        print(test2(points1, points2))
##        print(test3(points1, points2))
        
        print("\nnowhere near diagonal")
        points1 = np.random.random((n,2))
        points2 = np.random.random((n,2))+20
        print(test1(points1, points2))
        print(test2(points1, points2))
##        print(test3(points1, points2))
        
        print("\nnowhere near horizontal")
        points1 = np.random.random((n,2))
        points2 = np.random.random((n,2))
        points2[:,0] += 20
        print(test1(points1, points2))
        print(test2(points1, points2))
##        print(test3(points1, points2))
        
        print("\nquarter overlap")
        points1 = np.random.random((n,2))
        points2 = np.random.random((n,2))+.5
        print(test1(points1, points2))
        print(test2(points1, points2))
##        print(test3(points1, points2))
        
        print("\nhalf overlap")
        points1 = np.random.random((n,2))
        points2 = np.random.random((n,2))
        points2[:,0] += .5
        print(test1(points1, points2))
        print(test2(points1, points2))
##        print(test3(points1, points2))
        
        print("\nfully inside")
        points1 = np.random.random((n,2))
        points2 = np.random.random((n,2))*4-1
        print(test1(points1, points2))
        print(test2(points1, points2))
##        print(test3(points1, points2))


        print("\ninside crescent")
        points1 = np.random.random((n,2))
        points2 = (np.random.random((n,2))-.5)*3
        move_mask = (points2[:,0]>0)*(points2[:,1]>0)
        points2[move_mask] *= -1
        print(test1(points1, points2))
        print(test2(points1, points2))
        
        input("continue...")
        n <<= 1


def image_drawing_3d_testing():
    dome = point_ball(12, 1)
    drawing = ImageDrawing()
    
    drawing.pitch = np.pi/3
    drawing.yaw = np.pi/16
    drawing.tilt = np.pi/16

    for circle in dome:
        obj = drawing.lines(*point_lines(circle))
        obj.set_color(r=1)
        obj.set_color_bg(b=1, a=0.1)
        
    drawing.test_draw(1000, camera_dist=2)  #


def bezier_curve_testing():
    # curve
    point1 = (-1,1)
    normal1 = (-1,1)
    
    point2 = (1,0)
    normal2 = (3,1)
    #

    tangents = rotate_vectors(np.array((normal1, normal2)), np.pi/2)
    curve_intsect = bezier_curve_intersection(point1, point2, *tangents)
    curve = bezier_curve(point1, point2, curve_intsect, 50)

    drawing = ImageDrawing()
    obj = drawing.circles([curve_intsect], 0.03)
    
    obj = drawing.circles([point1, point2], 0.03)
    obj.set_color(b=1)
    
    obj = drawing.lines([point1,point1,point2], [point2,curve_intsect,curve_intsect])
    obj.set_color(b=1)
    
    obj = drawing.circles(curve, 0.02)
    obj.set_color(r=1)
    
    drawing.test_draw(1000)





if __name__ == "__main__":
##    hausdorff_distance_testing()
##    image_drawing_3d_testing()
##    bezier_curve_testing()
    pass
