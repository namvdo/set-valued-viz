from _imports import *
from _quick_visuals import *


@function_timer
def test1(*args, **kwargs):
    return hausdorff_distance7(*args, **kwargs)

@function_timer
def test2(*args, **kwargs):
    return hausdorff_distance8(*args, **kwargs)

@function_timer
def test3(*args, **kwargs):
    return hausdorff_distance5(*args, **kwargs)

if __name__ == "__main__":
    # hausdorff distance function testing
    n = 128
    for i in range(10):
        print("\n\n", n)
        print("\nvery close")
        points1 = np.random.random((n,2))
        points2 = points1+0.001
        print(test1(points1, points2))
        print(test2(points1, points2))
        
        print("\nnowhere near diagonal")
        points1 = np.random.random((n,2))
        points2 = np.random.random((n,2))+20
        print(test1(points1, points2))
        print(test2(points1, points2))
        
        print("\nnowhere near horizontal")
        points1 = np.random.random((n,2))
        points2 = np.random.random((n,2))
        points2[:,0] += 20
        print(test1(points1, points2))
        print(test2(points1, points2))
        
        print("\nquarter overlap")
        points1 = np.random.random((n,2))
        points2 = np.random.random((n,2))+.5
        print(test1(points1, points2))
        print(test2(points1, points2))
        
        print("\nhalf overlap")
        points1 = np.random.random((n,2))
        points2 = np.random.random((n,2))
        points2[:,0] += .5
        print(test1(points1, points2))
        print(test2(points1, points2))
        
        print("\nfully inside")
        points1 = np.random.random((n,2))
        points2 = np.random.random((n,2))*4-1
        print(test1(points1, points2))
        print(test2(points1, points2))
        input("continue...")
        n <<= 1
    pass
