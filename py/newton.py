def derivative(f, x, h=1e-7):
    return (f(x + h) - f(x)) / h

def f(x):
    return x**3 - x + 2

def derivative_central(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

def partial_derivative_x(f, x, y, h=1e-7):
    return (f(x + h, y) - f(x, y)) / h 

def partial_derivative_y(f, x, y, h=1e-7):
    return (f(x, y + h) - f(x, y)) / h

print(derivative(f, 2))
print(derivative_central(f, 2))


def g(x, y):
    return x**2 + y**2 - 4
print(partial_derivative_x(g, 1, 1))
print(partial_derivative_y(g, 1, 1))



def gradient(f, x, h=1e-7):
    """
    Computes gradient of f at point x 
    f: function takes a list or array and returns a scalar
    x: point (list or array) at which to compute gradient
    Returns: list of partial derivatives
    """
    n = len(x)
    grad = []

    for i in range(n):
        x_plus = list(x)
        x_plus[i] += h 

        df_dxi = (f(x_plus) - f(x)) / h 
        grad.append(df_dxi)
    
    return grad

def k(x):
    return x[0]**2 + x[1]**2 + x[2]**2

point = [1.0, 2.0, 3.0]
grad_f = gradient(k, point)

print("Gradient of k at", point, "is", grad_f)

# TODO: Implement Newton's method for multivariate functions start with those initial values
(x0,y0) = (0.6381939926271558578, -0.2120300331658224337)





