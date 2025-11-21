from typing import Tuple, List, Optional
import numpy as np 
from scipy.linalg import solve, norm

# TODO: Implement Newton's method for multivariate functions start with those initial values
(x0,y0) = (0.6381939926271558578, -0.2120300331658224337)


A = 1.4
B = 0.3

MAX_ITER = 100
TOLERANCE = 1e-7
VERIFICATION_TOLERANCE = 1e-8


def henon(x: float, y: float, a: float = A, b: float = B) -> Tuple[float, float]:
    """Compute the Henon map."""
    x_new = 1 - a * x**2 + y
    y_new = b * x
    return x_new, y_new 

def henon_jacobian(x: float, y: float, a: float = A, b: float = B) -> List[List[float]]:
    return [
        [-2 * a * x, 1],
        [b, 0]
    ] 

def compute_n_iterate(
    x0: float,
    y0: float,
    n: int,
    a: float = A,
    b: float = B
) -> Tuple[float, float]:
    x, y = x0, y0 
    for _ in range(n):
        x, y = henon(x, y, a, b)
    return x, y


def compute_jaccobian_n_iterate(
    x0: float,
    y0: float,
    n: int,
    a: float = A,
    b: float = B 
) -> List[List[float]]:
    J_total = np.eye(2)
    x, y = x0, y0
    for _ in range(n):
        J_current = henon_jacobian(x, y, a, b)
        J_total = np.dot(J_current, J_total)
        x, y = henon(x, y, a, b)
    return J_total.tolist()

def newton_periodic_orbit(
    x0: float,
    y0: float,
    period: int,
    a: float =A,
    b: float =B,
    max_iter: int = MAX_ITER,
    tol: float = TOLERANCE
) -> Tuple[Optional[float], Optional[float], bool, int]:
    """
    Use Newton's method to find a periodic orbit of given period.
    
    We solve: F(x, y) = f^(n)(x, y) - (x, y) = (0, 0)
    
    Newton iteration:
        (x, y)_new = (x, y)_old - J_F^(-1) * F(x, y)_old
    
    where J_F = J_f^(n) - I
    
    Args:
        x0, y0: Initial guess
        period: Desired period
        a, b: Hénon parameters
        max_iter: Maximum Newton iterations
        tol: Convergence tolerance
        verbose: Print iteration details
        
    Returns:
        (x, y, converged, iterations)
        - x, y: Periodic point (if converged)
        - converged: Whether Newton's method converged
        - iterations: Number of iterations performed
    """
    x, y = x0, y0
    for iteration in range(max_iter):
        # Compute f^n(x, y)
        fx, fy = compute_n_iterate(x, y, period, a, b)
        
        # Compute F(x, y) = f^n(x, y) - (x, y)
        Fx = fx - x 
        Fy = fy - y 
        F = np.array([Fx, Fy])
        
        # Check if we are already at the solution 
        residual = norm(F) 

        if residual < tol: 
            return x, y, True, iteration
        
        J_fn = compute_jaccobian_n_iterate(x, y, period, a, b)
        
        # Compute Jaccobian of F(x, y) = f^n(x, y) - (x, y)
        # J_F = J_f^(n) - I 
        J_F = np.array(J_fn) - np.eye(2)
        if abs(np.linalg.det(J_F)) < 1e-14:
            # Jacobian is singular, cannot proceed
            return None, None, False, iteration
        try: 
            delta = solve(J_F, -F)
        except np.linalg.LinAlgError:
            return None, None, False, iteration

        x += delta[0]
        y += delta[1]
        
        step_size = norm(delta)
        
        if step_size < tol: 
            return x, y, True, iteration + 1
    return None, None, False, max_iter

def verify_periodic_orbit(
    x: float,
    y: float, 
    period: int,
    a: float = A,
    b: float = B, 
    tol: float = VERIFICATION_TOLERANCE
) -> bool:  
    x0, y0 = x, y
    for _ in range(period):
        x, y = henon(x, y, a, b)
    dist = np.sqrt((x - x0)**2 + (y - y0)**2)
    return dist < tol 



def classify_periodic_orbit(
    x: float,
    y: float,
    period: int,
    a: float = A,
    b: float = B
) -> Tuple[str, complex, complex]:
    """
    Classify the stabability of the periodic orbit using eigenvalues
    For a periodic orbit, stability is determined by eigenvalues of J_f^n:
    - Stable: |lambda1|, |lambda2| < 1
    - Unstable: |lambda1|, |lambda2| > 1
    - Saddle: One lambda < 1 and one lambda > 1
    """
    J = compute_jaccobian_n_iterate(x, y, period, a, b)
    
    eigenvalues = np.linalg.eigvals(J)
    lambda1, lambda2 = eigenvalues[0], eigenvalues[1]
    
    # compute magnitudes 
    mag1 = abs(lambda1)
    mag2 = abs(lambda2)
    
    if mag1 < 1 and mag2 < 1:
        if np.isreal(lambda1) and np.isreal(lambda2):
            classfication = "Stable node"
        else:
            classification = "Stable spiral"
    elif (mag1 < 1 and mag2) > 1 or (mag1 > 1 and mag2 < 1):
        classification = "Saddle"
    elif mag1 > 1 and mag2 > 1:
        if np.isreal(mag1) and np.isreal(mag2):
            classfication = "Unstable Node"
        else:
            classification = "Unstable spiral"
    else: 
        classification = "Center (marginally stable)"
    return classification, lambda1, lambda2
    
        
    





