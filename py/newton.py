from typing import Tuple, List, Optional
import numpy as np 
from scipy.linalg import solve, norm
import matplotlib.pyplot as plt

# TODO: Implement Newton's method for multivariate functions start with those initial values
(x0,y0) = (0.6381939926271558578, -0.2120300331658224337)


A = 1.4
B = 0.3

MAX_ITER = 100
TOLERANCE = 1e-7
VERIFICATION_TOLERANCE = 1e-8
GRID_SIZE = 50
X_MIN, X_MAX = -2.0, 2.0
Y_MIN, Y_MAX = -2.0, 2.0
DUPLICATE_TOL = 1e-6


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
        if not np.isfinite(x) or not np.isfinite(y):
            return None, None, False, iteration
        fx, fy = compute_n_iterate(x, y, period, a, b)
        
        # Compute F(x, y) = f^n(x, y) - (x, y)
        Fx = fx - x 
        Fy = fy - y 
        F = np.array([Fx, Fy])
        
        if not np.all(np.isfinite(F)):
            return None, None, False, iteration
        
        # Check if we are already at the solution 
        residual = norm(F) 

        if residual < tol: 
            return x, y, True, iteration
        
        J_fn = compute_jaccobian_n_iterate(x, y, period, a, b)

        if not np.all(np.isfinite(J_fn)):
            return None, None, False, iteration
        
        
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
        if not np.all(np.isfinite(delta)):
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
    """
    Verify that (x, y) is a periodic point of EXACTLY the given period.
    Returns False if it's a lower-period orbit.
    """
    x_orig, y_orig = x, y
    
    # Check all proper divisors of period
    for divisor in range(1, period):
        if period % divisor == 0:  # Only check if divisor divides period
            x_test, y_test = x, y
            for _ in range(divisor):
                x_test, y_test = henon(x_test, y_test, a, b)
            
            dist = np.sqrt((x_test - x_orig)**2 + (y_test - y_orig)**2)
            if dist < tol:
                return False
    
    # Now check if it's actually period-n
    x_test, y_test = x, y
    for _ in range(period):
        x_test, y_test = henon(x_test, y_test, a, b)
    
    dist = np.sqrt((x_test - x_orig)**2 + (y_test - y_orig)**2)
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
    
def find_periodic_orbits_grid(
    period: int,
    a: float = A,
    b: float = B,
    grid_size: int = GRID_SIZE,
    x_range: Tuple[float, float] = (X_MIN, X_MAX),
    y_range: Tuple[float, float] = (Y_MIN, Y_MAX),
    dup_tol: float = DUPLICATE_TOL,
) -> List[Tuple[float, float]]:
    """
    Search for periodic orbits using a grid of initial guesses.
    Returns one representative point per orbit (not all points in each orbit).
    """
    found_orbits = []
    
    x_min, x_max = x_range
    y_min, y_max = y_range

    for i in range(grid_size):
        for j in range(grid_size):
            x0 = x_min + (x_max - x_min) * i / (grid_size - 1)
            y0 = y_min + (y_max - y_min) * j / (grid_size - 1)
            
            x, y, converged, iters = newton_periodic_orbit(x0, y0, period, a, b) 
            if not converged:
                continue
            
            if not verify_periodic_orbit(x, y, period, a, b):
                continue
            
            is_new_orbit = True
            
            for (x_old, y_old) in found_orbits:
                x_test, y_test = x_old, y_old
                for k in range(period):
                    dist = np.sqrt((x - x_test)**2 + (y - y_test)**2)
                    if dist < dup_tol:
                        is_new_orbit = False
                        break
                    x_test, y_test = henon(x_test, y_test, a, b)
                
                if not is_new_orbit:
                    break
            
            if is_new_orbit:
                found_orbits.append((x, y))
                
    return found_orbits

def analyze_orbit(
    x: float,
    y: float,
    period: int,
    a: float = A,
    b: float = B
) -> None:
    print(f"\n{'='*60}")
    print(f"PERIODIC ORBIT ANALYSIS")
    print(f"{'='*60}")
    print(f"Period: {period}")
    print(f"Point: ({x:.10f}, {y:.10f})")

    is_periodic = verify_periodic_orbit(x, y, period, a, b)
    print(f"Verified: {is_periodic}")

    # classify stability 
    classification, lambda1, lambda2 = classify_periodic_orbit(x, y, period, a, b)
    print(f"\nStability: {classification}")
    print(f"Eigenvalues:")
    print(f"  λ₁ = {lambda1:.6f}, |λ₁| = {abs(lambda1):.6f}")
    print(f"  λ₂ = {lambda2:.6f}, |λ₂| = {abs(lambda2):.6f}")

    print(f"\nFull orbit (all {period} points):")
    x_cur, y_cur = x, y
    for i in range(period): 
        print(f"  Point {i}: ({x_cur:.10f}, {y_cur:.10f})")
        x_cur, y_cur = henon(x_cur, y_cur, a, b)
    
    print(f"  Point {period} (returns to start): ({x_cur:.10f}, {y_cur:.10f})")
    
    dist = np.sqrt((x_cur - x)**2 + (y_cur - y)**2)
    print(f"  Distance from Point 0: {dist:.2e}")
    
    print(f"{'='*60}\n")
    
    
def plot_periodic_orbits(
    orbits: List[Tuple[float, float]],
    period: int,
    a: float = A,
    b: float = B,
    show_attractor: bool = True
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if show_attractor:
        x, y = 0.1, 0.1
        xs, ys = [], []
        for _ in range(10000):
            x, y = henon(x, y, a, b)
            if _ > 100:
                xs.append(x)
                ys.append(y)
        ax.scatter(xs, ys, s=0.1, c='lightgray', alpha=0.5, label='Attractor')
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(orbits)))
    
    for idx, (x0, y0) in enumerate(orbits):
        # Compute all points in the periodic orbit
        xs, ys = [], []
        x, y = x0, y0
        for i in range(period):
            xs.append(x)
            ys.append(y)
            x, y = henon(x, y, a, b)
        
        ax.scatter(xs, ys, s=100, c=[colors[idx]], marker='o', 
                  edgecolors='black', linewidths=2, 
                  label=f'Orbit {idx+1}', zorder=5)
        
        xs_cycle = xs + [xs[0]]  # Close the loop
        ys_cycle = ys + [ys[0]]
        ax.plot(xs_cycle, ys_cycle, 'o-', color=colors[idx], 
               linewidth=2, markersize=8, alpha=0.7)
        
        ax.plot(x0, y0, 's', color=colors[idx], markersize=12, 
               markeredgecolor='black', markeredgewidth=2, zorder=6)
        
        for i, (px, py) in enumerate(zip(xs, ys)):
            ax.annotate(f'{i}', (px, py), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Period-{period} Orbits of Hénon Map (a={a}, b={b})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()   
    
def main():
    import warnings 
    warnings.filterwarnings('ignore')
    print("\n" + "="*60)
    print("HÉNON MAP PERIODIC ORBIT FINDER")
    print("="*60)
    print(f"Parameters: a = {A}, b = {B}")
    print("="*60 + "\n")
    
    # Example 1: Find period-1 orbits (fixed points)
    print("\n" + "─"*60)
    print("EXAMPLE 1: Finding period-1 orbits (fixed points)")
    print("─"*60)
    
    period1_orbits = find_periodic_orbits_grid(
        period=1, 
        grid_size=30,
    )
    
    if period1_orbits:
        print(f"\nAnalyzing first period-1 orbit:")
        analyze_orbit(period1_orbits[0][0], period1_orbits[0][1], 1)
    
    # Example 2: Find period-2 orbits
    print("\n" + "─"*60)
    print("EXAMPLE 2: Finding period-2 orbits")
    print("─"*60)
    
    period2_orbits = find_periodic_orbits_grid(
        period=2,
        grid_size=40,
    )
    
    if period2_orbits:
        print(f"\nAnalyzing first period-2 orbit:")
        analyze_orbit(period2_orbits[0][0], period2_orbits[0][1], 2)
    
    # Example 3: Find period-3 orbits
    print("\n" + "─"*60)
    print("EXAMPLE 3: Finding period-3 orbits")
    print("─"*60)
    
    period3_orbits = find_periodic_orbits_grid(
        period=3,
        grid_size=50,
    )
    
    if period3_orbits:
        print(f"\nAnalyzing first period-3 orbit:")
        analyze_orbit(period3_orbits[0][0], period3_orbits[0][1], 3)
    
    # Example 3: Find period-3 orbits
    print("\n" + "─"*60)
    print("EXAMPLE 4: Finding period-4 orbits")
    print("─"*60)
    
    period4_orbits = find_periodic_orbits_grid(
        period=4,
        grid_size=50,
    )
    
    if period4_orbits:
        print(f"\nAnalyzing first period-3 orbit:")
        analyze_orbit(period4_orbits[0][0], period4_orbits[0][1], 4)
    
    for i in range(5, 10):
        periodic_i_orbit = find_periodic_orbits_grid(period=i, grid_size=50)
        if periodic_i_orbit:
            print(f"Found {i}-periodic orbit!")
            print(f"Analyzing {i}-period orbit")
            analyze_orbit(periodic_i_orbit[0][0], periodic_i_orbit[0][1], i)
    
    # Visualization
    print("\n" + "─"*60)
    print("VISUALIZATION")
    print("─"*60)
    
    if period1_orbits:
        print("Plotting period-1 orbits...")
        plot_periodic_orbits(period1_orbits, 1)
    
    if period2_orbits:
        print("Plotting period-2 orbits...")
        plot_periodic_orbits(period2_orbits, 2)
    
    if period3_orbits:
        print("Plotting period-3 orbits...")
        plot_periodic_orbits(period3_orbits, 3)
    if period4_orbits:
        plot_periodic_orbits(period4_orbits, 4)

if __name__ == '__main__':
    main()
     
    

            




