from typing import Tuple, List, Optional
import numpy as np 
from scipy.linalg import solve, norm
import matplotlib.pyplot as plt
from dataclasses import dataclass
import scipy.sparse as sparse

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



# Monte Carlo simulation 
@dataclass 
class Particle:
    x: float
    y: float

    def __repr__(self):
        return f"Particle(x={self.x:.4f}, y={self.y:.4f})"
    
class ProbabilityGrid:
    """
    Represents 2d grid that discretizes the phase space into cells
    """

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, nx_cells: int, ny_cells: int):
        """
        Args:
            x_min, x_max: Range of x coordinates to cover
            y_min, y_max: Range of y coordinates to cover
            nx_cells: Number of cells in the x direction
            ny_cells: Number of cells in the y direction
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.nx_cells = nx_cells
        self.ny_cells = ny_cells

        self.cell_width = (x_max - x_min) / nx_cells
        self.cell_height = (y_max - y_min) / ny_cells

        self.probabilities = np.zeros(nx_cells * ny_cells)

    def point_to_cell(self, x: float, y: float) -> Optional[int]:
        """
        Map a point (x, y) to a cell index in the grid.
        Returns None if the point is out of bounds.
        """
        if x < self.x_min or x >= self.x_max or y < self.y_min or y >= self.y_max:
            return None
        i = int((x - self.x_min) / self.cell_width)
        j = int((y - self.y_min) / self.cell_height)

        # handle edge case where the point is exactly on the max boundary
        i = min(i, self.nx_cells - 1)
        j = min(j, self.ny_cells - 1)
        # convert 2d indices to 1d index, row-major order
        cell_index = j * self.nx_cells + i
        return cell_index

    def cell_to_indices(self, cell_index: int) -> Tuple[int, int]:
        # Math: if cell_index = j * nx_cells + i, then:
        i = cell_index % self.nx_cells 
        j = cell_index // self.nx_cells
        return i, j
    
    def cell_center(self, cell_index:int) -> Tuple[float, float]: 
        i, j = self.cell_to_indices(cell_index)

        x_center = self.x_min + (i + 0.5) * self.cell_width
        y_center = self.y_min + (j + 0.5) * self.cell_height
        return x_center, y_center
    
    
    def get_cell_bounds(self, cell_index: int) -> Tuple[float, float, float, float]:
        i, j = self.cell_to_indices(cell_index)
        x_min_cell = self.x_min + i * self.cell_width
        x_max_cell = x_min_cell + self.cell_width
        y_min_cell = self.y_min + j * self.cell_height
        y_max_cell = y_min_cell + self.cell_height
        return x_min_cell, x_max_cell, y_min_cell, y_max_cell
        


class MonteCarloSimulation:
    def __init__(self, henon_a: float, henon_b: float, noise_radius: float, num_particles: int, grid: ProbabilityGrid):
        self.henon_a = henon_a
        self.henon_b = henon_b
        self.noise_radius = noise_radius
        self.num_particles = num_particles
        self.grid = grid
        self.particles: List[Particle] = []
        
        self.current_iteration = 0
        self.probability_history = []
        self.rng = np.random.default_rng()
        
    def sample_uniform_circle(self, radius: float) -> Tuple[float, float]:
        theta = self.rng.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(self.rng.uniform(0, 1))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return x, y 
    
    def initialize_particles(self, init_type: str, x0: float = 0.0, y0: float = 0.0, radius: float = 0.1):
        self.particles = []
        self.current_iteration = 0
        self.probability_history = []
        
        
        if init_type == 'point': 
            for _ in range(self.num_particles):
                self.particles.append(Particle(x0, y0))
        elif init_type == 'circle': 
            for _ in range(self.num_particles):
                dx, dy = self.sample_uniform_circle(radius)
                self.particles.append(Particle(x0 + dx, y0 + dy))
        elif init_type == 'grid':
            for _ in range(self.num_particles):
                x = self.rng.uniform(self.grid.x_min, self.grid.x_max)
                y = self.rng.uniform(self.grid.y_min, self.grid.y_max)
                self.particles.append(Particle(x, y))
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        self.compute_probabilities()
        self.probability_history.append(self.grid.probabilities.copy())
        
    def step_forward(self):
        for particle in self.particles: 
            x_new = 1 - self.henon_a * particle.x**2 + particle.y
            y_new = self.henon_b * particle.x
            
            noise_x, noise_y = self.sample_uniform_circle(self.noise_radius)

            # this models the bounded noise: x_{n+1} = f(x_n) + θ where ||θ|| ≤ ε
            particle.x = x_new + noise_x 
            particle.y = y_new + noise_y 
        
        self.current_iteration += 1
        self.compute_probabilities()
        self.probability_history.append(self.grid.probabilities.copy())
    
    def compute_probabilities(self):
        """
        This implements the Monte Carlo estimation 
        p_i = (number of particles in cell i) / (total number of particles)

        This is approximating the intergal: p_i = ∫_{C_i} ρ(x,y) dx dy
        by sampling: if we have N particles distributed according to p(x, y), then
        the fraction landing in cell_i converges to p_i as N → ∞
        """

        self.grid.probabilities.fill(0)
        
        valid_particles = 0 # track the particles that are in the valid grid area
        for particle in self.particles:
            cell_index = self.grid.point_to_cell(particle.x, particle.y)
            
            if cell_index is not None: 
                self.grid.probabilities[cell_index] += 1
                valid_particles += 1
        # normalize by the total number of particles to convert counts to probabilities
        if valid_particles > 0:
            self.grid.probabilities /= valid_particles
        
        
    def probability_in_region(self, x_min: float, x_max:float, y_min:float, y_max:float) -> float: 
        """
        Calculate the probability of the particle being in this particular retangular region
        """

        total_prob = 0.0
        for cell_index in range(len(self.grid.probabilities)):
            cell_x_min, cell_x_max, cell_y_min, cell_y_max = self.grid.get_cell_bounds(cell_index)
            overlap = not (
                cell_x_max <= x_min or # cell is to the left 
                cell_x_min >= x_max or # cell is to the right 
                cell_y_max <= y_min or # cell is below 
                cell_y_min >= y_max # cell is above
            ) 

            if overlap: 
                total_prob += self.grid.probabilities[cell_index]
        return total_prob

    def run_simulation(self, num_steps: int, verbose:bool = True):
        for iteration in range(num_steps):
            self.step_forward()
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Completed iteration {self.current_iteration}")
    
    def get_probability_at_cell(self, cell_index: int) -> float:
        return self.grid.probabilities[cell_index]

    def find_cell_at_point(self, x: float, y: float) -> Optional[int]:
        return self.grid.point_to_cell(x, y)



def plot_probability_heatmap(sim: MonteCarloSimulation, title: str):
    """
    This creates a color-coded image where bright regions indicate high
    probability (many particles) and dark regions indicate low probability
    (few particles).
    """ 

    prob_2d = sim.grid.probabilities.reshape((sim.grid.ny_cells, sim.grid.nx_cells))

    
    fig, ax = plt.subplots(figsize=(10,8))
    
    
    im = ax.imshow(
        prob_2d,
        extent=(sim.grid.x_min, sim.grid.x_max, sim.grid.y_min, sim.grid.y_max),
        origin='lower',
        cmap='hot',
        interpolation='nearest',
        aspect='auto'
    )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Probability', rotation=270, labelpad=20)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{title}\nIteration {sim.current_iteration}, '
                 f'{sim.num_particles} particles, ε={sim.noise_radius}')
    
    plt.tight_layout()
    plt.show()


def plot_probability_evolution(
    sim: MonteCarloSimulation,
    cell_indices: List[int],
    cell_labels: Optional[List[str]] = None
):
    """
    Plot how probability in specific cells changes over iterations.
    This creates a line graph showing p_i(n) versus n for selected cells.
    It helps visualize the convergence to the invariant measure.
    """
    if cell_labels is None:
        cell_labels = [f'Cell {i}' for i in cell_indices]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract probability history for each selected cell
    for idx, cell_index in enumerate(cell_indices):
        # Get the center coordinates of this cell for the label
        x_center, y_center = sim.grid.cell_center(cell_index)
        
        # Extract probabilities across all iterations
        probs_over_time = [
            prob_dist[cell_index] for prob_dist in sim.probability_history
        ]
        
        # Plot the evolution
        ax.plot(
            range(len(probs_over_time)),
            probs_over_time,
            marker='o',
            label=f'{cell_labels[idx]} at ({x_center:.2f}, {y_center:.2f})'
        )
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Probability')
    ax.set_title(f'Probability Evolution (ε={sim.noise_radius})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_particles_and_grid(
    sim: MonteCarloSimulation,
    show_particles: bool = True,
    show_grid: bool = True
):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot probability heatmap in the background
    prob_2d = sim.grid.probabilities.reshape(
        (sim.grid.ny_cells, sim.grid.nx_cells)
    )
    ax.imshow(
        prob_2d,
        origin='lower',
        extent=[sim.grid.x_min, sim.grid.x_max, sim.grid.y_min, sim.grid.y_max],
        cmap='YlOrRd',
        alpha=0.6,
        aspect='auto'
    )
    
    # Overlay particle positions if requested
    if show_particles:
        x_coords = [p.x for p in sim.particles]
        y_coords = [p.y for p in sim.particles]
        ax.scatter(x_coords, y_coords, s=1, c='blue', alpha=0.5, label='Particles')
    
    # Draw grid lines if requested
    if show_grid:
        # Vertical lines
        for i in range(sim.grid.nx_cells + 1):
            x = sim.grid.x_min + i * sim.grid.cell_width
            ax.axvline(x, color='gray', linewidth=0.5, alpha=0.3)
        
        # Horizontal lines
        for j in range(sim.grid.ny_cells + 1):
            y = sim.grid.y_min + j * sim.grid.cell_height
            ax.axhline(y, color='gray', linewidth=0.5, alpha=0.3)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Particles and Probability Grid\n'
                 f'Iteration {sim.current_iteration}, {sim.num_particles} particles')
    ax.legend()
    
    plt.tight_layout()
    plt.show() 



def example_1_single_point_initial():
    """
    Start all particles at a single point and watch uncertainty spread.
    
    This demonstrates how noise causes an initially precise state to disperse
    over iterations. Even though all particles start at the same location, they
    quickly diverge due to:

    1. Different noise realizations
    2. Chaotic dynamics of the Hénon map (sensitive dependence)
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Point Initial Condition")
    print("="*60)
    print("\nStarting all 10,000 particles at (0.1, 0.1)")
    print("Noise radius ε = 0.05")
    print("This shows how uncertainty spreads from a precise initial condition.\n")
    
    # Create the probability grid 40x40 cells over [-2, 2] x [-2, 2]
    grid = ProbabilityGrid(
        x_min=-2.0, x_max=2.0,
        y_min=-2.0, y_max=2.0,
        nx_cells=40,
        ny_cells=40
    )
    
    # Create the Monte Carlo simulation
    sim = MonteCarloSimulation(
        henon_a=A,
        henon_b=B,
        noise_radius=0.05,
        num_particles=10000,
        grid=grid
    )
    
    # Initialize all particles at a single point
    sim.initialize_particles(init_type='point', x0=0.1, y0=0.1)
    
    print(f"Initial state:")
    print(f"  All particles at: (0.1, 0.1)")
    
    # Find which cell this point is in
    initial_cell = sim.find_cell_at_point(0.1, 0.1)
    if initial_cell is not None:
        prob = sim.get_probability_at_cell(initial_cell)
        print(f"  Cell index: {initial_cell}")
        print(f"  Probability in that cell: {prob:.4f} (should be ~1.0)")
    
    # Run for 50 iterations
    print(f"\nRunning simulation for 50 iterations...")
    sim.run_simulation(num_steps=1000, verbose=True)
    
    # Analyze the final state
    print(f"\nFinal state (iteration {sim.current_iteration}):")
    
    # Check probability in a specific region (e.g., near the attractor)
    prob_attractor = sim.probability_in_region(-1.5, 1.5, -0.5, 0.5)
    print(f"  Probability in region [-1.5, 1.5] × [-0.5, 0.5]: {prob_attractor:.4f}")
    
    # Find the cell with maximum probability
    max_prob_cell = np.argmax(sim.grid.probabilities)
    max_prob = sim.grid.probabilities[max_prob_cell]
    max_prob_x, max_prob_y = sim.grid.cell_center(max_prob_cell)
    print(f"  Highest probability cell: {max_prob:.4f} at ({max_prob_x:.2f}, {max_prob_y:.2f})")
    
    # Visualize
    print("\nGenerating visualizations...")
    plot_probability_heatmap(sim, "Single Point Initial Condition")
    plot_particles_and_grid(sim, show_particles=True, show_grid=False)
    
    # Track a few specific cells over time
    interesting_cells = [
        initial_cell,  # Where we started
        max_prob_cell,  # Where most probability ended up
    ]
    plot_probability_evolution(
        sim,
        interesting_cells,
        ['Initial cell', 'Final max prob cell']
    )

def interactive_henon_stepper():
    print("\n" + "="*60)
    print("INTERACTIVE HÉNON MAP STEPPER")
    print("="*60)
    
    # Create a smaller, focused grid around the attractor region
    # The Hénon attractor lives roughly in [-1.5, 1.5] × [-0.4, 0.4]
    grid = ProbabilityGrid(
        x_min=-1.5, x_max=1.5,
        y_min=-0.5, y_max=0.5,
        nx_cells=30,  # 30x30 = 900 cells
        ny_cells=30
    )
    
    # Create simulation with reasonable parameters
    sim = MonteCarloSimulation(
        henon_a=A,
        henon_b=B,
        noise_radius=0.02,  # Smaller noise
        num_particles=1,  
        grid=grid
    )
    
    sim.initialize_particles(init_type='point', x0=0.6, y0=-0.2)
    
    print(f"\nInitialized {sim.num_particles} particles at (0.6, -0.2)")
    print(f"Grid: {grid.nx_cells}×{grid.ny_cells} cells")
    print(f"Grid covers: x ∈ [{grid.x_min}, {grid.x_max}], y ∈ [{grid.y_min}, {grid.y_max}]")
    print(f"Noise radius: ε = {sim.noise_radius}")
    
    print(f"\n{'='*60}")
    print(f"ITERATION 0 (Initial State)")
    print(f"{'='*60}")
    show_current_state(sim)
    plot_probability_heatmap(sim, "Iteration 0")
    
    iteration = 0
    max_iterations = 100  
    
    while iteration < max_iterations:
        print(f"\n{'─'*60}")
        user_input = input("Press ENTER to step forward, 'q' to quit, or number of steps: ").strip()
        
        if user_input.lower() == 'q':
            print("Exiting...")
            break
        
        # Determine how many steps to take
        if user_input == '':
            steps = 1
        else:
            try:
                steps = int(user_input)
                steps = max(1, min(steps, 50))  # Between 1 and 50
            except ValueError:
                steps = 1
        
        # Take the steps
        for _ in range(steps):
            sim.step_forward()
            iteration += 1
        
        # Show the new state
        print(f"\n{'='*60}")
        print(f"ITERATION {sim.current_iteration}")
        print(f"{'='*60}")
        show_current_state(sim)
        plot_probability_heatmap(sim, f"Iteration {sim.current_iteration}")
    
    print("\n" + "="*60)
    print("SESSION COMPLETE")
    print("="*60)


def show_current_state(sim: MonteCarloSimulation):
    """
    Display key statistics about the current state of the simulation.
    """
    # Count how many particles are still in bounds
    particles_in_grid = sum(
        1 for p in sim.particles 
        if sim.grid.point_to_cell(p.x, p.y) is not None
    )
    
    print(f"Particles in grid: {particles_in_grid}/{sim.num_particles} "
          f"({100*particles_in_grid/sim.num_particles:.1f}%)")
    
    if particles_in_grid == 0:
        print("The system has diverged.")
        return
    
    # Find the cell with maximum probability
    max_prob_cell = np.argmax(sim.grid.probabilities)
    max_prob = sim.grid.probabilities[max_prob_cell]
    max_prob_x, max_prob_y = sim.grid.cell_center(max_prob_cell)
    
    print(f"Maximum probability: {max_prob:.6f} at cell ({max_prob_x:.3f}, {max_prob_y:.3f})")
    
    # Show the top 5 cells by probability
    top_cells = np.argsort(sim.grid.probabilities)[-5:][::-1]
    print(f"\nTop 5 cells by probability:")
    for rank, cell_idx in enumerate(top_cells, 1):
        prob = sim.grid.probabilities[cell_idx]
        cx, cy = sim.grid.cell_center(cell_idx)
        if prob > 0:
            print(f"  {rank}. Cell at ({cx:6.3f}, {cy:6.3f}): p = {prob:.6f}")
    
    # Calculate entropy as a measure of spread
    probs = sim.grid.probabilities[sim.grid.probabilities > 0]
    if len(probs) > 0:
        entropy = -np.sum(probs * np.log(probs))
        print(f"\nEntropy: {entropy:.4f} (higher = more spread out)")
    
    # Show probability in the attractor region
    prob_core = sim.probability_in_region(-1.0, 1.0, -0.4, 0.4)
    print(f"Probability in core region [-1, 1] × [-0.4, 0.4]: {prob_core:.4f}")


def quick_demo():
    """
    Quick non-interactive demo showing the first few iterations.
    """
    print("\n" + "="*60)
    print("QUICK DEMO: First 10 Iterations")
    print("="*60)
    
    grid = ProbabilityGrid(
        x_min=-1.5, x_max=1.5,
        y_min=-0.5, y_max=0.5,
        nx_cells=30,
        ny_cells=30
    )
    
    sim = MonteCarloSimulation(
        henon_a=A,
        henon_b=B,
        noise_radius=0.02,
        num_particles=5000,
        grid=grid
    )
    
    # Start on the attractor
    sim.initialize_particles(init_type='point', x0=0.6, y0=-0.2)
    
    # Show iterations 0, 1, 2, 5, 10
    for target_iter in [0, 1, 2, 5, 10]:
        while sim.current_iteration < target_iter:
            sim.step_forward()
        
        print(f"\n{'='*60}")
        print(f"ITERATION {sim.current_iteration}")
        print(f"{'='*60}")
        show_current_state(sim)
        plot_probability_heatmap(sim, f"Iteration {sim.current_iteration}")



    
    

def main():
    print("\n" + "="*70)
    print("HÉNON MAP: PERIODIC ORBITS + MONTE CARLO PROBABILITY ESTIMATION")
    print("="*70)
    
    print("\n" + "─"*70)
    print("PART 1: PERIODIC ORBIT ANALYSIS")
    print("─"*70)
    print("\nFinding period-1 orbits (fixed points)...")
    period1_orbits = find_periodic_orbits_grid(period=1, grid_size=30)
    
    if period1_orbits:
        print(f"Found {len(period1_orbits)} period-1 orbit(s)")
        x, y = period1_orbits[0]
        analyze_orbit(x, y, 1)
        plot_periodic_orbits(period1_orbits, 1)
    
    # Now run Monte Carlo examples
    print("\n" + "─"*70)
    print("PART 2: MONTE CARLO PROBABILITY ESTIMATION")
    
    # Run all Monte Carlo examples
    example_1_single_point_initial()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETE")
    print("="*70)



def interactive_henon_stepper():
    print("\n" + "="*60)
    print("INTERACTIVE HÉNON MAP STEPPER")
    print("="*60)
    
    # Create a smaller, focused grid around the attractor region
    # The Hénon attractor lives roughly in [-1.5, 1.5] × [-0.4, 0.4]
    grid = ProbabilityGrid(
        x_min=-1.5, x_max=1.5,
        y_min=-0.5, y_max=0.5,
        nx_cells=500,  
        ny_cells=500
    )
    
    sim = MonteCarloSimulation(
        henon_a=A,
        henon_b=B,
        noise_radius=0.02,  
        num_particles=10_000, 
        grid=grid
    )
    
    # Start near a known point on the attractor
    sim.initialize_particles(init_type='point', x0=0.6, y0=-0.2)
    
    print(f"\nInitialized {sim.num_particles} particles at (0.6, -0.2)")
    print(f"Grid: {grid.nx_cells}×{grid.ny_cells} cells")
    print(f"Grid covers: x ∈ [{grid.x_min}, {grid.x_max}], y ∈ [{grid.y_min}, {grid.y_max}]")
    print(f"Noise radius: ε = {sim.noise_radius}")
    
    # Show initial state
    print(f"\n{'='*60}")
    print(f"ITERATION 0 (Initial State)")
    print(f"{'='*60}")
    show_current_state(sim)
    plot_probability_heatmap(sim, "Iteration 0")
    
    iteration = 0
    max_iterations = 1000 
    
    while iteration < max_iterations:
        print(f"\n{'─'*60}")
        user_input = input("Press ENTER to step forward, 'q' to quit, or number of steps: ").strip()
        
        if user_input.lower() == 'q':
            print("Exiting...")
            break
        
        if user_input == '':
            steps = 1
        else:
            try:
                steps = int(user_input)
                steps = max(1, min(steps, 500)) 
            except ValueError:
                steps = 1
        
        # Take the steps
        for _ in range(steps):
            sim.step_forward()
            iteration += 1
        
        # Show the new state
        print(f"\n{'='*60}")
        print(f"ITERATION {sim.current_iteration}")
        print(f"{'='*60}")
        show_current_state(sim)
        plot_probability_heatmap(sim, f"Iteration {sim.current_iteration}")
    
    print("\n" + "="*60)
    print("SESSION COMPLETE")
    print("="*60)


def show_current_state(sim: MonteCarloSimulation):
    # Count how many particles are still in bounds
    particles_in_grid = sum(
        1 for p in sim.particles 
        if sim.grid.point_to_cell(p.x, p.y) is not None
    )
    
    print(f"Particles in grid: {particles_in_grid}/{sim.num_particles} "
          f"({100*particles_in_grid/sim.num_particles:.1f}%)")
    
    if particles_in_grid == 0:
        print("   The system has diverged.")
        return
    
    # Find the cell with maximum probability
    max_prob_cell = np.argmax(sim.grid.probabilities)
    max_prob = sim.grid.probabilities[max_prob_cell]
    max_prob_x, max_prob_y = sim.grid.cell_center(max_prob_cell)
    
    print(f"Maximum probability: {max_prob:.6f} at cell ({max_prob_x:.3f}, {max_prob_y:.3f})")
    
    # Show the top 5 cells by probability
    top_cells = np.argsort(sim.grid.probabilities)[-5:][::-1]
    print(f"\nTop 5 cells by probability:")
    for rank, cell_idx in enumerate(top_cells, 1):
        prob = sim.grid.probabilities[cell_idx]
        cx, cy = sim.grid.cell_center(cell_idx)
        if prob > 0:
            print(f"  {rank}. Cell at ({cx:6.3f}, {cy:6.3f}): p = {prob:.6f}")
    
    # Calculate entropy as a measure of spread
    probs = sim.grid.probabilities[sim.grid.probabilities > 0]
    if len(probs) > 0:
        entropy = -np.sum(probs * np.log(probs))
        print(f"\nEntropy: {entropy:.4f} (higher = more spread out)")
    
    # Show probability in the attractor region
    prob_core = sim.probability_in_region(-1.0, 1.0, -0.4, 0.4)
    print(f"Probability in core region [-1, 1] × [-0.4, 0.4]: {prob_core:.4f}")


def quick_demo():
    print("\n" + "="*60)
    print("QUICK DEMO: First 10 Iterations")
    print("="*60)
    
    grid = ProbabilityGrid(
        x_min=-1.5, x_max=1.5,
        y_min=-0.5, y_max=0.5,
        nx_cells=30,
        ny_cells=30
    )
    
    sim = MonteCarloSimulation(
        henon_a=A,
        henon_b=B,
        noise_radius=0.02,
        num_particles=5000,
        grid=grid
    )
    
    sim.initialize_particles(init_type='point', x0=0.6, y0=-0.2)
    
    for target_iter in [0, 1, 2, 5, 10]:
        while sim.current_iteration < target_iter:
            sim.step_forward()
        
        print(f"\n{'='*60}")
        print(f"ITERATION {sim.current_iteration}")
        print(f"{'='*60}")
        show_current_state(sim)
        plot_probability_heatmap(sim, f"Iteration {sim.current_iteration}")
    
    
def compute_transition_matrix_sparse(
    grid: ProbabilityGrid, 
    henon_a: float, 
    henon_b: float, 
    noise_radius: float = 0.02, 
    samples_per_cell: int = 500
) -> sparse.csr_matrix:
    """
    Compute the transition matrix for the noisy Henon map.
    
    Returns a sparse matrix P where P[i, j] is the probability that 
    a particle starting in cell i ends up in cell j after one iteration.
    """
    n_cells = grid.nx_cells * grid.ny_cells
    # Building the matrix using COO format, then convert to CSR
    row_indices = []
    col_indices = []
    values = []
    
    print(f"Computing transition matrix for {n_cells} cells...")
    print(f"Using {samples_per_cell} samples per cell")
    print(f"Total particle updates: {n_cells * samples_per_cell:,}")
    
    for source_cell in range(n_cells):
        if source_cell % 100 == 0:
            print(f"  Processing source cell {source_cell}/{n_cells}...")
        
        x_min_cell, x_max_cell, y_min_cell, y_max_cell = grid.get_cell_bounds(source_cell)
        
        x_samples = np.random.uniform(x_min_cell, x_max_cell, samples_per_cell)
        y_samples = np.random.uniform(y_min_cell, y_max_cell, samples_per_cell)
        
        x_new = 1 - henon_a * x_samples**2 + y_samples
        y_new = henon_b * x_samples
        
        # add noise to each particle
        theta = np.random.uniform(0, 2 * np.pi, samples_per_cell)
        r = noise_radius * np.sqrt(np.random.uniform(0, 1, samples_per_cell))
        x_final = x_new + r * np.cos(theta)
        y_final = y_new + r * np.sin(theta)
        
        # count destinations
        dest_counts = {}
        for x, y in zip(x_final, y_final):
            dest_cell = grid.point_to_cell(x, y)
            if dest_cell is not None:
                dest_counts[dest_cell] = dest_counts.get(dest_cell, 0) + 1
        
        for dest_cell, count in dest_counts.items(): 
            probability = count / samples_per_cell
            row_indices.append(source_cell)
            col_indices.append(dest_cell)
            values.append(probability)
    
    P_coo = sparse.coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_cells, n_cells)
    )
    
    P_csr = P_coo.tocsr()
    
    print(f"\nTransition matrix computed:")
    print(f"  Size: {n_cells} × {n_cells}")
    print(f"  Nonzero entries: {P_csr.nnz:,} ({100*P_csr.nnz/n_cells**2:.2f}% sparse)")
    
    return P_csr  

def find_invariant_measure(P: sparse.csr_matrix, tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
    """
    Find the invariant measure (stationary distribution) of the Markov chain defined by transition matrix P.
    
    Starting from uniform distribution, repeatedly apply P^T until convergence.
    This finds the left eigenvector associated with eigenvalue 1.
    """

    n = P.shape[0]
    
    # start with uniform distribution
    p = np.ones(n) / n 
    for iteration in range(max_iter):
        p_new = P.T @ p

        change = np.linalg.norm(p_new - p, 1)

        if change < tol: 
            print(f"Converged after {iteration} iterations")
            return p_new / np.sum(p_new)
        p = p_new
    print(f"Warning: Did not converge after {max_iter} iterations")
    return p / np.sum(p)



def visualize_sampled_next_positions(
    current_point: Tuple[float, float], 
    P: sparse.csr_matrix, 
    grid: ProbabilityGrid, 
    n_samples: int = 500, 
    henon_a: float = A, 
    henon_b: float = B,
    ax: Optional[plt.Axes] = None
):
    """ 
    Sample next possible positions and visualize them as a scatter plot.
    """
    if ax is None: 
        fig, ax = plt.subplots(figsize=(10, 8))
    
    x_curr, y_curr = current_point
    
    source_cell = grid.point_to_cell(x_curr, y_curr)
    if source_cell is None: 
        ax.text(0.5, 0.5, 'Current point is outside the grid!',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=16, color='red')
        return ax
    
    # get the probability distribution over the destination cells 
    next_step_probs = P[source_cell, :].toarray().flatten()
    
    # sample destination cells according to these probabilities, cells with higher probability will be sampled more often
    nonzero_cells = np.where(next_step_probs > 0)[0]
    nonzero_probs = next_step_probs[nonzero_cells]
    nonzero_probs /= np.sum(nonzero_probs)  # normalize
    
    sampled_cells = np.random.choice(
        nonzero_cells, 
        size=n_samples,
        p=nonzero_probs
    )

    sampled_x = []
    sampled_y = []
    
    for cell in sampled_cells:
        x_min, x_max, y_min, y_max = grid.get_cell_bounds(cell)
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        
        sampled_x.append(x)
        sampled_y.append(y)
        
    ax.scatter(sampled_x, sampled_y, alpha=0.3, s=20, c='orange',
               label=f'{n_samples} sampled next positions')
    
    # Mark the current point
    ax.plot(x_curr, y_curr, 'bo', markersize=12, label='Current point',
            markeredgecolor='white', markeredgewidth=2, zorder=10)
    
    # Mark the deterministic image
    x_det = 1 - henon_a * x_curr**2 + y_curr
    y_det = henon_b * x_curr
    ax.plot(x_det, y_det, 'gs', markersize=12, label='Deterministic image',
            markeredgecolor='white', markeredgewidth=2, zorder=10)
    
    # Draw arrow
    ax.annotate('', xy=(x_det, y_det), xytext=(x_curr, y_curr),
                arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.6))
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Sampled next positions from ({x_curr:.3f}, {y_curr:.3f})')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return ax



def main_transition_matrix_demo():
    print("=== Transition Matrix Visualization Demo ===\n")
    
    # Step 1: Create the grid covering the attractor region
    print("Step 1: Creating grid...")
    grid = ProbabilityGrid(
        x_min=-1.5, x_max=1.5,
        y_min=-0.5, y_max=0.5,
        nx_cells=30, ny_cells=30
    )
    print(f"Grid created: {grid.nx_cells}×{grid.ny_cells} = {grid.nx_cells * grid.ny_cells} cells\n")
    
    # Step 2: Compute the transition matrix (this takes a few minutes)
    print("Step 2: Computing transition matrix...")
    print("(This is the expensive one-time computation)\n")
    P = compute_transition_matrix_sparse(
        grid,
        henon_a=A,
        henon_b=B,
        noise_radius=0.02,
        samples_per_cell=500  # Use 500 samples per cell for reasonable accuracy
    )
    print("\nTransition matrix ready!\n")
    
    # Step 3: Choose a starting point on the attractor
    # Using a point that's known to be on the Hénon attractor
    starting_point = (0.6, -0.2)
    print(f"Step 3: Visualizing predictions from starting point {starting_point}\n")
    
    # Step 4: Create the visualization
    fig, ax = plt.subplots(figsize=(12, 10))
    visualize_sampled_next_positions(
        current_point=starting_point,
        P=P,
        grid=grid,
        n_samples=1000,  # Sample 1000 possible next positions
        henon_a=A,
        henon_b=B,
        ax=ax
    )
    
    plt.suptitle('Transition Matrix Prediction: Sampled Next Positions', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_transition_matrix_demo()
