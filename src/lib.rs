// henon_dynamics.rs - Rust WebAssembly Henon Map Library

// High-performance mathematical core for Henon map and set-valued dynamical systems
// Compiled to WebAssembly for browser execution

// Author: Nam V. Do
// Project: Visualization of Set-valued dynamical systems
// Supervisor: Dr. Kella Timperi, University of Oulu

use core::f64;

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use js_sys::Math;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

// Macro for console.log debugging
macro_rules! console_log {
    ($($t:tt)*) => {
        log(&format_args!($($t)*).to_string())
    };
}

/// System parameters that define Henon map properties
/// 
/// These parameters define the fundamental behavior of the dynamical system:
/// - a: Non-linear stretching parameter (typically 1.4)
/// - b: Dissipitation parameter (typically 0.3)
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SystemParameters {
    /// Non-linear stretching parameter
    a: f64,
    /// Dissipitation parameter
    b: f64
}

#[wasm_bindgen]
impl SystemParameters {
    /// Create new system parameters
    #[wasm_bindgen(constructor)]
    pub fn new(a: f64, b: f64) -> Result<SystemParameters, JsValue> {
        if a.is_nan() || a.is_infinite() {
            return Err(JsValue::from_str("Parameter 'a' must be a finite number"));
        }
        if b.is_nan() || b.is_infinite() {
            return Err(JsValue::from_str("Parameter 'b' must be a finite number"));
        }

        Ok(SystemParameters{a, b})
    }

    /// Create default parameters (a=1.4, b=0.3)
    #[wasm_bindgen]
    pub fn default() -> SystemParameters {
        SystemParameters {a: 1.4, b: 0.3}
    }

    /// Get Jacobian determinant for Henon map: det(J) = -b
    /// 
    /// This value is independent of position for the Henon map and indicates
    /// whether the system is dissipitative (|det(J)| < 1)
    #[wasm_bindgen]
    pub fn jacobian_determinant(&self) -> f64 {
        -self.b
    }

    /// Check if the system is dissipative (volume contracting)
    #[wasm_bindgen]
    pub fn is_dissipative(&self) -> bool {
        self.jacobian_determinant().abs() < 1.0
    }

    /// Rate at which volumes contract per iteration
    #[wasm_bindgen]
    pub fn volume_contraction_rate(&self) -> f64 {
        self.jacobian_determinant().abs()
    }

    /// Get parameter values as array [a, b]
    #[wasm_bindgen]
    pub fn to_array(&self) -> Vec<f64> {
        vec![self.a, self.b]
    }

    /// Get parameter 'a'
    #[wasm_bindgen(getter)]
    pub fn a(&self) -> f64 {
        self.a
    }

    /// Get parameter 'b'
    #[wasm_bindgen(getter)]
    pub fn b(&self) -> f64 {
        self.b
    }

    /// Set parameter 'a'
    #[wasm_bindgen(setter)]
    pub fn set_a(&mut self, a: f64) {
        self.a = a;
    }

    /// Set parameter 'b'
    #[wasm_bindgen(setter)]
    pub fn set_b(&mut self, b: f64) {
        self.b = b;
    }

}


/// Noise parameters for set-valued dynamics 
/// 
/// These define the bounds for random pertubations
/// - epsilon_x: Noise bound in the x-direction (ξ ∈ [-ε_x, +ε_x])
/// - epsilon_y: Noise bound in the y-direction (η ∈ [-ε_y, +ε_y])
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct NoiseParameters {
    /// Noise bound in the x-direction
    epsilon_x: f64,
    /// Noise bound in the y-direction
    epsilon_y: f64
}

#[wasm_bindgen]
impl NoiseParameters {
    /// Create noise parameters
    #[wasm_bindgen(constructor)]
    pub fn new(epsilon_x: f64, epsilon_y: f64) -> Result<NoiseParameters, JsValue> {
        if epsilon_x.is_nan() || epsilon_x.is_infinite() || epsilon_x < 0.0 {
            return Err(JsValue::from_str("Parameter 'epsilon_x' must be a finite non-negative number"))
        }
        if epsilon_y.is_nan() || epsilon_y.is_infinite() || epsilon_y < 0.0 {
            return Err(JsValue::from_str("Parameter 'epsilon_y' must be a finite non-negative number"))
        }
        Ok(NoiseParameters { epsilon_x: epsilon_x, epsilon_y: epsilon_y })
    }

    /// Create default noise parameters (εₓ=0.05, εᵧ=0.05)
    #[wasm_bindgen]
    pub fn default() -> NoiseParameters {
        NoiseParameters { 
            epsilon_x: 0.05,
            epsilon_y: 0.05
        }
    }

    /// Get the noise bounds as array [epsilon_x, epsilon_y]
    #[wasm_bindgen]
    pub fn to_array(&self) -> Vec<f64> {
        vec![self.epsilon_x, self.epsilon_y]
    }

    /// Get epsilon_x
    #[wasm_bindgen(getter)]
    pub fn epsilon_x(&self) -> f64 {
        self.epsilon_x
    }

    /// Get epsilon_y
    #[wasm_bindgen(getter)]
    pub fn epsilon_y(&self) -> f64 {
        self.epsilon_y
    }

    /// Set epsilon_x
    #[wasm_bindgen(setter)]
    pub fn set_epsilon_x(&mut self, epsilon_x: f64) {
        self.epsilon_x = epsilon_x;
    }

    /// Set epsilon_y
    #[wasm_bindgen(setter)]
    pub fn set_epsilon_y(&mut self, epsilon_y: f64) {
        self.epsilon_y = epsilon_y;
    }
}

/// State vector representing a point in 2D space (variables that evolve)
/// 
/// This represents the current state of the dynamical system
/// - x: First coordinate in 2D space
/// - y: Second coordinate in 2D space
#[wasm_bindgen]
pub struct StateVector {
    /// First coordinate in 2D space
    x: f64,
    /// Second coordinate in 2D space
    y: f64
}

#[wasm_bindgen]
impl StateVector {
    // Create a new state vector
    #[wasm_bindgen(constructor)]
    pub fn new(x: f64, y: f64) -> StateVector {
        StateVector { x: x, y: y }
    }

    /// Get position coordinates as [x, y]
    #[wasm_bindgen]
    pub fn to_array(&self) -> Vec<f64> {
        vec![self.x, self.y]
    }

    /// Calculate distance to another state vector
    #[wasm_bindgen]
    pub fn calculate_distance(&self, other: &StateVector) -> f64 {
        let distance_x = other.x - self.x;
        let distance_y = other.y - self.y;
        (distance_x * distance_x + distance_y * distance_y).sqrt() 
    }

    /// Get coordinate values 
    #[wasm_bindgen(getter)]
    pub fn x(&self) -> f64 {
        self.x
    }
    #[wasm_bindgen(getter)]
    pub fn y(&self) -> f64 {
        self.y
    }
    /// Get coordinate values
    
    #[wasm_bindgen(setter)]
    pub fn set_x(&mut self, x: f64) { self.x = x }

    #[wasm_bindgen(setter)]
    pub fn set_y(&mut self, y: f64) { self.y = y }
}


#[derive(Clone, Debug)]
struct BoundaryPoint {
    /// Position in 2D space (x, y)
    position: (f64, f64),
    /// Outward unit normal vector (nx, ny)
    normal: (f64, f64)
}


impl BoundaryPoint {
    fn new(x: f64, y: f64, nx: f64, ny: f64) -> Self {
        // Normalize the normal vector
        let mag = (nx * nx + ny * ny).sqrt();
        let normal = if mag > 1e-10 {
            (nx / mag, ny / mag)
        } else {
            console_log!("Warning: Zero normal vector at: ({}, {}), using default", x, y);
            (1.0, 1.0)
        };

        BoundaryPoint {
            position: (x, y),
            normal: normal,
        }
    }

    fn on_circle(cx: f64, cy: f64, epsilon: f64, angle: f64) -> Self {
        let x = cx + epsilon * angle.cos();
        let y = cy + epsilon * angle.sin();

        // For a circle, the outward normal is just the radial direction 
        let nx = angle.cos();
        let ny = angle.sin();

        BoundaryPoint {
            position: (x, y),
            normal: (nx, ny)
        }
    }
}

#[wasm_bindgen]
pub struct SetValuedSimulation {
    henon_map: HenonMap,
    epsilon: f64,
    /// Boundary points with their normal vectors
    boundary_points: Vec<BoundaryPoint>,
    iteration_count: usize,
    /// Convergence threshold for termination
    convergence_threshold: f64
}

#[wasm_bindgen]
impl SetValuedSimulation {
    /// Initialize with circular initial set
    /// 
    /// Algorithm 1, Step (i)-(iii):
    /// - Pick initial point (center)
    /// - Map it forward [we start at the mapped point]
    /// - Sample boundary points on ∂B_ε(center)
    #[wasm_bindgen(constructor)]
    pub fn new(
        params: &SystemParameters,
        epsilon: f64,
        initial_center_x: f64,
        initial_center_y: f64,
        n_boundary_points: usize
    ) -> Result<SetValuedSimulation, JsValue> {
        if epsilon <= 0.0 || epsilon.is_nan() || epsilon.is_infinite() {
            return Err(JsValue::from_str("Parameter 'epsilon' must be a positive number"));
        } 
        if n_boundary_points < 3 {
            let err = format!("Parameter 'n_boundary_points' must be at least 3, got: {}", n_boundary_points);
            return Err(JsValue::from_str(err.as_str()));
        }
        
        let henon_map = HenonMap::new(params);
        let state0 = StateVector::new(initial_center_x, initial_center_y);
        let state1 = henon_map.iterate(&state0);
        let center_x = state1.x();
        let center_y = state1.y();
        
        let mut boundary_points = Vec::with_capacity(n_boundary_points);
        for i in 0..n_boundary_points {
            let angle = 2.0 * f64::consts::PI * (i as f64) / (n_boundary_points as f64);
            let boundary_point = BoundaryPoint::on_circle(center_x, center_y, epsilon, angle);
            boundary_points.push(boundary_point);
        }
        
        console_log!("Initialized set-valued simulation with {} boundary points", boundary_points.len());
        Ok(SetValuedSimulation {
            henon_map,
            epsilon,
            boundary_points,
            iteration_count: 0,
            convergence_threshold: 1e-6
        }) 
    }


    /// Get initial state for visualization (iteration 0)
    /// Returns JSON string with boundary points that haven't been iterated yet
    #[wasm_bindgen]
    pub fn get_initial_state_details(&self) -> Result<String, JsValue> {
        if self.boundary_points.is_empty() {
            return Err(JsValue::from_str("No boundary points initialized"));
        }

        // For initial state, the boundary_points ARE already the projected points
        // (they were created as points on the circle boundary in the constructor)
        let mut projected_points = Vec::new();
        let mut normals = Vec::new();

        // Compute centroid of current boundary
        let mut centroid_x = 0.0;
        let mut centroid_y = 0.0;

        for point in &self.boundary_points {
            let (x, y) = point.position;
            let (nx, ny) = point.normal;

            projected_points.push(x);
            projected_points.push(y);
            normals.push(nx);
            normals.push(ny);

            centroid_x += x;
            centroid_y += y;
        }

        let n = self.boundary_points.len() as f64;
        centroid_x /= n;
        centroid_y /= n;

        // For initial state, mapped_points are at the centroid (the initial f(x0))
        let mapped_points = vec![centroid_x, centroid_y];

        let result = format!(
            r#"{{"iteration": 0, "mapped_points": {:?}, "projected_points": {:?}, "normals": {:?}, "epsilon": {}, "max_movement": 0.0, "centroid": [{}, {}]}}"#,
            mapped_points,
            projected_points,
            normals,
            self.epsilon,
            centroid_x,
            centroid_y
        );

        Ok(result)
    }

    /// Perform 1 iteration and return detailed visualization data
    /// Returns JSON string with all intermediate steps
    #[wasm_bindgen]
    pub fn iterate_boundary_with_details(&mut self) -> Result<String, JsValue> {
        if self.boundary_points.is_empty() {
            return Err(JsValue::from_str("No boundary points initialized"));
        }

        let n = self.boundary_points.len();
        let mut mapped_points = Vec::new();
        let mut projected_points = Vec::new();
        let mut normals = Vec::new();
        let mut new_boundary_points = Vec::with_capacity(n);
        let mut max_movement: f64 = 0.0;
        let mut centroid_x = 0.0;
        let mut centroid_y = 0.0;
        let mut n_valid = 0;

        let a = self.henon_map.get_parameters().a();
        let b = self.henon_map.get_parameters().b();

        // Step (v): For each boundary point zk
        for old_point in &self.boundary_points {
            let (x, y) = old_point.position;
            let (nx, ny) = old_point.normal;
            
            // Safety check
            if !x.is_finite() || !y.is_finite() || x.abs() > 100.0 || y.abs() > 100.0 {
                continue;
            }
            
            // Step (v-a): Compute f(zk)
            let f_x = 1.0 - a * x * x + y;
            let f_y = b * x;
            
            if !f_x.is_finite() || !f_y.is_finite() || f_x.abs() > 100.0 || f_y.abs() > 100.0 {
                continue;
            }
            
            mapped_points.push(f_x);
            mapped_points.push(f_y);

            centroid_x += f_x;
            centroid_y += f_y;
            n_valid += 1;
            
            // Step (v-b): Transform normal using (J^(-1))^T (inverse transpose)
            // For Hénon map: J = [[-2ax, 1], [b, 0]]
            // det(J) = -b
            // J^(-1) = (1/-b) * [[0, -1], [-b, -2ax]] = [[0, 1/b], [1, 2ax/b]]
            // (J^(-1))^T = [[0, 1], [1/b, 2ax/b]]
            // Normal vectors are covectors and transform with inverse transpose!

            // Check for degenerate case where b ≈ 0
            if b.abs() < 1e-10 {
                console_log!("⚠️ Warning: b ≈ 0, Jacobian is nearly singular");
                continue;
            }

            let nx_transformed = ny;  // First row: 0*nx + 1*ny
            let ny_transformed = nx / b + (2.0 * a * x / b) * ny;  // Second row: (1/b)*nx + (2ax/b)*ny

            // Safety check for transformed normal
            if !nx_transformed.is_finite() || !ny_transformed.is_finite() {
                console_log!("⚠️ Warning: Non-finite transformed normal");
                continue;
            }
            
            let normal_mag = (nx_transformed * nx_transformed + ny_transformed * ny_transformed).sqrt();
            
            let (nx_new, ny_new) = if normal_mag > 1e-8 {
                (nx_transformed / normal_mag, ny_transformed / normal_mag)
            } else {
                let radial_mag = (f_x * f_x + f_y * f_y).sqrt();
                if radial_mag > 1e-8 {
                    (f_x / radial_mag, f_y / radial_mag)
                } else {
                    (nx, ny)
                }
            };

            normals.push(nx_new);
            normals.push(ny_new);

            // Step (v-c): Project outward
            let new_x = f_x + self.epsilon * nx_new;
            let new_y = f_y + self.epsilon * ny_new;
            
            if !new_x.is_finite() || !new_y.is_finite() {
                continue;
            }

            projected_points.push(new_x);
            projected_points.push(new_y);

            let movement = ((new_x - x).powi(2) + (new_y - y).powi(2)).sqrt();
            if movement.is_finite() && movement > max_movement {
                max_movement = movement;
            }

            new_boundary_points.push(BoundaryPoint::new(new_x, new_y, nx_new, ny_new));
        }

        if new_boundary_points.is_empty() {
            return Err(JsValue::from_str("All boundary points diverged"));
        }

        // Compute centroid
        if n_valid > 0 {
            centroid_x /= n_valid as f64;
            centroid_y /= n_valid as f64;
        }

        self.boundary_points = new_boundary_points;
        self.iteration_count += 1;

        // Create JSON with all visualization data
        let result = format!(
            r#"{{"iteration": {}, "mapped_points": {:?}, "projected_points": {:?}, "normals": {:?}, "epsilon": {}, "max_movement": {}, "centroid": [{}, {}]}}"#,
            self.iteration_count,
            mapped_points,
            projected_points,
            normals,
            self.epsilon,
            max_movement,
            centroid_x,
            centroid_y
        );

        Ok(result)
    }


    /// Perform 1 iteration of Algorithm 1
    /// This implements step (iv) and (v) of Algorithm 1
    #[wasm_bindgen]
    pub fn iterate_boundary(&mut self) -> Result<f64, JsValue> {
        if self.boundary_points.is_empty() {
            return Err(JsValue::from_str("No boundary points initialized"));
        }

        let n = self.boundary_points.len();
        let mut new_boundary_points = Vec::with_capacity(n);
        let mut max_movement: f64 = 0.0;

        // Step (v): For each boundary point zk
        for old_point in &self.boundary_points {
            let (x, y) = old_point.position;
            let (nx, ny) = old_point.normal;
            
            // Safety check: Skip diverged points
            if !x.is_finite() || !y.is_finite() || x.abs() > 100.0 || y.abs() > 100.0 {
                console_log!("Skipping diverged point: ({}, {})", x, y);
                continue;
            }
            
            // Step (v-a): Compute f(zk) - Direct Hénon map calculation
            let a = self.henon_map.get_parameters().a();
            let b = self.henon_map.get_parameters().b();
            let f_x = 1.0 - a * x * x + y;
            let f_y = b * x;
            
            // Check if mapped point is valid
            if !f_x.is_finite() || !f_y.is_finite() || f_x.abs() > 100.0 || f_y.abs() > 100.0 {
                console_log!("Skipping point with invalid mapping: f({}, {}) = ({}, {})", 
                            x, y, f_x, f_y);
                continue;
            }
            
            // Step (v-b): Transform normal using (J^(-1))^T (inverse transpose)
            // For Hénon map: J = [[-2ax, 1], [b, 0]]
            // det(J) = -b
            // J^(-1) = [[0, 1/b], [1, 2ax/b]]
            // (J^(-1))^T = [[0, 1], [1/b, 2ax/b]]
            // Normal vectors are covectors and must transform with inverse transpose!

            let nx_transformed = ny;  // 0*nx + 1*ny
            let ny_transformed = nx / b + (2.0 * a * x / b) * ny;  // (1/b)*nx + (2ax/b)*ny
            
            // Normalize the transformed normal
            let normal_mag = (nx_transformed * nx_transformed + ny_transformed * ny_transformed).sqrt();
            
            let (nx_new, ny_new) = if normal_mag > 1e-8 {
                (nx_transformed / normal_mag, ny_transformed / normal_mag)
            } else {
                // Fallback: Use radial direction from origin if normal is too small
                let radial_mag = (f_x * f_x + f_y * f_y).sqrt();
                if radial_mag > 1e-8 {
                    (f_x / radial_mag, f_y / radial_mag)
                } else {
                    // Ultimate fallback: Use previous normal
                    console_log!("Using previous normal at f({}, {}) = ({}, {})", 
                                x, y, f_x, f_y);
                    (nx, ny)
                }
            };

            // Step (v-c): Project outward ẑk = f(zk) + ε * n(f(zk))
            let new_x = f_x + self.epsilon * nx_new;
            let new_y = f_y + self.epsilon * ny_new;
            
            // Final safety check
            if !new_x.is_finite() || !new_y.is_finite() {
                console_log!("Skipping non-finite projection: ({}, {})", new_x, new_y);
                continue;
            }

            // Track maximum movement for convergence check
            let movement = ((new_x - x).powi(2) + (new_y - y).powi(2)).sqrt();
            if movement.is_finite() && movement > max_movement {
                max_movement = movement;
            }

            new_boundary_points.push(BoundaryPoint::new(new_x, new_y, nx_new, ny_new));
        }

        // Check if we lost too many points
        if new_boundary_points.is_empty() {
            return Err(JsValue::from_str("All boundary points diverged"));
        }
        
        if new_boundary_points.len() < n / 2 {
            console_log!("⚠️  Lost {} boundary points (from {} to {})", 
                        n - new_boundary_points.len(), n, new_boundary_points.len());
        }

        // Step (vi): Replace old boundary with new boundary
        self.boundary_points = new_boundary_points;
        self.iteration_count += 1;

        console_log!("✓ Iteration {}: {} points, max movement = {:.6e}", 
                    self.iteration_count, self.boundary_points.len(), max_movement);
        Ok(max_movement)
        } 
        /// Runs the Algorithm 1 until convergence or maximum iterations reached
        /// Returns number of iterations performed
        #[wasm_bindgen]
        pub fn run_until_convergence(&mut self, max_iterations: usize) -> Result<usize, JsValue> {
            console_log!("Running Algorithm 1 until convergence (max {} iterations)", max_iterations);

            for iter in 0..max_iterations {
                let movement = self.iterate_boundary()?;

                // Step (T): check termination condition
                if movement < self.convergence_threshold {
                    console_log!("Converged after: {} iterations (movement = {:.6e} < threshold = {:.6e})", iter + 1, movement, self.convergence_threshold);
                    return Ok(iter + 1);
                }
            }

            console_log!("Reach maximum iterations ({}) without full convergence", max_iterations);
            Ok(max_iterations)
        }

        /// Run Algorithm 1 and track the full history of boundary points for visualization.
        /// Returns a flat array of all boundary points from all iterations,
        /// with each iteration separated by a NaN value.
        /// Format: [x1, y1, ..., xN, yN, NaN, x1', y1', ..., xN', yN', NaN, ...]
        #[wasm_bindgen]
        pub fn track_boundary_evolution(&mut self, max_iterations: usize) -> Result<Vec<f64>, JsValue> {
            let mut history = Vec::new();

            // Add initial state to history
            let initial_positions = self.get_boundary_positions();
            history.extend(initial_positions);
            history.push(f64::NAN);

            for iter in 0..max_iterations {
                let movement = self.iterate_boundary()?;

                // Add current boundary to history
                let positions = self.get_boundary_positions();
                history.extend(positions);
                history.push(f64::NAN);

                if movement < self.convergence_threshold {
                    console_log!("Converged after {} iterations.", iter + 1);
                    break;
                }
            }

            Ok(history)
        }

    /// Get boundary positions as flat array [x1, y1, x2, y2, ....]
    #[wasm_bindgen(js_name = getBoundaryPositions)]
    pub fn get_boundary_positions(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.boundary_points.len() * 2);
        for point in &self.boundary_points {
            result.push(point.position.0);
            result.push(point.position.1);
        }
        result
    }

    /// Get current normal vectors as flat array [nx1, ny1, nx2, ny2, ...]
    #[wasm_bindgen(js_name = getNormalVectors)]
    pub fn get_normal_vectors(&self) -> Vec<f64> {
        let mut result = Vec::with_capacity(self.boundary_points.len() * 2);
        for point in &self.boundary_points {
            result.push(point.normal.0);
            result.push(point.normal.1);
        }
        result
    }

    /// Get number of iterations performed
    #[wasm_bindgen(js_name = getIterationCount)]
    pub fn get_iteration_count(&self) -> usize {
        self.iteration_count
    }

    /// Set convergence threshold 
    pub fn set_convergence_threshold(&mut self, threshold: f64) {
        self.convergence_threshold = threshold;
        console_log!("Set convergence threshold to {}", threshold);
    }

    /// Get the center of mass of boundary points
    #[wasm_bindgen(js_name = getCentroid)]
    pub fn get_centroid(&self) -> Vec<f64> {
        if self.boundary_points.is_empty() {
            return vec![0.0, 0.0];
        }

        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        for point in &self.boundary_points {
            sum_x += point.position.0;
            sum_y += point.position.1;
        }

        let n = self.boundary_points.len() as f64;
        vec![sum_x / n, sum_y / n]
    }

    /// Compute approximate area enclosed by boundary points (using shoelace formula)
    #[wasm_bindgen(js_name = computeEnclosedArea)]
    pub fn compute_enclosed_area(&self) -> f64 {
        let n = self.boundary_points.len();
        if n < 3 {
            return 0.0;
        }
        let mut area = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            let (xi, yi) = self.boundary_points[i].position;
            let (xj, yj) = self.boundary_points[j].position;
            area += xi * yj - xj * yi;
        }

        (area / 2.0).abs()
    }
}

/// Deterministic Henon map implementation 
/// 
/// Mathematical formulation 
/// x_{n+1} = 1 - a*x_n^2 + y_n
/// y_{n+1} = b*x_n

#[wasm_bindgen]
pub struct HenonMap {
    params: SystemParameters
}

#[wasm_bindgen]
impl HenonMap {
    /// Create a new Henon map with given parameters
    #[wasm_bindgen(constructor)]
    pub fn new(params: &SystemParameters) -> HenonMap {
        console_log!("Create Henon map with a={}, b={}", params.a(), params.b());
        HenonMap { params: *params }
    } 

    /// Create Henon map with custom parameters
    #[wasm_bindgen(js_name = withParameters)]
    pub fn with_parameters(a: f64, b: f64) -> Result<HenonMap, JsValue> {
        let system_parameters = SystemParameters::new(a, b)?;
        Ok(HenonMap::new(&system_parameters))
    }

    /// Create Henon map with default parameters (a=1.4, b=0.3)
    pub fn default() -> HenonMap {
        let params = SystemParameters::default();
        HenonMap::new(&params)
    }

    /// Single iteration of deterministic Henon map
    /// 
    /// Applies the transformation (x, y) = (1 - a*x^2 + y, bx)
    /// 
    #[wasm_bindgen]
    pub fn iterate(&self, state: &StateVector) -> StateVector {
        let a = self.params.a();
        let b = self.params.b();
        let x = state.x();
        let y = state.y();

        let x_new = 1.0 - (a * x * x) + y;
        let y_new = b * x;
        StateVector::new(x_new, y_new)
    }


    /// Returns a flattened array of coordinates [x1, y1, x2, y2, ...]
    /// This format is optimized for JavaScript/WebAssembly transfer
    #[wasm_bindgen]
    pub fn generate_trajectory(
        &self,
        initial_x: f64,
        initial_y: f64,
        n_iterations: u32,
        skip_transient: u32
    ) -> Vec<f64> {
        let mut trajectory = Vec::with_capacity((n_iterations as usize) * 2  );
        let mut state = StateVector::new(initial_x, initial_y);
        // Skip transient iterations
        for _ in 0..skip_transient {
            state = self.iterate(&state);
        }
        
        // Collect main trajectory

        for _ in 0..n_iterations {
            state = self.iterate(&state);
            trajectory.push(state.x());
            trajectory.push(state.y());
        }

        console_log!("Generate trajectory with {} points", n_iterations);
        trajectory
    }

    /// Compute Jacobian matrix at a given state 
    /// 
    /// Returns flatten 2x2 matrix: [∂f/∂x, ∂f/∂y, ∂g/∂x, ∂g/∂y] 
    /// J = [-2ax   1]
    ///     [b      0]        
    #[wasm_bindgen(js_name = jacobianMatrix)]
    pub fn jacobian_matrix(&self, state: &StateVector) -> Vec<f64> {
        let a = self.params.a();
        let b = self.params.b();
        let x = state.x();
        vec![
            -2.0 * a * x, //  ∂f/∂x
            1.0, // ∂f/∂y
            b, // ∂g/∂x 
            0.0 // ∂g/∂y
        ]
    }

    /// Calculate fixed points of Henon map
    /// 
    /// Returns flatten array of fixed point coordinates [x1, y1, x2, y2, ...]
    /// Can return 0, 2, or 4 values dependent on the discriminant
    #[wasm_bindgen]
    pub fn fixed_points(&self) -> Result<Vec<f64>, JsValue> {
        let a = self.params.a();
        let b = self.params.b();

        // Quadratic coefficients: ax^2 - (1+b)x + 1 = 0
        let coefficient_a = a;
        let coefficient_b = -(1.0 + b);
        let coefficient_c = 1.0;

        let discriminant = (coefficient_b * coefficient_b) - 4.0 * (coefficient_a * coefficient_c);
        if discriminant < 0.0 {
            // No real fixed-point
            console_log!("No real fixed points (discriminant = {})", discriminant);
            Ok(vec![])
        } else if discriminant == 0.0 {
            // One fixed point
            let x = - coefficient_b / (2.0 * coefficient_a);
            let y = b * x;
            console_log!("One fixed point: ({}, {})", x, y);
            Ok(vec![x, y])
        } else {
            // Two fixed points
            let sqrt_disc = discriminant.sqrt();
            let x1 = (-coefficient_b + sqrt_disc) / (2.0 * coefficient_a);
            let x2 = (-coefficient_b - sqrt_disc) / (2.0 * coefficient_a);
            let y1 = b * x1;
            let y2 = b * x2;
            console_log!("Two fixed-points: ({},{}), ({}.{})", x1, y1, x2, y2);
            Ok(vec![x1, y1, x2, y2])
        }

    }


    /// Get system parameters
    #[wasm_bindgen]
    pub fn get_parameters(&self) -> SystemParameters {
        self.params
    }

    /// Update system parameters
    #[wasm_bindgen]
    pub fn update_parameters(&mut self, params: SystemParameters) {
        console_log!("Update system parameters to a={}, b={}", params.a(), params.b());
        self.params = params;
    }

}

/// Set-valued Henon Map with bounded noise
/// 
/// Mathematical formulation
/// x_{n+1} = [1 - a*x_n^2 + y_n - ε_x, 1 - a*x_n^2 + y_n + ε_x]
/// y_{n+1} = [b*x_n - ε_y, b*x_n + ε_y]
/// This represents a more realistic model where the system is subject to 
/// bounded permutations, preventing collapse to measure-zero attractor.
#[wasm_bindgen]
pub struct SetValuedHenonMap {
    params: SystemParameters,
    noise_params: NoiseParameters,
}

#[wasm_bindgen]
impl SetValuedHenonMap {
    /// Create a new set-valued Henon map
    #[wasm_bindgen(constructor)]
    pub fn new(params: &SystemParameters, noise_params: &NoiseParameters) -> Result<SetValuedHenonMap, JsValue> {
        Ok(SetValuedHenonMap { params: *params, noise_params: *noise_params })
    }

    /// Create set-valued map with custom parameters
    #[wasm_bindgen(js_name = withParameters)]
    pub fn with_parameters(
        a: f64,
        b: f64,
        epsilon_x: f64,
        epsilon_y: f64
    ) -> Result<SetValuedHenonMap, JsValue> {
        let params = SystemParameters::new(a, b)?;
        let noise_params = NoiseParameters::new(epsilon_x, epsilon_y)?;
        SetValuedHenonMap::new(&params, &noise_params)
    }

    /// Create set-valued map with default parameters
    pub fn default() -> SetValuedHenonMap{
        let params = SystemParameters::default();
        let noise_params = NoiseParameters::default();
        SetValuedHenonMap { params: params, noise_params: noise_params}
    }

    /// Single iteration of set-valued Henon map
    /// 
    /// Sample uniformly from set-valued intervals 
    /// Mathematical model:
    /// x_{n+1} = 1 - a*x_n^2 + y + ξ
    /// y_{n+1} = b*x_n + η 
    /// Where (ξ, η) ∈ B_ε(0) = {(u,v) : √(u² + v²) ≤ ε}
    #[wasm_bindgen]
    pub fn iterate(&self, state: &StateVector) -> StateVector {
        let a = self.params.a();
        let b = self.params.b();
        let x = state.x();
        let y = state.y();
        let epsilon = self.noise_params.epsilon_x();

        // Bounds checking
        const MAX_VALUE: f64 = 10.0;
        if x.abs() > MAX_VALUE || y.abs() > MAX_VALUE {
            console_log!("Noisy state values too large: x={}, y={}", x, y);
            // Clamp to attractor bounds instead of hard reset
            let clamped_x = x.clamp(-2.0, 2.0);
            let clamped_y = y.clamp(-1.0, 1.0);
            return StateVector::new(clamped_x, clamped_y);
        }

        // Deterministic part f(x, y) = (1 - a*x^2 + y, b*x)

        let x_det = 1.0 - (a * x * x) + y;
        let y_det = b * x;

        if !x_det.is_finite() || !y_det.is_finite() {
            console_log!("Noisy iteration invalid: x_det={}, y_det={}, x={}, y={}, a={}, b={}", 
                        x_det, y_det, x, y, a, b);
            // Return a safe state to avoid divergence
            return StateVector::new(0.1, 0.1);
        }

        // Circular noise sampling 
        // Step 1: Generate uniform distribution
        let u1: f64 = Math::random();
        let u2: f64 = Math::random();

        // Step 2: Transform to get uniform distribution on disk
        let r = epsilon * (u1).sqrt(); // Distance from the centner
        let theta = 2.0 * f64::consts::PI * u2; // 0 to 2π - the angle

        // Step 3: Convert to Cartesian coordinates
        let noise_x = r * theta.cos();
        let noise_y = r * theta.sin();

        // Final state with noise 
        let x_new = x_det + noise_x;
        let y_new = y_det + noise_y;

        if !x_new.is_finite() || !y_new.is_finite() {
            console_log!("Noisy iteration invalid after noise: x_new={}, y_new={}, x_det={}, y_det={}, noise_x={}, noise_y={}, x={}, y={}, a={}, b={}", 
                        x_new, y_new, x_det, y_det, noise_x, noise_y, x, y, a, b);
            return StateVector::new(x_det, y_det);
        }
        return StateVector::new(x_new, y_new);
    }

    /// Get a set-valued intervals
    /// Returns: [x_min, x_max, y_min, y_max]
    #[wasm_bindgen]
    pub fn iterate_set(&self, state: &StateVector) -> Vec<f64> {
        let a = self.params.a();
        let b = self.params.b();
        let x = state.x();
        let y = state.y();
        let epsilon = self.noise_params.epsilon_x();
        // Deterministic part

        let x_det = 1.0 - (a * x * x) + y;
        let y_det = b * x;

        // Set-valued intervals
        vec![
            x_det - epsilon, // x_min
            x_det + epsilon, // x_max
            y_det - epsilon, // y_min
            y_det + epsilon, // y_max
        ]
    }

    /// Generate trajectory with bounded noise
    #[wasm_bindgen]
    pub fn generate_trajectory(
        &self,
        initial_x: f64,
        initial_y: f64,
        n_iterations: u32,
        skip_transient: u32
    ) -> Vec<f64> {
        let mut trajectory = Vec::with_capacity((n_iterations as usize) * 2);
        let mut state = StateVector::new(initial_x, initial_y);

        // Skip transient iterations
        for _ in 0..skip_transient {
            state = self.iterate(&state);
        }

        // Collect main trajectory
        for _ in 0..n_iterations {
            state = self.iterate(&state);
            trajectory.push(state.x());
            trajectory.push(state.y());
        }

        console_log!("Generate noisy trajectory with {} points", n_iterations);
        trajectory
    }

    /// Estimate effective dissipation rate in presense of noise
    #[wasm_bindgen]
    pub fn effective_dissipation_rate(&self) -> f64 {
        // The noise doesn't change the average contraction rate
        self.params.volume_contraction_rate()
    }

    /// Get system parameters
    #[wasm_bindgen]
    pub fn get_parameters(&self) -> SystemParameters {
        self.params
    }

    /// Get noise parameters
    #[wasm_bindgen]
    pub fn get_noise_parameters(&self) -> NoiseParameters {
        self.noise_params
    }

    /// Update system parameters
    #[wasm_bindgen]
    pub fn set_parameters(&mut self, params: &SystemParameters) {
        console_log!("Update system parameters to a={}, b={}", params.a(), params.b());
        self.params = *params;
    }

    /// Update noise parameters
    pub fn set_noise_parameters(&mut self, noise_params: &NoiseParameters) {
        console_log!("Update noise parameters to epsilon_x={}, epsilon_y={}", noise_params.epsilon_x(), noise_params.epsilon_y());
        self.noise_params = *noise_params;
    }
}

/// Comparison utilities for analyzing different trajectory types
#[wasm_bindgen]
pub struct TrajectoryComparison;

#[wasm_bindgen]
impl TrajectoryComparison {
    /// Compare deterministic and noisy trajectories
    /// 
    /// Returns analysis results as JSON string for easy JavaScript computation
    #[wasm_bindgen]
    pub fn compare_trajectories(
        deterministic_trajectory: &[f64],
        noisy_trajectory: &[f64]
    ) -> String {
        let det_len = deterministic_trajectory.len() / 2;
        let noisy_len = noisy_trajectory.len() / 2;

        if det_len == 0 || noisy_len == 0 {
            return "{}".to_string();
        }

        // Calculate statistics for deterministic trajectory
        let mut det_x_sum = 0.0;
        let mut det_y_sum = 0.0;
        let mut det_x_min = f64::INFINITY;
        let mut det_x_max = f64::NEG_INFINITY;
        let mut det_y_min = f64::INFINITY;
        let mut det_y_max = f64::NEG_INFINITY;

        for i in 0..det_len {
            let x = deterministic_trajectory[i * 2];
            let y = deterministic_trajectory[i * 2 + 1];

            det_x_sum += x;
            det_y_sum += y;
            det_x_min = det_x_min.min(x);
            det_x_max = det_x_max.max(x);
            det_y_min = det_y_min.min(y);
            det_y_max = det_y_max.max(y);
        }

        let det_x_mean = det_x_sum / (det_len as f64);
        let det_y_mean = det_y_sum / (det_len as f64);

        // Calculate statistics for noisy trajectory
        let mut noisy_x_sum = 0.0;
        let mut noisy_y_sum = 0.0;
        let mut noisy_x_min = f64::INFINITY;
        let mut noisy_x_max = f64::NEG_INFINITY;
        let mut noisy_y_min = f64::INFINITY;
        let mut noisy_y_max = f64::NEG_INFINITY;

        for i in 0..noisy_len {
            let x = noisy_trajectory[i * 2];
            let y = noisy_trajectory[i * 2 + 1];

            noisy_x_sum += x;
            noisy_y_sum += y;
            noisy_x_min = noisy_x_min.min(x);
            noisy_x_max = noisy_x_max.max(x);
            noisy_y_min = noisy_y_min.min(y);
            noisy_y_max = noisy_y_max.max(y);
        }

        let noisy_x_mean = noisy_x_sum / (noisy_len as f64);
        let noisy_y_mean = noisy_y_sum / (noisy_len as f64);

        // Estimate attractor thickness

        let thickness_x = noisy_x_max - noisy_x_min;
        let thickness_y = noisy_y_max - noisy_y_min;

        // Create JSON result
        format!(
            r#"{{
                "deterministic": {{
                    "mean": [{}, {}],
                    "bounds": [{}, {}, {}, {}],
                    "n_points": {}
                }},
                "noisy": {{
                    "mean": [{}, {}],
                    "bounds": [{}, {}, {}, {}],
                    "n_points": {}
                }},
                "attractor_thickness": {{
                    "x": {},
                    "y": {}
                }}
            }}"#,
            det_x_mean, det_y_mean, det_x_min, det_x_max, det_y_min, det_y_max, det_len,
            noisy_x_mean, noisy_y_mean, noisy_x_min, noisy_x_max, noisy_y_min, noisy_y_max, noisy_len,
            thickness_x, thickness_y
        )
    }
}

/// Utility function for WebAssembly integration 

#[wasm_bindgen]
pub struct Utils;

#[wasm_bindgen]
impl Utils {
    /// Initialize WebAssembly module (call this first from JavaScript)
    #[wasm_bindgen]
    pub fn init() {
        console_log!("Henon Map WebAssembly module initialized succesfully");
    }

    /// Get version information
    pub fn version() -> String {
        "1.0.0".to_string()
    }

    /// Performance test - measure iteration speed
    #[wasm_bindgen]
    pub fn performance_test(n_iterations: u32) -> f64 {
        let start = js_sys::Date::now();
        let henon = HenonMap::default();
        let mut state = StateVector::new(0.1, 0.1);

        for _ in 0..n_iterations {
            state = henon.iterate(&state);
        }

        let end = js_sys::Date::now();
        let duration = end - start;

        console_log!("Performed {} iterations in {:.2}ms", n_iterations, duration);
        console_log!("Speed: {:.0} iterations/second", n_iterations as f64 / (duration / 1000.0));

        duration
    }
}



#[wasm_bindgen(start)]
pub fn main() {
    console_log!("Rust WebAsssembly Henon Map library loaded")
}
