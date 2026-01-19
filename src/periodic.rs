use web_sys::console;
use std::f64;
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy)]
#[wasm_bindgen]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[wasm_bindgen]
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum StabilityType {
    Stable,
    Unstable,
    Saddle,
}

#[derive(Debug, Clone)]
pub enum PointClassification {
    Regular,
    NearPeriodicOrbit {
        period: usize,
        stability: StabilityType,
        distance: f64,
    },
}

#[derive(Debug, Clone)]
pub struct BoundaryPoint {
    pub x: f64,
    pub y: f64,
    pub nx: f64,
    pub ny: f64,
    pub classification: PointClassification,
}

#[derive(Debug, Clone)]
pub struct Jacobian {
    pub j11: f64,
    pub j12: f64,
    pub j21: f64,
    pub j22: f64,
}

impl Jacobian {
    pub fn inverse(&self) -> Option<Jacobian> {
        let det = self.j11 * self.j22 - self.j12 * self.j21;
        if det.abs() < 1e-12 {
            return None;
        }
        Some(Jacobian {
            j11: self.j22 / det,
            j12: -self.j12 / det,
            j21: -self.j21 / det,
            j22: self.j11 / det,
        })
    }

    pub fn multiply(&self, other: &Jacobian) -> Jacobian {
        Jacobian {
            j11: self.j11 * other.j11 + self.j12 * other.j21,
            j12: self.j11 * other.j12 + self.j12 * other.j22,
            j21: self.j21 * other.j11 + self.j22 * other.j21,
            j22: self.j21 * other.j12 + self.j22 * other.j22,
        }
    }

    pub fn eigenvalues(&self) -> (f64, f64, bool) {
        let trace = self.j11 + self.j22;
        let det = self.j11 * self.j22 - self.j12 * self.j21;
        let discriminant = trace * trace - 4.0 * det;

        if discriminant >= 0.0 {
            let sqrt_disc = discriminant.sqrt();
            (
                (trace + sqrt_disc) / 2.0,
                (trace - sqrt_disc) / 2.0,
                false,
            )
        } else {
            let modulus = det.sqrt();
            (modulus, modulus, true)
        }
    }
}

#[derive(Debug, Clone)]
pub struct PeriodicOrbit {
    pub points: Vec<Point>,
    pub period: usize,
    pub stability: StabilityType,
}


#[derive(Debug)]
pub struct BoundarySnapshot {
    pub iteration: usize,
    pub points: Vec<BoundaryPoint>,
}

fn henon_map(x: f64, y: f64, a: f64, b: f64) -> (f64, f64) {
    (1.0 - a * x * x + y, b * x)
}

fn henon_jacobian(x: f64, _y: f64, a: f64, b: f64) -> Jacobian {
    Jacobian {
        j11: -2.0 * a * x,
        j12: 1.0,
        j21: b,
        j22: 0.0,
    }
}

fn classify_stability(jac: &Jacobian) -> StabilityType {
    let (lambda1, lambda2, is_complex) = jac.eigenvalues();

    if is_complex {
        if lambda1 < 1.0 {
            StabilityType::Stable
        } else {
            StabilityType::Unstable
        }
    } else {
        let abs_lambda1 = lambda1.abs();
        let abs_lambda2 = lambda2.abs();

        if abs_lambda1 < 1.0 && abs_lambda2 < 1.0 {
            StabilityType::Stable
        } else if abs_lambda1 > 1.0 && abs_lambda2 > 1.0 {
            StabilityType::Unstable
        } else {
            StabilityType::Saddle
        }
    }
}

fn compose_henon_n_times(
    x0: f64,
    y0: f64,
    n: usize,
    a: f64,
    b: f64,
) -> ((f64, f64), Jacobian) {
    let mut x = x0;
    let mut y = y0;
    let mut accumulated_jacobian = henon_jacobian(x, y, a, b);

    for _ in 1..n {
        let (x_new, y_new) = henon_map(x, y, a, b);

        if !x_new.is_finite() || !y_new.is_finite() || x_new.abs() > 1e10 || y_new.abs() > 1e10 {
            return ((f64::NAN, f64::NAN), accumulated_jacobian);
        }

        let jac_current = henon_jacobian(x_new, y_new, a, b);
        accumulated_jacobian = jac_current.multiply(&accumulated_jacobian);
        x = x_new;
        y = y_new;
    }

    ((x, y), accumulated_jacobian)
}

fn find_periodic_point_near(
    x0: f64,
    y0: f64,
    period: usize,
    a: f64,
    b: f64,
    max_iterations: usize,
    tolerance: f64,
) -> Option<Point> {
    let mut x = x0;
    let mut y = y0;

    for _ in 0..max_iterations {
        if !x.is_finite() || !y.is_finite() || x.abs() > 100.0 || y.abs() > 100.0 {
            return None;
        }

        let ((fx, fy), jac_fn) = compose_henon_n_times(x, y, period, a, b);

        if !fx.is_finite() || !fy.is_finite() {
            return None;
        }

        let gx = fx - x;
        let gy = fy - y;

        if gx.abs() < tolerance && gy.abs() < tolerance {
            return Some(Point { x, y });
        }

        let jac_gn = Jacobian {
            j11: jac_fn.j11 - 1.0,
            j12: jac_fn.j12,
            j21: jac_fn.j21,
            j22: jac_fn.j22 - 1.0,
        };

        let jac_inv = jac_gn.inverse()?;

        let dx = -(jac_inv.j11 * gx + jac_inv.j12 * gy);
        let dy = -(jac_inv.j21 * gx + jac_inv.j22 * gy);

        if !dx.is_finite() || !dy.is_finite() {
            return None;
        }

        x += dx;
        y += dy;

        if dx.abs() < tolerance && dy.abs() < tolerance {
            break;
        }
    }

    None
}

fn verify_minimal_period(point: &Point, claimed_period: usize, a: f64, b: f64) -> bool {
    for divisor in 1..claimed_period {
        if claimed_period % divisor == 0 {
            let ((fx, fy), _) = compose_henon_n_times(point.x, point.y, divisor, a, b);
            if (fx - point.x).abs() < 1e-8 && (fy - point.y).abs() < 1e-8 {
                return false;
            }
        }
    }
    true
}

#[derive(Debug, Clone)]
pub struct PeriodicOrbitDatabase {
    pub orbits: Vec<PeriodicOrbit>,
}

impl PeriodicOrbitDatabase {
    fn new() -> Self {
        Self { orbits: Vec::new() }
    }

    fn add_orbit(&mut self, orbit: PeriodicOrbit) {
        self.orbits.push(orbit);
    }

    fn contains_point(&self, x: f64, y: f64, tolerance: f64) -> bool {
        self.orbits.iter().any(|orbit| {
            orbit
                .points
                .iter()
                .any(|p| (p.x - x).abs() < tolerance && (p.y - y).abs() < tolerance)
        })
    }

    fn find_matching_orbit(&self, x: f64, y: f64, tolerance: f64) -> Option<(usize, StabilityType, f64)> {
        for orbit in &self.orbits {
            for point in &orbit.points {
                let dist = ((point.x - x).powi(2) + (point.y - y).powi(2)).sqrt();
                if dist < tolerance {
                    return Some((orbit.period, orbit.stability.clone(), dist));
                }
            }
        }
        None
    }

    fn classify_point(&self, x: f64, y: f64, tolerance: f64) -> PointClassification {
        if let Some((period, stability, distance)) = self.find_matching_orbit(x, y, tolerance) {
            PointClassification::NearPeriodicOrbit {
                period,
                stability,
                distance,
            }
        } else {
            PointClassification::Regular
        }
    }

    pub fn total_count(&self) -> usize {
        self.orbits.len()
    }
}

fn precompute_periodic_orbits(
    a: f64,
    b: f64,
    max_period: usize,
) -> PeriodicOrbitDatabase {
    let mut database = PeriodicOrbitDatabase::new();

    for period in 1..=max_period {
        let grid_size = if period == 1 { 30 } else { 50 };
        let x_range = (-2.0, 2.0);
        let y_range = (-1.0, 1.0);

        for i in 0..grid_size {
            for j in 0..grid_size {
                let x0 = x_range.0 + (x_range.1 - x_range.0) * (i as f64) / (grid_size as f64);
                let y0 = y_range.0 + (y_range.1 - y_range.0) * (j as f64) / (grid_size as f64);

                if let Some(fixed_point) =
                    find_periodic_point_near(x0, y0, period, a, b, 100, 1e-10)
                {
                    if !database.contains_point(fixed_point.x, fixed_point.y, 1e-6) {
                        if verify_minimal_period(&fixed_point, period, a, b) {
                            let mut orbit_points = vec![fixed_point.clone()];
                            let mut x = fixed_point.x;
                            let mut y = fixed_point.y;

                            for _ in 1..period {
                                let (x_new, y_new) = henon_map(x, y, a, b);
                                orbit_points.push(Point { x: x_new, y: y_new });
                                x = x_new;
                                y = y_new;
                            }

                            let (_, jac_fn) =
                                compose_henon_n_times(fixed_point.x, fixed_point.y, period, a, b);
                            let stability = classify_stability(&jac_fn);

                            database.add_orbit(PeriodicOrbit {
                                points: orbit_points,
                                period,
                                stability,
                            });
                        }
                    }
                }
            }
        }
    }

    database
}

fn initialize_circular_boundary(
    center: (f64, f64),
    epsilon: f64,
    num_points: usize,
) -> Vec<BoundaryPoint> {
    (0..num_points)
        .map(|k| {
            let theta = 2.0 * f64::consts::PI * (k as f64) / (num_points as f64);
            let x = center.0 + epsilon * theta.cos();
            let y = center.1 + epsilon * theta.sin();
            let nx = theta.cos();
            let ny = theta.sin();
            BoundaryPoint {
                x,
                y,
                nx,
                ny,
                classification: PointClassification::Regular,
            }
        })
        .collect()
}

fn evolve_boundary_step(
    boundary: &[BoundaryPoint],
    a: f64,
    b: f64,
    epsilon: f64,
) -> Vec<BoundaryPoint> {
    boundary
        .iter()
        .filter(|p| p.x.is_finite() && p.y.is_finite() && p.x.abs() < 100.0 && p.y.abs() < 100.0)
        .map(|point| {
            let (fx, fy) = henon_map(point.x, point.y, a, b);

            if !fx.is_finite() || !fy.is_finite() || fx.abs() > 100.0 || fy.abs() > 100.0 {
                return BoundaryPoint {
                    x: fx.clamp(-100.0, 100.0),
                    y: fy.clamp(-100.0, 100.0),
                    nx: 0.0,
                    ny: 0.0,
                    classification: PointClassification::Regular,
                };
            }

            let jac = henon_jacobian(point.x, point.y, a, b);
            let jac_inv_transpose = jac.inverse().map(|j| Jacobian {
                j11: j.j11,
                j12: j.j21,
                j21: j.j12,
                j22: j.j22,
            });

            if let Some(jac_it) = jac_inv_transpose {
                let nx_new = jac_it.j11 * point.nx + jac_it.j12 * point.ny;
                let ny_new = jac_it.j21 * point.nx + jac_it.j22 * point.ny;
                let norm = (nx_new * nx_new + ny_new * ny_new).sqrt();

                if norm > 1e-12 && norm.is_finite() {
                    let nx_normalized = nx_new / norm;
                    let ny_normalized = ny_new / norm;

                    BoundaryPoint {
                        x: fx + epsilon * nx_normalized,
                        y: fy + epsilon * ny_normalized,
                        nx: nx_normalized,
                        ny: ny_normalized,
                        classification: PointClassification::Regular,
                    }
                } else {
                    BoundaryPoint {
                        x: fx,
                        y: fy,
                        nx: 0.0,
                        ny: 0.0,
                        classification: PointClassification::Regular,
                    }
                }
            } else {
                BoundaryPoint {
                    x: fx,
                    y: fy,
                    nx: 0.0,
                    ny: 0.0,
                    classification: PointClassification::Regular,
                }
            }
        })
        .collect()
}

fn compute_hausdorff_distance(set_a: &[BoundaryPoint], set_b: &[BoundaryPoint]) -> f64 {
    if set_a.is_empty() || set_b.is_empty() {
        console::log_1(&"Warning: Computing Hausdorff distance with empty set".into());
        return f64::INFINITY;
    }

    // Filter to only valid points
    let valid_a: Vec<_> = set_a
        .iter()
        .filter(|p| p.x.is_finite() && p.y.is_finite() && p.x.abs() < 100.0 && p.y.abs() < 100.0)
        .collect();
    
    let valid_b: Vec<_> = set_b
        .iter()
        .filter(|p| p.x.is_finite() && p.y.is_finite() && p.x.abs() < 100.0 && p.y.abs() < 100.0)
        .collect();

    if valid_a.is_empty() || valid_b.is_empty() {
        return f64::INFINITY;
    }

    let max_dist_a_to_b = valid_a
        .iter()
        .map(|pa| {
            valid_b
                .iter()
                .map(|pb| ((pa.x - pb.x).powi(2) + (pa.y - pb.y).powi(2)).sqrt())
                .fold(f64::INFINITY, |a, b| a.min(b))
        })
        .fold(0.0_f64, |a, b| a.max(b));

    let max_dist_b_to_a = valid_b
        .iter()
        .map(|pb| {
            valid_a
                .iter()
                .map(|pa| ((pb.x - pa.x).powi(2) + (pb.y - pa.y).powi(2)).sqrt())
                .fold(f64::INFINITY, |a, b| a.min(b))
        })
        .fold(0.0_f64, |a, b| a.max(b));

    max_dist_a_to_b.max(max_dist_b_to_a)
}


pub struct HenonSystemAnalysis {
    pub a: f64,
    pub b: f64,
    pub epsilon: f64,
    pub orbit_database: PeriodicOrbitDatabase,
    pub boundary_history: Vec<BoundarySnapshot>,
}

impl HenonSystemAnalysis {
    pub fn new(a: f64, b: f64, epsilon: f64, max_period: usize) -> Self {
        let orbit_database = precompute_periodic_orbits(a, b, max_period);
        Self {
            a,
            b,
            epsilon,
            orbit_database,
            boundary_history: Vec::new(),
        }
    }

    pub fn track_boundary(
    &mut self,
    initial_center: (f64, f64),
    num_boundary_points: usize,
    max_iterations: usize,
    convergence_tolerance: f64,
) {
    console::log_1(&format!("track_boundary called with center: {:?}, num_points: {}, max_iter: {}, tolerance: {}", 
        initial_center, num_boundary_points, max_iterations, convergence_tolerance).into());
    
    self.boundary_history.clear();
    
    // Validate inputs
    if num_boundary_points == 0 {
        console::log_1(&"Error: num_boundary_points must be > 0".into());
        return;
    }
    
    if max_iterations == 0 {
        console::log_1(&"Error: max_iterations must be > 0".into());
        return;
    }

    let mut current_boundary = initialize_circular_boundary(
        initial_center,
        self.epsilon,
        num_boundary_points,
    );

    if current_boundary.is_empty() {
        console::log_1(&"Error: Failed to initialize boundary".into());
        return;
    }

    current_boundary = self.classify_boundary(&current_boundary);

    self.boundary_history.push(BoundarySnapshot {
        iteration: 0,
        points: current_boundary.clone(),
    });

    for iter in 1..=max_iterations {
        let new_boundary = evolve_boundary_step(&current_boundary, self.a, self.b, self.epsilon);

        if new_boundary.is_empty() {
            console::log_1(&format!("Boundary became empty at iteration {}", iter).into());
            break;
        }

        let diverged_count = new_boundary
            .iter()
            .filter(|p| !p.x.is_finite() || !p.y.is_finite() || p.x.abs() > 100.0 || p.y.abs() > 100.0)
            .count();

        console::log_1(&format!("Iteration: {}, Boundary points: {}, Diverged: {}", 
            iter, new_boundary.len(), diverged_count).into());

        if diverged_count as f64 / new_boundary.len() as f64 > 0.5 {
            console::log_1(&format!("Too many points diverged at iteration {}, stopping", iter).into());
            break;
        }

        let valid_boundary: Vec<BoundaryPoint> = new_boundary
            .into_iter()
            .filter(|p| p.x.is_finite() && p.y.is_finite() && p.x.abs() < 100.0 && p.y.abs() < 100.0)
            .collect();

        if valid_boundary.is_empty() {
            console::log_1(&format!("No valid points remaining at iteration {}", iter).into());
            break;
        }

        let classified_boundary = self.classify_boundary(&valid_boundary);
        
        self.boundary_history.push(BoundarySnapshot {
            iteration: iter,
            points: classified_boundary.clone(),
        });

        if iter > 1 && self.boundary_history.len() >= 2 {
            let prev_idx = self.boundary_history.len() - 2;
            let prev_boundary = &self.boundary_history[prev_idx].points;
            
            if !prev_boundary.is_empty() && !classified_boundary.is_empty() {
                let hausdorff_distance = compute_hausdorff_distance(prev_boundary, &classified_boundary);

                console::log_1(&format!("Hausdorff distance: {}", hausdorff_distance).into());

                if hausdorff_distance.is_finite() && hausdorff_distance < convergence_tolerance {
                    console::log_1(&format!("Convergence achieved at iteration {}", iter).into());
                    break;
                }
            }
        }

        current_boundary = classified_boundary;
    }

    console::log_1(&format!("Tracking complete. Total snapshots: {}", self.boundary_history.len()).into());
}



 

    fn classify_boundary(&self, boundary: &[BoundaryPoint]) -> Vec<BoundaryPoint> {
        boundary
            .iter()
            .map(|point| {
                let classification =
                    self.orbit_database.classify_point(point.x, point.y, 1e-4);
                BoundaryPoint {
                    x: point.x,
                    y: point.y,
                    nx: point.nx,
                    ny: point.ny,
                    classification,
                }
            })
            .collect()
    }
}

#[derive(Serialize, Deserialize)]
pub struct BoundaryPointJS {
    pub x: f64,
    pub y: f64,
    pub nx: f64,
    pub ny: f64,
    pub classification: String,
    pub period: Option<usize>,
    pub stability: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct PeriodicOrbitJS {
    pub points: Vec<(f64, f64)>,
    pub period: usize,
    pub stability: String,
}

impl From<&StabilityType> for String {
    fn from(stability: &StabilityType) -> Self {
        match stability {
            StabilityType::Stable => "stable".to_string(),
            StabilityType::Unstable => "unstable".to_string(),
            StabilityType::Saddle => "saddle".to_string(),
        }
    }
}

impl From<&BoundaryPoint> for BoundaryPointJS {
    fn from(point: &BoundaryPoint) -> Self {
        match &point.classification {
            PointClassification::Regular => BoundaryPointJS {
                x: point.x,
                y: point.y,
                nx: point.nx,
                ny: point.ny,
                classification: "regular".to_string(),
                period: None,
                stability: None,
            },
            PointClassification::NearPeriodicOrbit {
                period,
                stability,
                distance: _,
            } => BoundaryPointJS {
                x: point.x,
                y: point.y,
                nx: point.nx,
                ny: point.ny,
                classification: "periodic".to_string(),
                period: Some(*period),
                stability: Some(String::from(stability)),
            },
        }
    }
}

#[wasm_bindgen]
pub struct HenonSystemWasm {
    system: HenonSystemAnalysis,
    current_iteration: usize,
}

#[wasm_bindgen]
impl HenonSystemWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(a: f64, b: f64, epsilon: f64, max_period: usize) -> Result<HenonSystemWasm, JsValue> {
        console_error_panic_hook::set_once();

        let system = HenonSystemAnalysis::new(a, b, epsilon, max_period);

        Ok(Self {
            system,
            current_iteration: 0,
        })
    }

    #[wasm_bindgen(js_name = getPeriodicOrbits)]
    pub fn get_periodic_orbits(&self) -> Result<JsValue, JsValue> {
        let orbits: Vec<PeriodicOrbitJS> = self
            .system
            .orbit_database
            .orbits
            .iter()
            .map(|orbit| PeriodicOrbitJS {
                points: orbit.points.iter().map(|p| (p.x, p.y)).collect(),
                period: orbit.period,
                stability: String::from(&orbit.stability),
            })
            .collect();

        serde_wasm_bindgen::to_value(&orbits)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    #[wasm_bindgen(js_name = trackBoundary)]
    pub fn track_boundary(
        &mut self,
        center_x: f64,
        center_y: f64,
        num_points: usize,
        max_iterations: usize,
        convergence_tolerance: f64,
    ) {
        self.system.track_boundary(
            (center_x, center_y),
            num_points,
            max_iterations,
            convergence_tolerance,
        );
        self.current_iteration = 0;
    }

    #[wasm_bindgen(js_name = getCurrentBoundary)]
    pub fn get_current_boundary(&self) -> Result<JsValue, JsValue> {
        if self.current_iteration < self.system.boundary_history.len() {
            let snapshot = &self.system.boundary_history[self.current_iteration];
            let points: Vec<BoundaryPointJS> =
                snapshot.points.iter().map(BoundaryPointJS::from).collect();

            serde_wasm_bindgen::to_value(&points)
                .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
        } else {
            Ok(JsValue::NULL)
        }
    }

    #[wasm_bindgen]
    pub fn step(&mut self) -> bool {
        if self.current_iteration + 1 < self.system.boundary_history.len() {
            self.current_iteration += 1;
            true
        } else {
            false
        }
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.current_iteration = 0;
    }

    #[wasm_bindgen(js_name = getTotalIterations)]
    pub fn get_total_iterations(&self) -> usize {
        self.system.boundary_history.len()
    }

    #[wasm_bindgen(js_name = getCurrentIteration)]
    pub fn get_current_iteration(&self) -> usize {
        self.current_iteration
    }

    #[wasm_bindgen(js_name = getOrbitCount)]
    pub fn get_orbit_count(&self) -> usize {
        self.system.orbit_database.total_count()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_henon_map() {
        let mut henon_system = HenonSystemAnalysis::new(1.4, 0.3, 0.01, 2);
        henon_system.track_boundary((0.0, 0.0), 128, 30, 1e-4);
        let boundaries: Vec<BoundarySnapshot> = henon_system.boundary_history;
        println!("Boundary history: {:?}", boundaries)
    }
}

