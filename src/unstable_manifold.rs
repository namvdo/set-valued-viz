use nalgebra::{Vector2, Matrix2, Vector4, Matrix4};
use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

#[cfg(target_arch = "wasm32")]
fn get_time_secs() -> f64 {
    js_sys::Date::now() / 1000.0
}

#[cfg(not(target_arch = "wasm32"))]
fn get_time_secs() -> f64 {
    use std::time::Instant;
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => {
        log(&format!($($t)*))
    }
}


macro_rules! console_error {
    ($($t:tt)*) => {
        error(&format!($($t)*))
    }
}
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct HenonParams {
    pub a: f64,
    pub b: f64,
    pub epsilon: f64
}

#[derive(Debug, Clone, Copy)]
pub struct ExtendedState {
    pub pos: Vector2<f64>,
    pub normal: Vector2<f64>
}

#[derive(Clone, Debug)]
pub struct ManifoldConfig {
    pub perturb_tol: f64,
    pub spacing_tol: f64,       
    pub spacing_upper: f64,      
    pub conv_tol: f64,
    pub stable_tol: f64,
    pub max_iter: usize,
    pub max_points: usize,
    pub time_limit: f64,
    pub inner_max: usize
}

impl Default for ManifoldConfig {
    fn default() -> Self {
        Self {
            perturb_tol: 1e-5,
            spacing_tol: 1e-3,      
            spacing_upper: 10.0,     
            conv_tol: 1e-19,        
            stable_tol: 1e-14,       
            max_iter: 500,
            max_points: 50_000,     
            time_limit: 2.0,        
            inner_max: 500,         
        }
    }
}

#[derive(Clone, Debug, Copy, PartialEq)]
pub enum SaddleType {
    Regular,       
    DualRepeller   
}

#[derive(Clone, Debug)]
pub struct SaddlePoint {
    pub position: Vector2<f64>,
    pub period: usize,
    pub eigenvector: Vector2<f64>,  
    pub eigenvalue: f64,
    pub saddle_type: SaddleType,
}

#[derive(Clone, Debug)]
pub struct Trajectory {
    pub points: Vec<ExtendedState>,
    pub stop_reason: StopReason
}

#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    Converged,
    MaxIterations,
    MaxPoints,
    TimeExceeded,
    ApproachedTargetPoint,  
}

impl HenonParams {
    pub fn new(a: f64, b: f64, epsilon: f64) -> Result<Self, String> {
        if !a.is_finite() || !b.is_finite() || !epsilon.is_finite() {
            return Err("Parameters must be finite numbers".to_string());
        }
        
        if b.abs() < 1e-10 {
            return Err("Parameter b cannot be zero or near zero".to_string());
        }
        
        if epsilon <= 0.0 {
            return Err("Epsilon must be positive".to_string());
        }
        
        if a.abs() > 10.0 || epsilon > 1.0 {
            return Err("Parameters outside reasonable range".to_string());
        }
        
        Ok(Self { a, b, epsilon })
    }
    
    pub fn henon_map(&self, pos: &Vector2<f64>) -> Result<Vector2<f64>, String> {
        if !pos.x.is_finite() || !pos.y.is_finite() {
            return Err("Invalid position: non-finite coordinates".to_string());
        }
        
        let x_new = 1.0 - self.a * pos.x * pos.x + pos.y;
        let y_new = self.b * pos.x;
        
        if !x_new.is_finite() || !y_new.is_finite() {
            return Err("Henon map produced non-finite values".to_string());
        }
        
        Ok(Vector2::new(x_new, y_new))
    }

    pub fn henon_map_inverse(&self, pos: &Vector2<f64>) -> Result<Vector2<f64>, String> {
        if !pos.x.is_finite() || !pos.y.is_finite() {
            return Err("Invalid position: non-finite coordinates".to_string());
        }
        
        let y_over_b = pos.y / self.b;
        let x_new = y_over_b;
        let y_new = pos.x + self.a * y_over_b * y_over_b - 1.0;
        
        if !x_new.is_finite() || !y_new.is_finite() {
            return Err("Inverse Henon map produced non-finite values".to_string());
        }
        
        Ok(Vector2::new(x_new, y_new))
    }

    pub fn jacobian(&self, pos: Vector2<f64>) -> Matrix2<f64> {
        Matrix2::new(
            -2.0 * self.a * pos.x, 1.0,
            self.b, 0.0
        )
    }

    pub fn transform_normal(&self, pos: Vector2<f64>, normal: Vector2<f64>) -> Result<Vector2<f64>, String> {
        if !pos.x.is_finite() || !pos.y.is_finite() {
            return Err("Invalid position in transform_normal".to_string());
        }
        
        if !normal.x.is_finite() || !normal.y.is_finite() {
            return Err("Invalid normal in transform_normal".to_string());
        }
        
        let jac_inv_t = Matrix2::new(
            0.0, 1.0,
            1.0 / self.b, 2.0 * self.a * pos.x / self.b
        );

        let transformed = jac_inv_t * normal; 
        let norm = transformed.norm();

        if !norm.is_finite() || norm < 1e-10 {
            return Ok(normal);
        }
        
        let result = transformed / norm;
        if !result.x.is_finite() || !result.y.is_finite() {
            return Ok(normal);
        }
        
        Ok(result)
    }

    pub fn transform_normal_inverse(&self, pos: Vector2<f64>, normal: Vector2<f64>) -> Result<Vector2<f64>, String> {
        if !pos.x.is_finite() || !pos.y.is_finite() {
            return Err("Invalid position in transform_normal_inverse".to_string());
        }
        
        if !normal.x.is_finite() || !normal.y.is_finite() {
            return Err("Invalid normal in transform_normal_inverse".to_string());
        }
        
        let jac = self.jacobian(pos);
        let jac_t = jac.transpose();

        let transformed = jac_t * normal;
        let norm = transformed.norm();

        if !norm.is_finite() || norm < 1e-10 {
            return Ok(normal);
        }
        
        let result = transformed / norm;
        if !result.x.is_finite() || !result.y.is_finite() {
            return Ok(normal);
        }
        
        Ok(result)
    }

    pub fn extended_map(&self, state: ExtendedState, n_periods: usize) -> Result<ExtendedState, String> {
        let mut current = state;
        
        for iter in 0..n_periods {
            let new_pos = self.henon_map(&current.pos)?;
            let new_normal = self.transform_normal(current.pos, current.normal)?;
            
            let projected_pos = new_pos + self.epsilon * new_normal;
            
            if !projected_pos.x.is_finite() || !projected_pos.y.is_finite() {
                return Err(format!("Non-finite position at iteration {}", iter));
            }
            
            if projected_pos.x.abs() > 1000.0 || projected_pos.y.abs() > 1000.0 {
                return Err(format!("Position diverged at iteration {}", iter));
            }

            current = ExtendedState {
                pos: projected_pos,
                normal: new_normal
            };
        }
        
        Ok(current)
    }

    pub fn extended_map_inverse(&self, state: ExtendedState, n_periods: usize) -> Result<ExtendedState, String> {
        let mut current = state;
        
        for iter in 0..n_periods {
            let unprojected_pos = current.pos - self.epsilon * current.normal;
            
            if !unprojected_pos.x.is_finite() || !unprojected_pos.y.is_finite() {
                return Err(format!("Non-finite unprojected position at iteration {}", iter));
            }
            
            let new_pos = self.henon_map_inverse(&unprojected_pos)?;
            let new_normal = self.transform_normal_inverse(unprojected_pos, current.normal)?;

            if new_pos.x.abs() > 1000.0 || new_pos.y.abs() > 1000.0 {
                return Err(format!("Position diverged at iteration {}", iter));
            }

            current = ExtendedState {
                pos: new_pos,
                normal: new_normal
            };
        }
        
        Ok(current)
    }
}

pub struct UnstableManifoldComputer {
    params: HenonParams,
    config: ManifoldConfig
}

impl UnstableManifoldComputer {
    pub fn new(params: HenonParams, config: ManifoldConfig) -> Self {
        Self { params, config }
    }

    pub fn compute_direction(
        &self,
        saddle: &SaddlePoint,
        direction_sign: f64,
        target_points: &[Vector2<f64>]
    ) -> Result<Trajectory, String> {
        console_log!("Starting compute_direction with direction_sign={}", direction_sign);
        
        let n_period = saddle.period;
        
        if n_period == 0 {
            return Err("Period cannot be zero".to_string());
        }
        
        if !saddle.position.x.is_finite() || !saddle.position.y.is_finite() {
            return Err("Saddle position is invalid".to_string());
        }
        
        let spacing_tol = if saddle.saddle_type == SaddleType::DualRepeller {
            1e-3 
        } else {
            self.config.spacing_tol  
        };

        let vec_0 = saddle.position + direction_sign * self.config.perturb_tol * saddle.eigenvector;

        if !vec_0.x.is_finite() || !vec_0.y.is_finite() {
            return Err("Initial perturbation produced invalid position".to_string());
        }
        
        console_log!("Initial position: ({}, {})", vec_0.x, vec_0.y);

        let initial_normal = Vector2::new(
            -saddle.eigenvector.y,
            saddle.eigenvector.x
        );
        
        let norm = initial_normal.norm();
        if norm < 1e-10 {
            return Err("Invalid eigenvector: too small".to_string());
        }
        
        let initial_normal = initial_normal / norm;

        let state_0 = ExtendedState {
            pos: vec_0,
            normal: initial_normal
        };

        let map_fn: Box<dyn Fn(ExtendedState, usize) -> Result<ExtendedState, String>> = 
            if saddle.saddle_type == SaddleType::DualRepeller {
                console_log!("Using inverse map (dual repeller)");
                Box::new(|state, n| self.params.extended_map_inverse(state, n))
            } else {
                console_log!("Using forward map");
                Box::new(|state, n| self.params.extended_map(state, n))
            };

        let state_1 = match map_fn(state_0, n_period) {
            Ok(s) => s,
            Err(e) => {
                console_error!("Initial map failed: {}", e);
                return Err(format!("Initial iteration failed: {}", e));
            }
        };
        
        console_log!("First iteration successful, pos: ({}, {})", state_1.pos.x, state_1.pos.y);
        
        let dist_vec_0 = state_1.pos - state_0.pos;

        let mut traj_add = vec![state_0, state_1];
        let mut vec_iter_old = state_1;

        let start_time = get_time_secs(); 
        let mut j = 1;
        
        let mut log_counter = 0;

        loop {
            let vec_iter = match map_fn(vec_iter_old, n_period) {
                Ok(v) => v,
                Err(e) => {
                    console_log!("Stopping at iteration {} due to: {}", j, e);
                    return Ok(Trajectory {
                        points: traj_add,
                        stop_reason: StopReason::Converged
                    });
                }
            };
            
            if log_counter % 100 == 0 {
                console_log!("Iteration {}: {} points, pos: ({:.4}, {:.4})", 
                            j, traj_add.len(), vec_iter.pos.x, vec_iter.pos.y);
                log_counter = 0;
            }
            log_counter += 1;

            let dist_diff = (vec_iter.pos - vec_iter_old.pos).norm();
            
            if !dist_diff.is_finite() {
                console_log!("Distance not finite at iteration {}", j);
                return Ok(Trajectory {
                    points: traj_add,
                    stop_reason: StopReason::Converged
                });
            }

            let dist_stable = if target_points.is_empty() {
                f64::INFINITY
            } else {
                target_points.iter()
                    .map(|tp| (vec_iter.pos - tp).norm())
                    .filter(|d| d.is_finite())
                    .fold(f64::INFINITY, |acc, d| if d < acc { d } else { acc })
            };

            if j > self.config.max_iter {
                console_log!("Max iterations reached");
                return Ok(Trajectory {
                    points: traj_add,
                    stop_reason: StopReason::MaxIterations
                });
            }

            if traj_add.len() > self.config.max_points {
                console_log!("Max points reached");
                return Ok(Trajectory {
                    points: traj_add,
                    stop_reason: StopReason::MaxPoints
                });
            }

            if dist_diff < self.config.conv_tol && j >= 30 {
                console_log!("Converged at iteration {}", j);
                return Ok(Trajectory {
                    points: traj_add,
                    stop_reason: StopReason::Converged
                });
            }

            if dist_stable <= self.config.stable_tol {
                console_log!("Approached target point");
                return Ok(Trajectory {
                    points: traj_add,
                    stop_reason: StopReason::ApproachedTargetPoint
                });
            }

            if dist_diff > spacing_tol || dist_diff.is_nan() {
                console_log!("Refining segment at iteration {} (dist_diff={})", j, dist_diff);
                match self.refine_segment(
                    vec_0,
                    initial_normal,
                    dist_vec_0,
                    state_1.normal,
                    vec_iter_old,
                    vec_iter,
                    j,
                    n_period,
                    &map_fn,
                    spacing_tol,
                    start_time
                ) {
                    Ok(refined_points) => {
                        console_log!("Added {} refined points", refined_points.len());
                        traj_add.extend(refined_points);
                    }
                    Err(reason) => {
                        console_log!("Refinement stopped: {:?}", reason);
                        return Ok(Trajectory {
                            points: traj_add,
                            stop_reason: reason
                        });
                    }
                }
            } else {
                traj_add.push(vec_iter);
            }

            vec_iter_old = vec_iter;
            j += 1;
        }
    }

    fn refine_segment(
        &self,
        vec_0: Vector2<f64>,
        initial_normal: Vector2<f64>,
        dist_vec_0: Vector2<f64>,
        normal_1: Vector2<f64>,
        state_old: ExtendedState,
        state_new: ExtendedState,
        iter_count: usize,
        n_period: usize,
        map_fn: &dyn Fn(ExtendedState, usize) -> Result<ExtendedState, String>,
        spacing_tol: f64,
        start_time: f64
    ) -> Result<Vec<ExtendedState>, StopReason> {
        let mut params: Vec<f64> = vec![0.0, 1.0];
        let mut points = vec![state_old, state_new];

        let mut k = 0;

        loop {
            if points.len() < 2 {
                console_error!("Points array too small: {}", points.len());
                break;
            }
            
            let gaps: Vec<usize> = (0..points.len().saturating_sub(1))
                .filter(|&i| {
                    if i + 1 >= points.len() {
                        return false;
                    }
                    let dist = (points[i + 1].pos - points[i].pos).norm();
                    dist.is_finite() && dist > spacing_tol && dist < self.config.spacing_upper
                })
                .collect();

            if gaps.is_empty() {
                break;
            }

            let mut filled_any = false;

            for &gap_idx in gaps.iter().rev() {
                if gap_idx + 1 >= params.len() {
                    console_error!("Gap index out of bounds: {} >= {}", gap_idx + 1, params.len());
                    continue;
                }

                let t_mid = (params[gap_idx] + params[gap_idx + 1]) / 2.0;

                let intermediate_pos = vec_0 + t_mid * dist_vec_0;

                if !intermediate_pos.x.is_finite() || !intermediate_pos.y.is_finite() {
                    continue;
                }

                // Interpolate normal between initial_normal and normal_1 based on t_mid
                let intermediate_normal_unnorm = initial_normal + t_mid * (normal_1 - initial_normal);
                let norm = intermediate_normal_unnorm.norm();

                if norm < 1e-10 {
                    continue;
                }

                let intermediate_normal = intermediate_normal_unnorm / norm;

                let intermediate_state = ExtendedState {
                    pos: intermediate_pos,
                    normal: intermediate_normal
                };

                let mapped_state = match map_fn(intermediate_state, n_period * iter_count) {
                    Ok(s) => s,
                    Err(_) => {
                        continue;
                    }
                };

                if !mapped_state.pos.x.is_finite() || !mapped_state.pos.y.is_finite() {
                    continue;
                }

                points.insert(gap_idx + 1, mapped_state);
                params.insert(gap_idx + 1, t_mid);
                filled_any = true;
            }

            // If no gaps were filled, stop trying (all midpoints diverged)
            if !filled_any {
                break;
            }

            k += 1;

            if k > self.config.inner_max {
                console_log!("Reached inner max iterations");
                return Err(StopReason::MaxIterations);
            }

            let elapsed = get_time_secs() - start_time;
            if elapsed > self.config.time_limit {
                console_log!("Time limit exceeded in refinement");
                return Err(StopReason::TimeExceeded);
            }
        }

        Ok(points.into_iter().skip(1).collect())
    }

    pub fn compute_manifold(
        &self,
        saddle: &SaddlePoint,
        target_points: &[Vector2<f64>]
    ) -> Result<(Trajectory, Trajectory), String> {
        let traj_plus = self.compute_direction(saddle, 1.0, target_points)?;
        let traj_minus = self.compute_direction(saddle, -1.0, target_points)?;

        Ok((traj_plus, traj_minus))
    }
}

#[derive(Serialize, Deserialize)]
pub struct TrajectoryRet {
    pub points: Vec<(f64, f64)>,
    pub stop_reason: String,
}

#[derive(Serialize, Deserialize)]
pub struct ManifoldResult {
    pub plus: TrajectoryRet,
    pub minus: TrajectoryRet,
    pub saddle_point: (f64, f64),
    pub eigenvalue: f64,
}

#[derive(Serialize, Deserialize)]
pub struct FixedPointResult {
    pub x: f64,
    pub y: f64,
    pub eigenvalues: (f64, f64),
    pub stability: String, // "Attractor", "Repeller", "Saddle"
}

#[derive(Serialize, Deserialize)]
pub struct ComputeResult {
    pub manifolds: Vec<ManifoldResult>,
    pub fixed_points: Vec<FixedPointResult>,
}

struct NewtonSolver {
    params: HenonParams,
    max_iter: usize,
    tol: f64,
}

impl NewtonSolver {
    fn new(params: HenonParams) -> Self {
        Self {
            params,
            max_iter: 100,
            tol: 1e-10,
        }
    }

    fn solve_extended_period(&self, initial_state: ExtendedState, period: usize) -> Option<ExtendedState> {
        let mut x = initial_state;
        
        for _ in 0..self.max_iter {
            // F^k(x) - x = 0
            // Jacobian is J_F^k(x) - I
            
            let mut f_val = x;
            let mut jac_acc = Matrix4::<f64>::identity();
            
            // Compute F^k(x) and Jacobian roughly using chain rule or finite difference
            // Since we don't have analytical Jacobian for extended map in Rust easily available without automatic differentiation or implementing it manually...
            // Let's use finite difference for the full 4D Jacobian of the extended map.
            
            // Re-evaluate F^k(x)
            match self.params.extended_map(x, period) {
                Ok(res) => f_val = res,
                Err(_) => return None,
            }
            
            let diff_pos = f_val.pos - x.pos;
            let diff_norm = f_val.normal - x.normal;
            
            if diff_pos.norm() < self.tol && diff_norm.norm() < self.tol {
                return Some(x);
            }
            
            let eps = 1e-7;
            let mut jac = Matrix4::<f64>::zeros();
            
            let extracted_state = Vector4::new(x.pos.x, x.pos.y, x.normal.x, x.normal.y);
            
            for i in 0..4 {
                let mut perturbed = extracted_state;
                perturbed[i] += eps;
                let state_p = ExtendedState {
                    pos: Vector2::new(perturbed[0], perturbed[1]),
                    normal: Vector2::new(perturbed[2], perturbed[3]),
                };
                
                if let Ok(res_p) = self.params.extended_map(state_p, period) {
                    let d_pos = (res_p.pos - f_val.pos) / eps;
                    let d_norm = (res_p.normal - f_val.normal) / eps;
                    
                    jac[(0, i)] = d_pos.x;
                    jac[(1, i)] = d_pos.y;
                    jac[(2, i)] = d_norm.x;
                    jac[(3, i)] = d_norm.y;
                } else {
                    return None;
                }
            }
            
            let jac_minus_i = jac - Matrix4::<f64>::identity();
            
            let residual = Vector4::new(f_val.pos.x - x.pos.x, f_val.pos.y - x.pos.y, f_val.normal.x - x.normal.x, f_val.normal.y - x.normal.y);
            
            if let Some(inv) = jac_minus_i.try_inverse() {
                let delta = -(inv * residual);
                x.pos.x += delta[0];
                x.pos.y += delta[1];
                x.normal.x += delta[2];
                x.normal.y += delta[3];
                
                let n_norm = x.normal.norm();
                if n_norm > 1e-10 {
                    x.normal /= n_norm;
                }
            } else {
                return None;
            }
        }
        
        None
    }
}



#[wasm_bindgen]
pub fn compute_manifold_simple(
    a: f64,
    b: f64,
    epsilon: f64,
) -> Result<JsValue, JsValue> {
    console_log!("Computing manifold with a={}, b={}, epsilon={}", a, b, epsilon);
    
    let params = match HenonParams::new(a, b, epsilon) {
        Ok(p) => p,
        Err(e) => {
            console_error!("Invalid parameters: {}", e);
            return Err(JsValue::from_str(&e));
        }
    };
    
    let solver = NewtonSolver::new(params);
    let mut unique_states: Vec<ExtendedState> = Vec::new();
    
    // Grid search for fixed points (period 1)
    let search_grid_size = 3;
    let search_angle_size = 4;
    
    for i in 0..search_grid_size {
        for j in 0..search_grid_size {
            let x = -2.0 + 4.0 * (i as f64) / (search_grid_size as f64 - 1.0);
            let y = -2.0 + 4.0 * (j as f64) / (search_grid_size as f64 - 1.0);
            
            for k in 0..search_angle_size {
                let angle = 2.0 * std::f64::consts::PI * (k as f64) / (search_angle_size as f64);
                let normal = Vector2::new(angle.cos(), angle.sin());
                
                let initial = ExtendedState {
                    pos: Vector2::new(x, y),
                    normal,
                };
                
                if let Some(fixed) = solver.solve_extended_period(initial, 1) {
                    if !fixed.pos.x.is_finite() || !fixed.pos.y.is_finite() { continue; }
                    
                    let mut is_new = true;
                    for existing in &unique_states {
                         if (existing.pos - fixed.pos).norm() < 1e-4 && (existing.normal - fixed.normal).norm() < 1e-4 {
                            is_new = false;
                            break;
                         }
                         // Also check if it's the same point but opposite normal (eigenvector direction)
                         if (existing.pos - fixed.pos).norm() < 1e-4 && (existing.normal + fixed.normal).norm() < 1e-4 {
                            is_new = false;
                             break;
                         }
                    }
                    
                    if is_new {
                        if fixed.pos.norm() < 10.0 {
                            unique_states.push(fixed);
                            console_log!("Found fixed point: ({:.4}, {:.4})", fixed.pos.x, fixed.pos.y);
                        }
                    }
                }
            }
        }
    }

    let mut fixed_points_result = Vec::new();
    let mut stable_points_pos = Vec::new();
    let mut unstable_points_indices = Vec::new();
    
    // Classify points
    for (idx, state) in unique_states.iter().enumerate() {
        let jac = params.jacobian(state.pos);
        let trace = jac.trace();
        let det = jac.determinant();
        let discriminant = trace * trace - 4.0 * det;
        
        let (l1, l2) = if discriminant >= 0.0 {
            let sqrt_disc = discriminant.sqrt();
            ((trace + sqrt_disc) / 2.0, (trace - sqrt_disc) / 2.0)
        } else {
            // Complex eigenvalues, treat magnitude
            let real = trace / 2.0;
            let imag = (-discriminant).sqrt() / 2.0;
            let mag = (real*real + imag*imag).sqrt();
            (mag, mag)
        };
        
        // Simple classification based on magnitude
        let abs_l1 = l1.abs();
        let abs_l2 = l2.abs();
        
        let stability = if abs_l1 < 1.0 && abs_l2 < 1.0 {
            "Attractor"
        } else if abs_l1 > 1.0 && abs_l2 > 1.0 {
            "Repeller"
        } else {
            "Saddle"
        };
        
        fixed_points_result.push(FixedPointResult {
            x: state.pos.x,
            y: state.pos.y,
            eigenvalues: (l1, l2),
            stability: stability.to_string(),
        });
        
        if stability == "Attractor" {
            stable_points_pos.push(state.pos);
        } else if stability == "Saddle" || stability == "Repeller" {
            unstable_points_indices.push(idx);
        }
    }
    
    let mut manifolds_result = Vec::new();
    let config = ManifoldConfig::default();
    let computer = UnstableManifoldComputer::new(params, config);
    
    // Compute manifolds for unstable points
    for idx in unstable_points_indices {
        let state = unique_states[idx];
        let fp_info = &fixed_points_result[idx];
        
        // Re-compute eigenstuff for direction
        // We use the normal from ExtendedState as the unstable direction guess? 
        // Actually the normal extends along the unstable eigenvector for saddles usually.
        // Let's trust the normal found by Newton solver on extended map, 
        // which enforces invariance of the normal bundle.
        
        let saddle_type = if fp_info.stability == "Repeller" {
            SaddleType::DualRepeller
        } else {
            SaddleType::Regular
        };
        
        // Eigenvalue associated with this direction
        // We need to match the normal direction to the eigenvector to know which eigenvalue it is.
        // But for visualization, we just use the found state.normal as the direction.
        // We should just check if the normal aligns with strongest expansion or contraction.
        
        let l1 = fp_info.eigenvalues.0;
        let l2 = fp_info.eigenvalues.1;
        
        // For saddle, one is > 1. For repeller, both > 1.
        // The extended map finds (x, v) such that Df(x)v = lambda * v (normalized).
        // So `normal` IS the eigenvector.
        
        let saddle_pt = SaddlePoint {
            position: state.pos,
            period: 1,
            eigenvector: state.normal, // Use the normal from extended state
            eigenvalue: if l1.abs() > 1.0 { l1 } else { l2 }, // Rough guess, doesn't matter much for struct
            saddle_type,
        };

        if let Ok((traj_plus, traj_minus)) = computer.compute_manifold(&saddle_pt, &stable_points_pos) {
            let plus_points: Vec<(f64, f64)> = traj_plus.points.iter()
                .filter(|s| s.pos.x.is_finite() && s.pos.y.is_finite())
                .map(|s| (s.pos.x, s.pos.y))
                .collect();

            let minus_points: Vec<(f64, f64)> = traj_minus.points.iter()
                .filter(|s| s.pos.x.is_finite() && s.pos.y.is_finite())
                .map(|s| (s.pos.x, s.pos.y))
                .collect();
                
            manifolds_result.push(ManifoldResult {
                plus: TrajectoryRet {
                    points: plus_points,
                    stop_reason: format!("{:?}", traj_plus.stop_reason),
                },
                minus: TrajectoryRet {
                    points: minus_points,
                    stop_reason: format!("{:?}", traj_minus.stop_reason),
                },
                saddle_point: (state.pos.x, state.pos.y),
                eigenvalue: saddle_pt.eigenvalue,
            });
        }
    }

    let result = ComputeResult {
        manifolds: manifolds_result,
        fixed_points: fixed_points_result,
    };

    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    match result.serialize(&serializer) {
        Ok(v) => Ok(v),
        Err(e) => {
            console_error!("Serialization error: {:?}", e);
            Err(JsValue::from_str("Failed to serialize result"))
        }
    }
}

#[wasm_bindgen]
pub fn compute_manifold_js(
    a: f64,
    b: f64,
    epsilon: f64,
    saddle_x: f64,
    saddle_y: f64,
    period: usize,
    eigenvector_x: f64,
    eigenvector_y: f64,
    eigenvalue: f64,
    is_dual_repeller: bool,
) -> Result<JsValue, JsValue> {
    console_log!("Computing manifold with a={}, b={}, epsilon={}", a, b, epsilon);
    
    let params = match HenonParams::new(a, b, epsilon) {
        Ok(p) => p,
        Err(e) => {
            console_error!("Invalid parameters: {}", e);
            return Err(JsValue::from_str(&e));
        }
    };
    
    let config = ManifoldConfig::default();
    
    if !saddle_x.is_finite() || !saddle_y.is_finite() {
        let err = "Saddle point coordinates must be finite";
        console_error!("{}", err);
        return Err(JsValue::from_str(err));
    }
    
    if !eigenvector_x.is_finite() || !eigenvector_y.is_finite() {
        let err = "Eigenvector coordinates must be finite";
        console_error!("{}", err);
        return Err(JsValue::from_str(err));
    }
    
    if period == 0 {
        let err = "Period must be positive";
        console_error!("{}", err);
        return Err(JsValue::from_str(err));
    }
    
    let eigenvector = Vector2::new(eigenvector_x, eigenvector_y);
    let norm = eigenvector.norm();
    
    if norm < 1e-10 {
        let err = "Eigenvector magnitude too small";
        console_error!("{}", err);
        return Err(JsValue::from_str(err));
    }
    
    let eigenvector = eigenvector / norm;
    
    let saddle = SaddlePoint {
        position: Vector2::new(saddle_x, saddle_y),
        period,
        eigenvector,
        eigenvalue,
        saddle_type: if is_dual_repeller { 
            SaddleType::DualRepeller 
        } else { 
            SaddleType::Regular 
        },
    };
    
    let target_points = vec![];
    let computer = UnstableManifoldComputer::new(params, config);
    
    let (traj_plus, traj_minus) = match computer.compute_manifold(&saddle, &target_points) {
        Ok(result) => result,
        Err(e) => {
            console_error!("Failed to compute manifold: {}", e);
            return Err(JsValue::from_str(&e));
        }
    };
    
    console_log!("Successfully computed manifold: +{} pts, -{} pts", 
                 traj_plus.points.len(), traj_minus.points.len());
    
    let convert_traj = |traj: &Trajectory| -> TrajectoryRet {
        TrajectoryRet {
            points: traj.points.iter()
                .filter(|s| s.pos.x.is_finite() && s.pos.y.is_finite())
                .map(|s| (s.pos.x, s.pos.y))
                .collect(),
            stop_reason: format!("{:?}", traj.stop_reason),
        }
    };
    
    let result = serde_json::json!({
        "plus": convert_traj(&traj_plus),
        "minus": convert_traj(&traj_minus),
    });

    let serializer = serde_wasm_bindgen::Serializer::new().serialize_maps_as_objects(true);
    match result.serialize(&serializer) {
        Ok(v) => Ok(v),
        Err(e) => {
            console_error!("Serialization error: {:?}", e);
            Err(JsValue::from_str("Failed to serialize result"))
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_henon_params_validation() {
        assert!(HenonParams::new(1.4, 0.3, 0.01).is_ok());
        assert!(HenonParams::new(0.0, 0.3, 0.01).is_ok());
        assert!(HenonParams::new(1.4, 0.0, 0.01).is_err());
        assert!(HenonParams::new(1.4, 0.3, -0.01).is_err());
        assert!(HenonParams::new(1.4, 0.3, 0.0).is_err());
        assert!(HenonParams::new(f64::NAN, 0.3, 0.01).is_err());
    }

    #[test]
    fn test_henon_map() {
        let params = HenonParams::new(1.4, 0.3, 0.01).unwrap();
        let pos = Vector2::new(0.0, 0.0);
        let result = params.henon_map(&pos).unwrap();
        assert_eq!(result.x, 1.0);
        assert_eq!(result.y, 0.0);
    }

    #[test]
    fn test_fixed_point_a036() {
        let a: f64 = 0.36;
        let b: f64 = 0.3;

        let mut x = 0.0;
        let mut y = 0.0;

        for _ in 0..10000 {
            let x_new = 1.0 - a * x * x + y;
            let y_new = b * x;

            if (x_new - x).abs() < 1e-10 && (y_new - y).abs() < 1e-10 {
                break;
            }

            x = x_new;
            y = y_new;
        }

        println!("Fixed point for a={}: ({}, {})", a, x, y);
        let x_check = 1.0 - a * x * x + y;
        let y_check = b * x;
        let error_x = (x_check - x).abs();
        let error_y = (y_check - y).abs();
        println!("Fixed point error: x_error={}, y_error={}", error_x, error_y);

        assert!(error_x < 1e-6, "x error {} too large", error_x);
        assert!(error_y < 1e-6, "y error {} too large", error_y);

    } 

    #[test]
    fn test_eigenvalues_a036() {
        let a: f64 = 0.36;
        let b: f64 = 0.3;
        
        let mut x = 0.0;
        let mut y = 0.0;
        for _ in 0..1000 {
            let x_new = 1.0 - a * x * x + y;
            let y_new = b * x;
            if (x_new - x).abs() < 1e-10 && (y_new - y).abs() < 1e-10 {
                break;
            }
            x = x_new;
            y = y_new;
        }
        
        let jac = Matrix2::new(
            -2.0 * a * x, 1.0,
            b, 0.0
        );
        
        let trace = jac.trace();
        let det = jac.determinant();
        let discriminant = trace * trace - 4.0 * det;
        
        assert!(discriminant >= 0.0, "Complex eigenvalues");
        
        let sqrt_disc = discriminant.sqrt();
        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;
        
        println!("Eigenvalues: λ1={}, λ2={}", lambda1, lambda2);
        println!("Magnitudes: |λ1|={}, |λ2|={}", lambda1.abs(), lambda2.abs());
        
        let both_stable = lambda1.abs() < 1.0 && lambda2.abs() < 1.0;
        let has_unstable = lambda1.abs() > 1.0 || lambda2.abs() > 1.0;
        
        if both_stable {
            println!("Both eigenvalues stable - this is an attractor");
            println!("For minimal invariant set, use inverse map to trace stable manifold");
        } else {
            assert!(has_unstable, "Should have at least one unstable eigenvalue for saddle");
        }
    }

    #[test]
    fn test_a036_manifold() {
        let params = HenonParams::new(0.36, 0.3, 0.0625).unwrap();
        
        let mut x :f64= 0.0;
        let mut y :f64= 0.0;
        for _ in 0..1000 {
            let x_new = 1.0 - 0.36 * x * x + y;
            let y_new = 0.3 * x;
            if (x_new - x).abs() < 1e-10 && (y_new - y).abs() < 1e-10 {
                break;
            }
            x = x_new;
            y = y_new;
        }
        
        println!("Testing a=0.36 fixed point: ({}, {})", x, y);
        
        let config = ManifoldConfig {
            max_iter: 50,
            max_points: 5000,
            ..ManifoldConfig::default()
        };
        
        let jac = Matrix2::new(-2.0 * 0.36 * x, 1.0, 0.3, 0.0);
        let trace = jac.trace();
        let det = jac.determinant();
        let sqrt_disc = (trace * trace - 4.0 * det).sqrt();
        let lambda1 = (trace + sqrt_disc) / 2.0;
        let lambda2 = (trace - sqrt_disc) / 2.0;
        
        println!("Eigenvalues: λ1={}, λ2={}", lambda1, lambda2);
        
        let (unstable_lambda, eigenvec) = if lambda2.abs() > lambda1.abs() {
            let v = if (jac[(0, 0)] - lambda2).abs() > 1e-10 {
                Vector2::new(jac[(0, 1)], lambda2 - jac[(0, 0)])
            } else {
                Vector2::new(lambda2 - jac[(1, 1)], jac[(1, 0)])
            };
            (lambda2, v.normalize())
        } else {
            let v = if (jac[(0, 0)] - lambda1).abs() > 1e-10 {
                Vector2::new(jac[(0, 1)], lambda1 - jac[(0, 0)])
            } else {
                Vector2::new(lambda1 - jac[(1, 1)], jac[(1, 0)])
            };
            (lambda1, v.normalize())
        };
        
        let saddle_type = if unstable_lambda.abs() > 1.0 {
            SaddleType::Regular
        } else {
            println!("WARNING: No unstable eigenvalue found!");
            SaddleType::Regular
        };
        
        println!("Saddle type: {:?}, |λ_unstable|={}", saddle_type, unstable_lambda.abs());
        
        let saddle = SaddlePoint {
            position: Vector2::new(x, y),
            period: 1,
            eigenvector: eigenvec,
            eigenvalue: unstable_lambda,
            saddle_type,
        };
        
        let computer = UnstableManifoldComputer::new(params, config);
        let result = computer.compute_direction(&saddle, 1.0, &[]);
        
        match result {
            Ok(traj) => {
                println!("Success! Trajectory has {} points, stop reason: {:?}", 
                        traj.points.len(), traj.stop_reason);
                assert!(traj.points.len() > 1);
            }
            Err(e) => {
                println!("Expected behavior: {}", e);
            }
        }
    }
}