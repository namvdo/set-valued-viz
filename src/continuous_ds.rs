use nalgebra::{Matrix2, Vector2};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::dynamical_systems::{DynamicalSystem, ExtendedState};
use crate::unstable_manifold::{
    ManifoldConfig, SaddlePoint, SaddleType, StopReason, Trajectory, UnstableManifoldComputer,
};

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

#[derive(Clone, Debug)]
pub struct DuffingODE {
    pub delta: f64,
}

impl DuffingODE {
    pub fn new(delta: f64) -> Result<Self, String> {
        if !delta.is_finite() {
            return Err("Damping delta must be finite".to_string());
        }
        if delta < 0.0 {
            return Err("Damping delta should typically be non-negative".to_string());
        }
        Ok(Self { delta })
    }

    /// continuous vector field: f(x) = (x2, x1 - x1^3 - delta * x2)
    pub fn vector_field(&self, pos: Vector2<f64>) -> Result<Vector2<f64>, String> {
        if !pos.x.is_finite() || !pos.y.is_finite() {
            return Err("Invalid position: non finite coordinates".to_string());
        }
        let x_dot = pos.y;
        let y_dot = pos.x - pos.x.powi(3) - self.delta * pos.y;
        Ok(Vector2::new(x_dot, y_dot))
    }

    /// continous Jacobian: Df(x) = [[0, 1], [1 - 3*x1^2, -delta]]
    pub fn jacobian(&self, pos: Vector2<f64>) -> Matrix2<f64> {
        Matrix2::new(0.0, 1.0, 1.0 - 3.0 * pos.x.powi(2), -self.delta)
    }

    /// RK4 step: x_{n+1} = x_n + (h/6) * (k1 + 2k2 + 2k3 + k4)
    pub fn rk4_step(&self, pos: Vector2<f64>, h: f64) -> Result<Vector2<f64>, String> {
        let k1 = self.vector_field(pos)?;
        let k2 = self.vector_field(pos + k1 * (h / 2.0))?;
        let k3 = self.vector_field(pos + k2 * (h / 2.0))?;
        let k4 = self.vector_field(pos + k3 * h)?;
        let next_pos = pos + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * (h / 6.0);
        if !next_pos.x.is_finite() || !next_pos.y.is_finite() {
            return Err("RK4 step produced non-finite values".to_string());
        }
        Ok(next_pos)
    }
}

#[derive(Clone, Debug)]
pub struct EulerMap {
    pub ode: DuffingODE,
    pub h: f64,
    pub epsilon: f64,
}

impl EulerMap {
    pub fn new(ode: DuffingODE, h: f64, epsilon: f64) -> Result<Self, String> {
        if !h.is_finite() || h <= 0.0 {
            return Err("Step size h must be finite and positive".to_string());
        }
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err("Epsilon must be positive and finite".to_string());
        }
        Ok(Self { ode, h, epsilon })
    }

    /// Euler map: F_h(x) = x + h * f(x)
    pub fn euler_step(&self, pos: Vector2<f64>) -> Result<Vector2<f64>, String> {
        let f_val = self.ode.vector_field(pos)?;
        let next_pos = pos + self.h * f_val;
        if !next_pos.x.is_finite() || !next_pos.y.is_finite() {
            return Err("Euler map produced non-finite values".to_string());
        }
        Ok(next_pos)
    }

    /// inverse Euler map map_inverse(y) using Newton iteration on y = x + h*f(x)
    pub fn euler_step_inverse(&self, pos: Vector2<f64>) -> Result<Vector2<f64>, String> {
        let mut x = pos;
        for _ in 0..100 {
            let f_val = self.ode.vector_field(x)?;
            let diff = x + self.h * f_val - pos;
            if diff.norm() < 1e-10 {
                return Ok(x);
            }
            let df = self.jacobian(x);
            let dx = df.try_inverse().ok_or("Singular Jacobian in inverse Euler map")? * diff;
            x -= dx;
        }
        Err("Inverse Euler map failed to converge".to_string())
    }

    /// linearization of Euler map: DF_h(x) = I + h * Df(x)
    pub fn jacobian(&self, pos: Vector2<f64>) -> Matrix2<f64> {
        let df = self.ode.jacobian(pos);
        Matrix2::identity() + self.h * df
    }
}

impl DynamicalSystem for EulerMap {
    fn map(&self, pos: Vector2<f64>) -> Result<Vector2<f64>, String> {
        self.ode.rk4_step(pos, self.h)
    }

    fn map_inverse(&self, pos: Vector2<f64>) -> Result<Vector2<f64>, String> {
        self.euler_step_inverse(pos)
    }

    fn jacobian(&self, pos: Vector2<f64>) -> Matrix2<f64> {
        self.jacobian(pos)
    }

    fn get_epsilon(&self) -> f64 {
        self.h * self.epsilon
    }
}

pub fn bde_rhs(ode: &DuffingODE, eps: f64, x: Vector2<f64>, nhat: Vector2<f64>) -> Result<(Vector2<f64>, Vector2<f64>), String> {
    let fx = ode.vector_field(x)?;
    let dx = Vector2::new(fx.x + eps * nhat.x, fx.y + eps * nhat.y);
    
    let jac = ode.jacobian(x);
    // Df(x)^T * n
    let jfn_x = jac.m11 * nhat.x + jac.m21 * nhat.y;
    let jfn_y = jac.m12 * nhat.x + jac.m22 * nhat.y;
    
    // v = -Df^T n
    let vx = -jfn_x;
    let vy = -jfn_y;
    
    // <n, v>
    let proj = vx * nhat.x + vy * nhat.y;
    
    // dn = v - proj * nhat
    let dn = Vector2::new(vx - proj * nhat.x, vy - proj * nhat.y);
    
    Ok((dx, dn))
}

// Approximate the next point using RK4 method
pub fn rk4_bde_step(ode: &DuffingODE, eps: f64, h: f64, x: Vector2<f64>, nhat: Vector2<f64>) -> Result<(Vector2<f64>, Vector2<f64>), String> {
    let (k1x, k1n) = bde_rhs(ode, eps, x, nhat)?;
    
    let x2 = x + k1x * (h / 2.0);
    let n2 = (nhat + k1n * (h / 2.0)).normalize();
    let (k2x, k2n) = bde_rhs(ode, eps, x2, n2)?;
    
    let x3 = x + k2x * (h / 2.0);
    let n3 = (nhat + k2n * (h / 2.0)).normalize();
    let (k3x, k3n) = bde_rhs(ode, eps, x3, n3)?;
    
    let x4 = x + k3x * h;
    let n4 = (nhat + k3n * h).normalize();
    let (k4x, k4n) = bde_rhs(ode, eps, x4, n4)?;
    
    let xp = x + (k1x + k2x * 2.0 + k3x * 2.0 + k4x) * (h / 6.0);
    let np = (nhat + (k1n + k2n * 2.0 + k3n * 2.0 + k4n) * (h / 6.0)).normalize();
    
    Ok((xp, np))
}

/// find fixed points of the ODE (which are fixed points of the Euler map)
/// duffing ODE: x2 = 0, x1 - x1^3 - delta * x2 = 0
/// => x2 = 0, x1 * (1 - x1^2) = 0
/// => x = (0, 0) or x = (+1, 0) or x = (-1, 0)
fn find_euler_map_fixed_points() -> Vec<(f64, f64)> {
    vec![(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0)]
}

/// classify stability of a fixed point based on discrete-time Jacobian eigenvalues
fn classify_euler_map_stability(l1: f64, l2: f64) -> &'static str {
    let abs_l1 = l1.abs();
    let abs_l2 = l2.abs();

    if abs_l1 < 1.0 && abs_l2 < 1.0 {
        "Attractor"
    } else if abs_l1 > 1.0 && abs_l2 > 1.0 {
        "Repeller"
    } else {
        "Saddle"
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EulerMapTrajectoryRet {
    pub points: Vec<(f64, f64)>,
    pub stop_reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EulerMapManifoldResult {
    pub plus: EulerMapTrajectoryRet,
    pub minus: EulerMapTrajectoryRet,
    pub saddle_point: (f64, f64),
    pub eigenvalue: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EulerMapFixedPointResult {
    pub x: f64,
    pub y: f64,
    pub eigenvalues: (f64, f64),
    pub stability: String,
}

#[derive(Serialize, Deserialize)]
pub struct EulerMapComputeResult {
    pub manifolds: Vec<EulerMapManifoldResult>,
    pub fixed_points: Vec<EulerMapFixedPointResult>,
}

#[wasm_bindgen]
pub fn compute_euler_map_manifold_simple(
    delta: f64,
    h: f64,
    epsilon: f64,
) -> Result<JsValue, JsValue> {
    console_log!(
        "Computing Euler map manifold with delta={}, h={}, epsilon={}",
        delta,
        h,
        epsilon
    );

    let ode = match DuffingODE::new(delta) {
        Ok(ode) => ode,
        Err(e) => {
            console_error!("Invalid ODE parameters: {}", e);
            return Err(JsValue::from_str(&e));
        }
    };

    let params = match EulerMap::new(ode, h, epsilon) {
        Ok(p) => p,
        Err(e) => {
            console_error!("Invalid Euler map parameters: {}", e);
            return Err(JsValue::from_str(&e));
        }
    };

    let fixed_points_raw = find_euler_map_fixed_points();

    let mut fixed_points_result = Vec::new();
    let mut all_fixed_points_pos = Vec::new();
    let mut unstable_points_indices = Vec::new();

    for (idx, &(x, y)) in fixed_points_raw.iter().enumerate() {
        if x.abs() > 2.0 || y.abs() > 2.0 {
            continue;
        }

        let pos = Vector2::new(x, y);
        let jac = params.jacobian(pos); // DF_h(x) = I + h * Df(x)
        let trace = jac.trace();
        let det = jac.determinant();
        let discriminant = trace * trace - 4.0 * det;

        let (l1, l2) = if discriminant >= 0.0 {
            let sqrt_disc = discriminant.sqrt();
            ((trace + sqrt_disc) / 2.0, (trace - sqrt_disc) / 2.0)
        } else {
            let real = trace / 2.0;
            let imag = (-discriminant).sqrt() / 2.0;
            let mag = (real * real + imag * imag).sqrt();
            (mag, mag)
        };

        let stability = classify_euler_map_stability(l1, l2);

        console_log!(
            "Euler map fixed point ({:.4}, {:.4}): {} with eigenvalues ({:.4}, {:.4})",
            x,
            y,
            stability,
            l1,
            l2
        );

        fixed_points_result.push(EulerMapFixedPointResult {
            x,
            y,
            eigenvalues: (l1, l2),
            stability: stability.to_string(),
        });

        all_fixed_points_pos.push(pos);

        if stability == "Saddle" || stability == "Repeller" {
            unstable_points_indices.push(fixed_points_result.len() - 1);
        }
    }

    console_log!(
        "Euler map: Will compute manifolds for {} unstable/saddle points",
        unstable_points_indices.len()
    );

    let mut manifolds_result = Vec::new();
    let config = ManifoldConfig::default();
    let computer = UnstableManifoldComputer::new(params.clone(), config);

    for idx in unstable_points_indices {
        let fp_info = &fixed_points_result[idx];
        let pos = Vector2::new(fp_info.x, fp_info.y);

        let saddle_type = if fp_info.stability == "Repeller" {
            SaddleType::DualRepeller
        } else {
            SaddleType::Regular
        };

        let l1 = fp_info.eigenvalues.0;
        let l2 = fp_info.eigenvalues.1;

        let unstable_lambda = if l1.abs() > l2.abs() { l1 } else { l2 };
        let jac = params.jacobian(pos);

        // find eigenvector for unstable_lambda
        // (J - \lambda I) v = 0 -> (J_11 - \lambda) v_1 + J_12 v_2 = 0
        let v1 = 1.0;
        let mut v2 = if jac.m12.abs() > 1e-10 {
            -(jac.m11 - unstable_lambda) / jac.m12
        } else {
            jac.m21 / (unstable_lambda - jac.m22)
        };
        let norm = (v1 * v1 + v2 * v2).sqrt();
        let eigenvector = if norm > 1e-10 {
            Vector2::new(v1 / norm, v2 / norm)
        } else {
            Vector2::new(1.0, 0.0)
        };

        let saddle_pt = SaddlePoint::from_2d_eigenvector(
            pos,
            eigenvector,
            1,
            unstable_lambda,
            saddle_type,
            None,
        );

        if let Ok((traj_plus, traj_minus)) =
            computer.compute_manifold(&saddle_pt, &all_fixed_points_pos)
        {
            console_log!(
                "Euler manifold from ({:.4}, {:.4}): plus {} pts, minus {} pts",
                pos.x,
                pos.y,
                traj_plus.points.len(),
                traj_minus.points.len()
            );

            let plus_points: Vec<(f64, f64)> = traj_plus
                .points
                .iter()
                .filter(|s| s.pos.x.is_finite() && s.pos.y.is_finite())
                .map(|s| (s.pos.x, s.pos.y))
                .collect();

            let minus_points: Vec<(f64, f64)> = traj_minus
                .points
                .iter()
                .filter(|s| s.pos.x.is_finite() && s.pos.y.is_finite())
                .map(|s| (s.pos.x, s.pos.y))
                .collect();

            manifolds_result.push(EulerMapManifoldResult {
                plus: EulerMapTrajectoryRet {
                    points: plus_points,
                    stop_reason: format!("{:?}", traj_plus.stop_reason),
                },
                minus: EulerMapTrajectoryRet {
                    points: minus_points,
                    stop_reason: format!("{:?}", traj_minus.stop_reason),
                },
                saddle_point: (pos.x, pos.y),
                eigenvalue: saddle_pt.eigenvalue,
            });
        }
    }

    let result = EulerMapComputeResult {
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
pub struct EulerMapSystemWasm {
    delta: f64,
    h: f64,
    epsilon: f64,
    max_period: usize,
}

#[wasm_bindgen]
impl EulerMapSystemWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(delta: f64, h: f64, epsilon: f64, max_period: usize) -> Result<EulerMapSystemWasm, JsValue> {
        Ok(EulerMapSystemWasm {
            delta,
            h,
            epsilon,
            max_period,
        })
    }

    #[wasm_bindgen(js_name = getPeriodicOrbits)]
    pub fn get_periodic_orbits(&self) -> Result<JsValue, JsValue> {
        let orbits: Vec<()> = Vec::new();
        serde_wasm_bindgen::to_value(&orbits)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    #[wasm_bindgen(js_name = trackTrajectory)]
    pub fn track_trajectory(&mut self, _initial_x: f64, _initial_y: f64, _max_iterations: usize) {
    }

    #[wasm_bindgen(js_name = getCurrentPoint)]
    pub fn get_current_point(&self) -> Result<JsValue, JsValue> {
        Ok(JsValue::NULL)
    }

    #[wasm_bindgen(js_name = getTrajectory)]
    pub fn get_trajectory(&self, _start: usize, _end: usize) -> Result<JsValue, JsValue> {
        let points: Vec<()> = Vec::new();
        serde_wasm_bindgen::to_value(&points)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    #[wasm_bindgen]
    pub fn step(&mut self) -> bool {
        false
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {}

    #[wasm_bindgen(js_name = getTotalIterations)]
    pub fn get_total_iterations(&self) -> usize {
        0
    }

    #[wasm_bindgen(js_name = getCurrentIteration)]
    pub fn get_current_iteration(&self) -> usize {
        0
    }

    #[wasm_bindgen(js_name = getOrbitCount)]
    pub fn get_orbit_count(&self) -> usize {
        0
    }
}

use crate::boundary_periodic::ExtendedPoint;
use core::f64;

#[wasm_bindgen]
pub struct BdeSimulatorWasm {
    ode: DuffingODE,
    epsilon: f64,
    points: Vec<ExtendedPoint>,
}

#[wasm_bindgen]
impl BdeSimulatorWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(delta: f64, epsilon: f64, cx: f64, cy: f64, r: f64, num_points: usize) -> Result<BdeSimulatorWasm, JsValue> {
        let ode = DuffingODE::new(delta).map_err(|e| JsValue::from_str(&e))?;
        
        let mut points = Vec::with_capacity(num_points);
        for i in 0..num_points {
            let theta = 2.0 * f64::consts::PI * (i as f64) / (num_points as f64);
            let nx = theta.cos();
            let ny = theta.sin();
            let x = cx + r * nx;
            let y = cy + r * ny;
            points.push(ExtendedPoint::new(x, y, nx, ny));
        }
        
        Ok(BdeSimulatorWasm { ode, epsilon, points })
    }
    
    pub fn step(&mut self, h: f64) -> JsValue {
        let mut next_points = Vec::with_capacity(self.points.len());
        for p in &self.points {
            let x_vec = Vector2::new(p.x, p.y);
            let n_vec = Vector2::new(p.nx, p.ny);
            match rk4_bde_step(&self.ode, self.epsilon, h, x_vec, n_vec) {
                Ok((xp, np)) => {
                    next_points.push(ExtendedPoint::new(xp.x, xp.y, np.x, np.y));
                }
                Err(_) => {
                    next_points.push(*p);
                }
            }
        }
        self.points = next_points;
        serde_wasm_bindgen::to_value(&self.points).unwrap()
    }
    
    pub fn get_points(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.points).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Vector2, Matrix2};

    #[test]
    fn test_duffing_ode_eval() {
        let delta = 0.5;
        let ode = DuffingODE::new(delta).unwrap();
        
        let pos = Vector2::new(1.0, 2.0);
        let val = ode.vector_field(pos).unwrap();
        // x_dot = 2.0
        // y_dot = 1.0 - 1.0^3 - 0.5*2.0 = 1.0 - 1.0 - 1.0 = -1.0
        assert!((val.x - 2.0).abs() < 1e-10);
        assert!((val.y - -1.0).abs() < 1e-10);
    }

    #[test]
    fn test_euler_step() {
        let delta = 0.5;
        let ode = DuffingODE::new(delta).unwrap();
        let h = 0.1;
        let eps = 0.05;
        let euler = EulerMap::new(ode, h, eps).unwrap();

        let pos = Vector2::new(1.0, 2.0);
        let next = euler.map(pos).unwrap();
        
        // f = (2.0, -1.0)
        // next = pos + h * f = (1.0 + 0.1 * 2.0, 2.0 + 0.1 * -1.0) = (1.2, 1.9)
        assert!((next.x - 1.2).abs() < 1e-10);
        assert!((next.y - 1.9).abs() < 1e-10);
    }

    #[test]
    fn test_euler_jacobian() {
        let delta = 0.2;
        let ode = DuffingODE::new(delta).unwrap();
        let h = 0.1;
        let eps = 0.05;
        let euler = EulerMap::new(ode, h, eps).unwrap();

        let pos = Vector2::new(2.0, 1.0);
        let jac = euler.jacobian(pos);

        // Df = [[0, 1], [1 - 3*(2^2), -0.2]] = [[0, 1], [-11, -0.2]]
        // DF_h = I + h * Df = [[1 + 0, 0 + 0.1], [0 - 1.1, 1 - 0.02]] = [[1.0, 0.1], [-1.1, 0.98]]
        assert!((jac.m11 - 1.0).abs() < 1e-10);
        assert!((jac.m12 - 0.1).abs() < 1e-10);
        assert!((jac.m21 - -1.1).abs() < 1e-10);
        assert!((jac.m22 - 0.98).abs() < 1e-10);
    }

    #[test]
    fn test_euler_get_epsilon() {
        let delta = 0.2;
        let ode = DuffingODE::new(delta).unwrap();
        let h = 0.3;
        let eps = 0.5;
        let euler = EulerMap::new(ode, h, eps).unwrap();

        assert!((euler.get_epsilon() - 0.15).abs() < 1e-10);
    }
}

#[wasm_bindgen]
pub fn boundary_map_duffing_ode(
    x: f64,
    y: f64,
    nx: f64,
    ny: f64,
    delta: f64,
    h: f64,
    epsilon: f64,
) -> JsValue {
    let ode = match DuffingODE::new(delta) {
        Ok(ode) => ode,
        Err(_) => return JsValue::NULL,
    };
    match rk4_bde_step(&ode, epsilon, h, Vector2::new(x, y), Vector2::new(nx, ny)) {
        Ok((next_x, next_n)) => {
            let p = ExtendedPoint::new(next_x.x, next_x.y, next_n.x, next_n.y);
            serde_wasm_bindgen::to_value(&p).unwrap_or(JsValue::NULL)
        }
        Err(_) => JsValue::NULL,
    }
}