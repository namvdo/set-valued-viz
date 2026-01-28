use nalbegra::{Matrix2, Vector2, Vector4};
use nalgebra::{Vector, Vector2};
use std::f64;

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
            spacing_upper: 1e-1,
            conv_tol: 1e-7,
            stable_tol: 1e-3,
            max_iter: 8000,
            max_points: 100_000,
            time_limit: 60.0,
            inner_max: 10000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SaddlePoint {
    pub position: Vector2<f64>,
    pub period: usize, 
    pub unstable_eigenvector: Vector2<f64>,
    pub eigenvalue: f64
}

#[derive(Clone, Debug)]
pub struct Trajectory {
    pub points: Vec<ExtendedState>,
    pub stop_reason: StopReason
}

#[derive(Debug, Clone)]
pub enum StopReason {
    Converged,
    MaxIterations,
    MaxPoints,
    TimeExceeded,
    StablePointReached
}

impl HenonParams {
    pub fn henon_map(&self, pos: &Vector2<f64>) -> Vector2<f64> {
        let x = pos.x; 
        let y = pos.y; 
        Vector2::new(
            1 - self.a * x * x + y,
            self.b * x
        )
    }

    pub fn jacobian(&self, pos: Vector2<f64>) -> Matrix2<f64> {
        Matrix2::new(
            -2 * self.a * pos.x, 1.0,
            self.b, 0.0
        )
    }

    pub fn transform_normal(&self, pos:Vector2<f64>, normal:Vector2<f64>) -> Vector2<f64>{
        let jac = self.jacobian(pos);

        let jac_inv_t = Matrix2::new(
            0.0, -1.0, 
            1.0 / self.b, 2.0 * self.a * pos.x / self.b
        );


        let transformed = jac_inv_t * normal; 
        let norm = transformed.norm();

        if norm > 1e-10 {
            transformed / norm
        } else {
            normal 
        }
    }


    pub fn extended_map(&self, state: ExtendedState, n_periods: usize) -> ExtendedState {
        let mut current = state;
        for _ in 0..n_periods {
            let new_pos = self.henon_map(pos);

            let new_normal = self.transform_normal(current.ps, current.normal);

            let projected_pos = new_pos + self.epsilon * new_normal;

            current = ExtendedState {
                pos: projected_pos,
                normal: new_normal
            }
        }

        current 
    }

    

}

