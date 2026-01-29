use nalgebra::{Vector2, Matrix2};
use std::time::Instant;

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
            spacing_tol: 2e-4,      
            spacing_upper: 10.0,     
            conv_tol: 1e-19,        
            stable_tol: 1e-14,       
            max_iter: 8000,
            max_points: 700_000,     
            time_limit: 10.0,        
            inner_max: 2000,         
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
    pub fn henon_map(&self, pos: &Vector2<f64>) -> Vector2<f64> {
        Vector2::new(
            1.0 - self.a * pos.x * pos.x + pos.y,
            self.b * pos.x
        )
    }

    pub fn henon_map_inverse(&self, pos: &Vector2<f64>) -> Vector2<f64> {
        let y_over_b = pos.y / self.b;
        Vector2::new(
            y_over_b,
            pos.x + self.a * y_over_b * y_over_b - 1.0
        )
    }

    pub fn jacobian(&self, pos: Vector2<f64>) -> Matrix2<f64> {
        Matrix2::new(
            -2.0 * self.a * pos.x, 1.0,
            self.b, 0.0
        )
    }

    pub fn transform_normal(&self, pos: Vector2<f64>, normal: Vector2<f64>) -> Vector2<f64> {
        let jac_inv_t = Matrix2::new(
            0.0, 1.0,
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

    pub fn transform_normal_inverse(&self, pos: Vector2<f64>, normal: Vector2<f64>) -> Vector2<f64> {
        let jac = self.jacobian(pos);
        let jac_t = jac.transpose();

        let transformed = jac_t * normal;
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
            let new_pos = self.henon_map(&current.pos);
            let new_normal = self.transform_normal(current.pos, current.normal);
            let projected_pos = new_pos + self.epsilon * new_normal;

            current = ExtendedState {
                pos: projected_pos,
                normal: new_normal
            };
        }
        current
    }

    pub fn extended_map_inverse(&self, state: ExtendedState, n_periods: usize) -> ExtendedState {
        let mut current = state;
        for _ in 0..n_periods {
            let unprojected_pos = current.pos - self.epsilon * current.normal;
            
            let new_pos = self.henon_map_inverse(&unprojected_pos);
            
            let new_normal = self.transform_normal_inverse(unprojected_pos, current.normal);

            current = ExtendedState {
                pos: new_pos,
                normal: new_normal
            };
        }
        current
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

    /// Compute manifold in one direction (+v or -v)
    /// 
    /// # Arguments
    /// * `saddle` - The saddle point to start from
    /// * `direction_sign` - +1.0 for +eigenvector, -1.0 for -eigenvector
    /// * `target_points` - Points to check proximity to (other saddles or pure attractors/repellers)
    pub fn compute_direction(
        &self,
        saddle: &SaddlePoint,
        direction_sign: f64,
        target_points: &[Vector2<f64>]
    ) -> Trajectory {
        let n_period = saddle.period;
        
        let spacing_tol = if saddle.saddle_type == SaddleType::DualRepeller {
            5e-3 
        } else {
            self.config.spacing_tol  
        };

        let vec_0 = saddle.position + direction_sign * self.config.perturb_tol * saddle.eigenvector;

        let initial_normal = Vector2::new(
            -saddle.eigenvector.y,
            saddle.eigenvector.x
        ).normalize();

        let state_0 = ExtendedState {
            pos: vec_0,
            normal: initial_normal
        };

        let map_fn: Box<dyn Fn(ExtendedState, usize) -> ExtendedState> = 
            if saddle.saddle_type == SaddleType::DualRepeller {
                Box::new(|state, n| self.params.extended_map_inverse(state, n))
            } else {
                Box::new(|state, n| self.params.extended_map(state, n))
            };

        let state_1 = map_fn(state_0, n_period);
        let dist_vec_0 = state_1.pos - state_0.pos;

        let mut traj_add = vec![state_0, state_1];
        let mut vec_iter_old = state_1;

        let start_time = Instant::now();
        let mut j = 1;  // Iteration counter

        loop {
            let vec_iter = map_fn(vec_iter_old, n_period);

            let dist_diff = (vec_iter.pos - vec_iter_old.pos).norm();

            let dist_stable = target_points.iter()
                .map(|tp| (vec_iter.pos - tp).norm())
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(f64::INFINITY);

            if j > self.config.max_iter {
                return Trajectory {
                    points: traj_add,
                    stop_reason: StopReason::MaxIterations
                };
            }

            if traj_add.len() > self.config.max_points {
                return Trajectory {
                    points: traj_add,
                    stop_reason: StopReason::MaxPoints
                };
            }

            if dist_diff < self.config.conv_tol && j >= 30 && !dist_diff.is_nan() {
                return Trajectory {
                    points: traj_add,
                    stop_reason: StopReason::Converged
                };
            }

            if dist_stable <= self.config.stable_tol {
                return Trajectory {
                    points: traj_add,
                    stop_reason: StopReason::ApproachedTargetPoint
                };
            }

            if dist_diff > spacing_tol || dist_diff.is_nan() {
                match self.refine_segment(
                    vec_0,
                    dist_vec_0,
                    vec_iter_old,
                    vec_iter,
                    j,
                    n_period,
                    &map_fn,
                    spacing_tol,
                    start_time
                ) {
                    Ok(refined_points) => {
                        traj_add.extend(refined_points);
                    }
                    Err(reason) => {
                        return Trajectory {
                            points: traj_add,
                            stop_reason: reason
                        };
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
        dist_vec_0: Vector2<f64>,
        state_old: ExtendedState,
        state_new: ExtendedState,
        iter_count: usize,
        n_period: usize,
        map_fn: &dyn Fn(ExtendedState, usize) -> ExtendedState,
        spacing_tol: f64,
        start_time: Instant
    ) -> Result<Vec<ExtendedState>, StopReason> {
        let mut params: Vec<f64> = vec![0.0, 1.0];
        let mut points = vec![state_old, state_new];

        let mut k = 0;

        loop {
            let gaps: Vec<usize> = (0..points.len() - 1)
                .filter(|&i| {
                    let dist = (points[i + 1].pos - points[i].pos).norm();
                    dist > spacing_tol && dist < self.config.spacing_upper
                })
                .collect();

            if gaps.is_empty() {
                break;  
            }

            // Add midpoints (iterate in reverse to preserve indices)
            for &gap_idx in gaps.iter().rev() {
                let t_mid = (params[gap_idx] + params[gap_idx + 1]) / 2.0;

                let intermediate_pos = vec_0 + t_mid * dist_vec_0;
                let intermediate_normal = Vector2::new(-dist_vec_0.y, dist_vec_0.x).normalize();

                let intermediate_state = ExtendedState {
                    pos: intermediate_pos,
                    normal: intermediate_normal
                };

                let mapped_state = map_fn(intermediate_state, n_period * iter_count);

                points.insert(gap_idx + 1, mapped_state);
                params.insert(gap_idx + 1, t_mid);
            }

            k += 1;

            if k > self.config.inner_max {
                eprintln!("Inner infinite loop at k={}", k);
                return Err(StopReason::MaxIterations);
            }

            if start_time.elapsed().as_secs_f64() > self.config.time_limit {
                eprintln!("Exceed time at inner loop");
                return Err(StopReason::TimeExceeded);
            }
        }

        Ok(points.into_iter().skip(1).collect())
    }

    /// Compute complete manifold (both +v and -v directions)
    pub fn compute_manifold(
        &self,
        saddle: &SaddlePoint,
        target_points: &[Vector2<f64>]
    ) -> (Trajectory, Trajectory) {
        let traj_plus = self.compute_direction(saddle, 1.0, target_points);
        let traj_minus = self.compute_direction(saddle, -1.0, target_points);

        (traj_plus, traj_minus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_repeller_saddle() {
        let params = HenonParams {
            a: 1.4,
            b: 0.3,
            epsilon: 0.01
        };

        let mut config = ManifoldConfig::default();
        config.max_iter = 100;  

        let saddle = SaddlePoint {
            position: Vector2::new(-0.5, 0.5),
            period: 1,
            eigenvector: Vector2::new(0.6, 0.8).normalize(),  // stable eigenvector
            eigenvalue: 0.4,  // stable eigenvalue < 1
            saddle_type: SaddleType::DualRepeller
        };

        let target_points = vec![];

        let computer = UnstableManifoldComputer::new(params, config);
        let (traj_plus, traj_minus) = computer.compute_manifold(&saddle, &target_points);

        println!("Dual Repeller Saddle:");
        println!("  Trajectory (+v): {} points, reason: {:?}",
                 traj_plus.points.len(), traj_plus.stop_reason);
        println!("  Trajectory (-v): {} points, reason: {:?}",
                 traj_minus.points.len(), traj_minus.stop_reason);
    }
}