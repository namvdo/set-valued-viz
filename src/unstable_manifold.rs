use nalgebra::{Vector2, Matrix2};
use core::f64;

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
        return Vector2::new(
            1.0 - self.a * x * x + y,
            self.b * x
        );
    }

    pub fn jacobian(&self, pos: Vector2<f64>) -> Matrix2<f64> {
        Matrix2::new(
            -2.0 * self.a * pos.x, 1.0,
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
            let new_pos = self.henon_map(&current.pos);

            let new_normal = self.transform_normal(current.pos, current.normal);

            let projected_pos = new_pos + self.epsilon * new_normal;

            current = ExtendedState {
                pos: projected_pos,
                normal: new_normal
            }
        }

        current 
    }



}



// compute unstable manifold from saddle periodic point 
pub struct UnstableManifoldComputer {
    params: HenonParams, 
    config: ManifoldConfig
}

impl UnstableManifoldComputer {
    pub fn new(params: HenonParams, config: ManifoldConfig) -> Self {
        Self {
            params,
            config
        }
    }

    pub fn compute_direction(
        &self, 
        saddle: &SaddlePoint,
        direction_sign: f64,
        stable_points: &[Vector2<f64>]
    ) -> Trajectory {
        let n_period = saddle.period;

        let vec_0 = saddle.position + direction_sign * saddle.unstable_eigenvector * self.config.perturb_tol;

        let initial_normal = Vector2::new(
            -saddle.unstable_eigenvector.y,
            saddle.unstable_eigenvector.x 
        ).normalize();  

        let state_0 = ExtendedState{
            pos: vec_0, 
            normal: initial_normal
        };

        let state_1 = self.params.extended_map(state_0, n_period);
        let dist_vec_0 = state_1.pos - state_0.pos;

        let mut trajectory = vec![state_0, state_1];
        let mut state_old = state_1;

        let start_time = std::time::Instant::now();
        let mut j = 1;

        loop {
            let state_new = self.params.extended_map(state_old, n_period);
            let dist_diff = (state_new.pos - state_old.pos).norm();

            if j >= self.config.max_iter {
                return Trajectory { 
                    points: trajectory, 
                    stop_reason: StopReason::MaxIterations 
                };
            }

            if dist_diff < self.config.conv_tol && j >= 30 {
                return Trajectory { 
                    points: trajectory, 
                    stop_reason: StopReason::Converged
                };
            }

            let dist_to_saddles = other_saddles.iter()
            .map(|sp| (state_new.pos - sp).norm())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(f64::INFINITY);

            
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
        start_time: std::time::Instant
    ) -> Result<Vec<ExtendedState>, StopReason>{
        let mut params: Vec<f64> = vec![0.0, 1.0];
        let mut points  = vec![state_old, state_new];

        let mut k = 0;
        loop {
            // store all the gaps with the distance between 2 consecutive points larger than the spacing tolerance threshold
            let gaps: Vec<usize> = (0..points.len() -1)
                .filter(|&i| {
                    let dist = (points[i+1].pos - points[i].pos).norm();
                    dist > self.config.spacing_tol && dist < self.config.spacing_upper
                }).collect();
            
            if gaps.is_empty() {
                // all gaps are within tolerance, break
                break;
            }


            // add midpoints
            for &gap_idx in gaps.iter().rev() {
                let t_mid = (params[gap_idx] + params[gap_idx + 1]) / 2.0;

                let intermediate_pos = vec_0 + t_mid * dist_vec_0;
                let intermediate_normal = Vector2::new(-dist_vec_0.y, dist_vec_0.x).normalize();

                let intermediate_state = ExtendedState {
                    pos: intermediate_pos, 
                    normal: intermediate_normal
                };

                let mapped_state = self.params.extended_map(intermediate_state, n_period * iter_count);
                points.insert(gap_idx + 1, mapped_state);
                params.insert(gap_idx + 1, t_mid);
            }


            k += 1;
            if k > self.config.inner_max {
                return Err(StopReason::MaxIterations);
            }

            if start_time.elapsed().as_secs_f64() > self.config.time_limit {
                return Err(StopReason::TimeExceeded);
            }
        }

        Ok(points.into_iter().skip(1).collect())
    }

    pub fn compute_manifold(
        &self, saddle: &SaddlePoint, stable_points: &[Vector2<f64>]
    ) -> (Trajectory, Trajectory) {
        let traj_plus = self.compute_direction(saddle, 1.0, stable_points);
        let traj_minus = self.compute_direction(saddle, -1.0, stable_points);

        (traj_plus, traj_minus)
    }



}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]

    fn test_unstable_manifold_computer() {
        let params = HenonParams {
            a: 1.4,
            b: 0.3,
            epsilon: 0.01
        };

        let config = ManifoldConfig {
            max_iter: 1000,
            ..Default::default()
        };

        let saddle = SaddlePoint {
            position: Vector2::new(0.631, 0.189),
            period: 1,
            unstable_eigenvector: Vector2::new(0.9, 0.1).normalize(),
            eigenvalue: 2.5
        };


        let other_saddles = vec![
            Vector2::new(-1.131, 0.339),
            Vector2::new(0.339, -1.131)
        ];

        let computer = UnstableManifoldComputer::new(params, config);
        let (traj_plus, traj_minus) = computer.compute_manifold(&saddle, &other_saddles);

        println!("Trajectory (+v): {} points, reason: {:?}", 
                 traj_plus.points.len(), traj_plus.stop_reason);
        println!("Trajectory (-v): {} points, reason: {:?}", 
                 traj_minus.points.len(), traj_minus.stop_reason);
    }
}