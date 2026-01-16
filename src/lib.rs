mod integrator;
mod lorenz;

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

use crate::integrator::{estimate_largest_lyapunov_exponent, RK4Integrator};
use crate::lorenz::{fixed_points, LorenzParameters, State};

/// JavaScript-friendly trajectory point
#[wasm_bindgen]
#[derive(Serialize, Deserialize)]
pub struct TrajectoryPoint {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub time: f64,
}

#[wasm_bindgen]
impl TrajectoryPoint {
    #[wasm_bindgen(constructor)]
    pub fn new(x: f64, y: f64, z: f64, time: f64) -> Self {
        Self { x, y, z, time }
    }
}

/// Main simulation struct exposed to JavaScript
#[wasm_bindgen]
pub struct LorenzSimulation {
    integrator: RK4Integrator,
    current_time: f64,
}

#[wasm_bindgen]
impl LorenzSimulation {
    /// Create new simulation with given parameters
    ///
    /// # Arguments
    /// * `sigma` - Prandtl number (typical: 10)
    /// * `rho` - Rayleigh number (typical: 28)
    /// * `beta` - Geometric factor (typical: 8/3 ≈ 2.667)
    /// * `dt` - Time step (typical: 0.01)
    #[wasm_bindgen(constructor)]
    pub fn new(sigma: f64, rho: f64, beta: f64, dt: f64) -> Self {
        let params = LorenzParameters { sigma, rho, beta };
        let integrator = RK4Integrator::new(params, dt);
        
        Self {
            integrator,
            current_time: 0.0,
        }
    }

    /// Compute trajectory from initial conditions
    ///
    /// # Arguments
    /// * `x0, y0, z0` - Initial state
    /// * `n_steps` - Number of integration steps
    /// * `sample_every` - Sample interval (1 = every step)
    ///
    /// # Returns
    /// JSON string containing array of {x, y, z, time} points
    pub fn compute_trajectory(
        &mut self,
        x0: f64,
        y0: f64,
        z0: f64,
        n_steps: usize,
        sample_every: usize,
    ) -> String {
        let initial = State::new(x0, y0, z0);
        let states = self.integrator.integrate(initial, n_steps, sample_every);

        let trajectory: Vec<TrajectoryPoint> = states
            .into_iter()
            .enumerate()
            .map(|(i, state)| TrajectoryPoint {
                x: state.x,
                y: state.y,
                z: state.z,
                time: i as f64 * sample_every as f64 * self.integrator.dt,
            })
            .collect();

        serde_json::to_string(&trajectory).unwrap_or_else(|_| "[]".to_string())
    }

    /// Compute multiple trajectories from slightly different initial conditions
    ///
    /// Demonstrates sensitive dependence on initial conditions
    pub fn compute_butterfly_effect(
        &mut self,
        x0: f64,
        y0: f64,
        z0: f64,
        perturbation: f64,
        n_trajectories: usize,
        n_steps: usize,
        sample_every: usize,
    ) -> String {
        let mut all_trajectories = Vec::new();

        for i in 0..n_trajectories {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_trajectories as f64);
            let dx = perturbation * angle.cos();
            let dy = perturbation * angle.sin();

            let initial = State::new(x0 + dx, y0 + dy, z0);
            let states = self.integrator.integrate(initial, n_steps, sample_every);

            let trajectory: Vec<TrajectoryPoint> = states
                .into_iter()
                .enumerate()
                .map(|(j, state)| TrajectoryPoint {
                    x: state.x,
                    y: state.y,
                    z: state.z,
                    time: j as f64 * sample_every as f64 * self.integrator.dt,
                })
                .collect();

            all_trajectories.push(trajectory);
        }

        serde_json::to_string(&all_trajectories).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get fixed points of the system
    ///
    /// Returns JSON string of equilibrium points
    pub fn get_fixed_points(&self) -> String {
        let fps = fixed_points(&self.integrator.params);
        
        let points: Vec<TrajectoryPoint> = fps
            .into_iter()
            .map(|state| TrajectoryPoint {
                x: state.x,
                y: state.y,
                z: state.z,
                time: 0.0,
            })
            .collect();

        serde_json::to_string(&points).unwrap_or_else(|_| "[]".to_string())
    }

    /// Estimate largest Lyapunov exponent
    ///
    /// Returns positive value for chaotic systems
    /// Classical Lorenz: λ ≈ 0.9
    pub fn compute_lyapunov_exponent(&self, x0: f64, y0: f64, z0: f64) -> f64 {
        let initial = State::new(x0, y0, z0);
        estimate_largest_lyapunov_exponent(
            &self.integrator,
            initial,
            1e-8,
            50000,
            10,
        )
    }

    /// Single integration step (for animations)
    pub fn step(&mut self, x: f64, y: f64, z: f64) -> TrajectoryPoint {
        let state = State::new(x, y, z);
        let next = self.integrator.step(&state);
        self.current_time += self.integrator.dt;

        TrajectoryPoint {
            x: next.x,
            y: next.y,
            z: next.z,
            time: self.current_time,
        }
    }

    /// Reset simulation time
    pub fn reset_time(&mut self) {
        self.current_time = 0.0;
    }

    /// Update parameters (creates new integrator)
    pub fn update_parameters(&mut self, sigma: f64, rho: f64, beta: f64) {
        let params = LorenzParameters { sigma, rho, beta };
        self.integrator.params = params;
    }
}

/// Utility function for web console logging
#[wasm_bindgen]
pub fn init_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_creation() {
        let sim = LorenzSimulation::new(10.0, 28.0, 8.0/3.0, 0.01);
        assert_eq!(sim.current_time, 0.0);
    }

    #[test]
    fn test_trajectory_computation() {
        let mut sim = LorenzSimulation::new(10.0, 28.0, 8.0/3.0, 0.01);
        let trajectory_json = sim.compute_trajectory(1.0, 1.0, 1.0, 100, 1);
        assert!(!trajectory_json.is_empty());
    }
}