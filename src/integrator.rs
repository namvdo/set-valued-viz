use crate::lorenz::{lorenz_derivatives, LorenzParameters, State};

/// Runge-Kutta 4th Order (RK4) Integrator
///
/// Mathematical Foundation:
/// RK4 is a numerical method for solving ODEs dx/dt = f(x,t)
///
/// Given current state x_n at time t_n, compute next state x_{n+1}:
///
/// k1 = f(x_n, t_n)
/// k2 = f(x_n + dt/2 * k1, t_n + dt/2)
/// k3 = f(x_n + dt/2 * k2, t_n + dt/2)
/// k4 = f(x_n + dt * k3, t_n + dt)
///
/// x_{n+1} = x_n + dt/6 * (k1 + 2k2 + 2k3 + k4)
///
pub struct RK4Integrator {
    pub params: LorenzParameters,
    pub dt: f64,
}

impl RK4Integrator {
    pub fn new(params: LorenzParameters, dt: f64) -> Self {
        assert!(dt > 0.0, "Time step must be positive");
        assert!(dt < 0.1, "Time step too large for accurate integration");
        Self { params, dt }
    }

    /// Perform one RK4 integration step
    ///
    /// This implements the classical 4-stage Runge-Kutta method
    pub fn step(&self, state: &State) -> State {
        let dt = self.dt;
        let params = &self.params;

        // Stage 1: evaluate at current point
        let k1 = lorenz_derivatives(state, params);

        // Stage 2: evaluate at midpoint using k1
        let state2 = State {
            x: state.x + 0.5 * dt * k1.x,
            y: state.y + 0.5 * dt * k1.y,
            z: state.z + 0.5 * dt * k1.z,
        };
        let k2 = lorenz_derivatives(&state2, params);

        // Stage 3: evaluate at midpoint using k2
        let state3 = State {
            x: state.x + 0.5 * dt * k2.x,
            y: state.y + 0.5 * dt * k2.y,
            z: state.z + 0.5 * dt * k2.z,
        };
        let k3 = lorenz_derivatives(&state3, params);

        // Stage 4: evaluate at endpoint using k3
        let state4 = State {
            x: state.x + dt * k3.x,
            y: state.y + dt * k3.y,
            z: state.z + dt * k3.z,
        };
        let k4 = lorenz_derivatives(&state4, params);

        // Weighted average: (k1 + 2k2 + 2k3 + k4) / 6
        State {
            x: state.x + dt / 6.0 * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x),
            y: state.y + dt / 6.0 * (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y),
            z: state.z + dt / 6.0 * (k1.z + 2.0 * k2.z + 2.0 * k3.z + k4.z),
        }
    }

    /// Integrate trajectory from initial state for n_steps
    ///
    /// Returns vector of states sampled every sample_every steps
    /// This reduces memory usage for long trajectories
    pub fn integrate(
        &self,
        initial: State,
        n_steps: usize,
        sample_every: usize,
    ) -> Vec<State> {
        let mut trajectory = Vec::with_capacity(n_steps / sample_every + 1);
        let mut state = initial;

        // Always include initial state
        trajectory.push(state);

        for i in 1..=n_steps {
            state = self.step(&state);

            // Sample at specified intervals
            if i % sample_every == 0 {
                trajectory.push(state);
            }

            // Divergence detection: if state magnitude exceeds threshold
            if state.norm() > 1000.0 {
                break;
            }
        }

        trajectory
    }

    /// Compute trajectory with adaptive time-stepping
    ///
    /// Not implemented yet, but would use error estimation to
    /// adjust dt dynamically for better efficiency
    #[allow(dead_code)]
    fn integrate_adaptive(&self, _initial: State, _tolerance: f64) -> Vec<State> {
        todo!("Adaptive time-stepping not yet implemented")
    }
}

/// Compute Lyapunov exponent via trajectory divergence
///
/// Mathematical Background:
/// Lyapunov exponents measure exponential divergence rate:
/// d(t) ≈ d₀ * exp(λt)
/// λ = lim_{t→∞} (1/t) * ln(d(t)/d₀)
///
/// For Lorenz system with classical parameters:
/// λ₁ ≈ 0.9 (positive → chaos)
/// λ₂ ≈ 0
/// λ₃ ≈ -14.6 (negative → dissipation)
pub fn estimate_largest_lyapunov_exponent(
    integrator: &RK4Integrator,
    initial: State,
    perturbation: f64,
    n_steps: usize,
    renormalize_every: usize,
) -> f64 {
    let mut state1 = initial;
    let mut state2 = State {
        x: initial.x + perturbation,
        y: initial.y,
        z: initial.z,
    };

    let mut sum_log = 0.0;
    let mut n_renorm = 0;

    for i in 1..=n_steps {
        state1 = integrator.step(&state1);
        state2 = integrator.step(&state2);

        if i % renormalize_every == 0 {
            let d = state1.distance(&state2);
            sum_log += (d / perturbation).ln();
            n_renorm += 1;

            // Renormalize separation vector to prevent overflow
            let factor = perturbation / d;
            state2.x = state1.x + factor * (state2.x - state1.x);
            state2.y = state1.y + factor * (state2.y - state1.y);
            state2.z = state1.z + factor * (state2.z - state1.z);
        }
    }

    // Average over all renormalizations
    sum_log / (n_renorm as f64 * renormalize_every as f64 * integrator.dt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rk4_energy_conservation() {
        // For conservative systems, RK4 should preserve energy well
        // Lorenz is dissipative, but we test basic stability
        let params = LorenzParameters::default();
        let integrator = RK4Integrator::new(params, 0.01);
        let initial = State::new(1.0, 1.0, 1.0);
        
        let trajectory = integrator.integrate(initial, 1000, 1);
        assert!(trajectory.len() > 0);
        
        // Check no explosion
        for state in trajectory {
            assert!(state.norm() < 100.0);
        }
    }

    #[test]
    fn test_lyapunov_positive() {
        let params = LorenzParameters::default();
        let integrator = RK4Integrator::new(params, 0.01);
        let initial = State::new(1.0, 1.0, 1.0);
        
        let lambda = estimate_largest_lyapunov_exponent(
            &integrator,
            initial,
            1e-8,
            50000,
            10,
        );
        
        // Should be positive for chaotic Lorenz system
        assert!(lambda > 0.0);
        // Typically around 0.9 for classical parameters
        assert!(lambda < 2.0);
    }
}