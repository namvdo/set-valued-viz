use serde::{Deserialize, Serialize};

/// Parameters for the Lorenz system
/// 
/// Mathematical Background:
/// - σ (sigma): Prandtl number, ratio of momentum diffusivity to thermal diffusivity
/// - ρ (rho): Rayleigh number, ratio of buoyancy to viscous forces
/// - β (beta): geometric factor related to domain aspect ratio
///
/// Classical chaos occurs at σ=10, ρ=28, β=8/3
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LorenzParameters {
    pub sigma: f64,
    pub rho: f64,
    pub beta: f64,
}

impl Default for LorenzParameters {
    fn default() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
        }
    }
}

/// State vector [x, y, z] in phase space
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct State {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl State {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Compute Euclidean norm (magnitude) of state vector
    pub fn norm(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    /// Compute Euclidean distance between two states
    pub fn distance(&self, other: &State) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Compute the Lorenz system derivatives
///
/// The Lorenz equations:
/// dx/dt = σ(y - x)                   [rate of convection]
/// dy/dt = x(ρ - z) - y               [horizontal temperature variation]
/// dz/dt = xy - βz                    [vertical temperature variation]
///
/// Mathematical Properties:
/// 1. Volume contraction: div(F) = -σ - 1 - β < 0
///    → volumes in phase space contract exponentially
/// 2. Symmetry: (x,y,z) → (-x,-y,z) preserves equations
/// 3. Bounded: trajectories remain in ellipsoid x² + y² + (z-ρ-σ)² ≤ C
pub fn lorenz_derivatives(state: &State, params: &LorenzParameters) -> State {
    State {
        x: params.sigma * (state.y - state.x),
        y: state.x * (params.rho - state.z) - state.y,
        z: state.x * state.y - params.beta * state.z,
    }
}

/// Calculate fixed points of the Lorenz system
///
/// Fixed points satisfy dx/dt = dy/dt = dz/dt = 0
///
/// For ρ > 1, there are three fixed points:
/// 1. Origin: (0, 0, 0) - unstable saddle
/// 2. C+: (√(β(ρ-1)), √(β(ρ-1)), ρ-1) - stable for 1 < ρ < σ(σ+β+3)/(σ-β-1)
/// 3. C-: (-√(β(ρ-1)), -√(β(ρ-1)), ρ-1) - stable for 1 < ρ < σ(σ+β+3)/(σ-β-1)
pub fn fixed_points(params: &LorenzParameters) -> Vec<State> {
    if params.rho <= 1.0 {
        // Only origin is fixed point
        vec![State::new(0.0, 0.0, 0.0)]
    } else {
        // Three fixed points
        let c = (params.beta * (params.rho - 1.0)).sqrt();
        vec![
            State::new(0.0, 0.0, 0.0),
            State::new(c, c, params.rho - 1.0),
            State::new(-c, -c, params.rho - 1.0),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_point_origin() {
        let params = LorenzParameters::default();
        let origin = State::new(0.0, 0.0, 0.0);
        let derivatives = lorenz_derivatives(&origin, &params);
        
        assert!((derivatives.x).abs() < 1e-10);
        assert!((derivatives.y).abs() < 1e-10);
        assert!((derivatives.z).abs() < 1e-10);
    }

    #[test]
    fn test_fixed_points_symmetry() {
        let params = LorenzParameters::default();
        let fps = fixed_points(&params);
        
        assert_eq!(fps.len(), 3);
        
        // C+ and C- should be symmetric
        assert!((fps[1].x + fps[2].x).abs() < 1e-10);
        assert!((fps[1].y + fps[2].y).abs() < 1e-10);
        assert!((fps[1].z - fps[2].z).abs() < 1e-10);
    }

    #[test]
    fn test_volume_contraction() {
        let params = LorenzParameters::default();
        // Divergence should be negative
        let div = -params.sigma - 1.0 - params.beta;
        assert!(div < 0.0);
    }
}