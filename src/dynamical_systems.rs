use nalgebra::{Matrix2, Vector2};

use crate::parameters::ParameterSet;
use crate::user_defined::ParsedEquations;

#[derive(Debug, Clone, Copy)]
pub struct ExtendedState {
    pub pos: Vector2<f64>,
    pub normal: Vector2<f64>,
}

pub trait DynamicalSystem: Send + Sync {
    fn map(&self, pos: Vector2<f64>) -> Result<Vector2<f64>, String>;
    fn map_inverse(&self, _pos: Vector2<f64>) -> Result<Vector2<f64>, String> {
        Err("Inverse map not implemented".to_string())
    }




    fn jacobian(&self, pos: Vector2<f64>) -> Matrix2<f64>;

    fn transform_normal(
        &self,
        pos: Vector2<f64>,
        normal: Vector2<f64>,
    ) -> Result<Vector2<f64>, String> {
        let j = self.jacobian(pos);
        // J^-T = (J^-1)^T
        // For 2x2 matrix [[a, b], [c, d]]
        // det = ad - bc
        // J^-1 = 1/det * [[d, -b], [-c, a]]
        // (J^-1)^T = 1/det * [[d, -c], [-b, a]]

        let det = j[(0, 0)] * j[(1, 1)] - j[(0, 1)] * j[(1, 0)];
        if det.abs() < 1e-12 {
            // Singular matrix, fallback to normal
            return Ok(normal);
        }

        let inv_t = Matrix2::new(j[(1, 1)], -j[(1, 0)], -j[(0, 1)], j[(0, 0)]) / det;
        let transformed = inv_t * normal;
        let norm = transformed.norm();

        if norm < 1e-12 {
            return Ok(normal);
        }

        Ok(transformed / norm)
    }

    fn transform_normal_inverse(
        &self,
        _pos: Vector2<f64>,
        normal: Vector2<f64>,
    ) -> Result<Vector2<f64>, String> {
        Ok(normal)
    }

    fn get_epsilon(&self) -> f64;

    fn extended_map(
        &self,
        state: ExtendedState,
        n_periods: usize,
    ) -> Result<ExtendedState, String> {
        let mut current = state;
        let epsilon = self.get_epsilon();

        for iter in 0..n_periods {
            let new_pos = self.map(current.pos)?;
            let new_normal = self.transform_normal(current.pos, current.normal)?;
            let projected_pos = new_pos + epsilon * new_normal;

            if !projected_pos.x.is_finite() || !projected_pos.y.is_finite() {
                return Err(format!("Non-finite position at iteration: {}", iter));
            }

            if projected_pos.x.abs() > 1000.0 || projected_pos.y.abs() > 1000.0 {
                return Err(format!("Position diverged at iteration: {}", iter));
            }

            current = ExtendedState {
                pos: projected_pos,
                normal: new_normal,
            };
        }
        Ok(current)
    }

    fn extended_map_inverse(
        &self,
        state: ExtendedState,
        n_periods: usize,
    ) -> Result<ExtendedState, String> {
        let mut current = state;
        let epsilon = self.get_epsilon();

        for iter in 0..n_periods {
            let unprojected_pos = current.pos - epsilon * current.normal;

            if !unprojected_pos.x.is_finite() || !unprojected_pos.y.is_finite() {
                return Err(format!(
                    "Non-finite unprojected position at iteration: {}",
                    iter
                ));
            }

            let new_pos = self.map_inverse(unprojected_pos)?;
            let new_normal = self.transform_normal_inverse(unprojected_pos, current.normal)?;

            if !new_pos.x.is_finite()
                || !new_pos.y.is_finite()
                || !new_normal.x.is_finite()
                || !new_normal.y.is_finite()
            {
                return Err(format!("Non-finite new state at iteration: {}", iter));
            }

            // Check for divergence?
            if new_pos.x.abs() > 1000.0 || new_pos.y.abs() > 1000.0 {
                return Err(format!("Position diverged at iteration: {}", iter));
            }

            current = ExtendedState {
                pos: new_pos,
                normal: new_normal,
            };
        }

        Ok(current)
    }
}

#[derive(Clone)]
pub struct UserDefinedDynamicalSystem {
    epsilon: f64,
    equations: ParsedEquations,
}

impl UserDefinedDynamicalSystem {
    pub fn new(
        x_str: &str,
        y_str: &str,
        epsilon: f64,
        params: ParameterSet,
    ) -> Result<Self, String> {
        let equations = ParsedEquations::new(x_str, y_str, params)?;
        Ok(Self { epsilon, equations })
    }
}

impl DynamicalSystem for UserDefinedDynamicalSystem {
    fn map(&self, pos: Vector2<f64>) -> Result<Vector2<f64>, String> {
        let (x_new, y_new) = self.equations.eval(pos.x, pos.y)?;
        Ok(Vector2::new(x_new, y_new))
    }

    fn jacobian(&self, pos: Vector2<f64>) -> Matrix2<f64> {
        let h = 1e-5;

        let fx = |x: f64, y: f64| -> f64 {
            self.equations.eval(x, y).map(|v| v.0).unwrap_or(0.0)
        };

        let fy = |x: f64, y: f64| -> f64 {
            self.equations.eval(x, y).map(|v| v.1).unwrap_or(0.0)
        };

        let dfx_dx = (fx(pos.x + h, pos.y) - fx(pos.x - h, pos.y)) / (2.0 * h);
        let dfx_dy = (fx(pos.x, pos.y + h) - fx(pos.x, pos.y - h)) / (2.0 * h);
        let dfy_dx = (fy(pos.x + h, pos.y) - fy(pos.x - h, pos.y)) / (2.0 * h);
        let dfy_dy = (fy(pos.x, pos.y + h) - fy(pos.x, pos.y - h)) / (2.0 * h);

        Matrix2::new(dfx_dx, dfx_dy, dfy_dx, dfy_dy)
    }

    fn get_epsilon(&self) -> f64 {
        self.epsilon
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameters::ParameterEntry;

    #[test]
    fn test_user_defined_system_basic() {
        // x_n+1 = y_n
        // y_n+1 = -0.1 * x_n + y_n^2
        let params = ParameterSet::new(vec![
            ParameterEntry {
                name: "a".to_string(),
                value: 1.4,
            },
            ParameterEntry {
                name: "b".to_string(),
                value: 0.3,
            },
        ])
        .unwrap();
        let system = UserDefinedDynamicalSystem::new("y", "-0.1 * x + y^2", 0.001, params)
            .unwrap();

        let pos = Vector2::new(1.0, 0.5);
        let result = system.map(pos).unwrap();

        assert!((result.x - 0.5).abs() < 1e-10);
        // y_new = -0.1 * 1.0 + 0.5^2 = -0.1 + 0.25 = 0.15
        assert!((result.y - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_user_defined_system_many_parameters() {
        let params = ParameterSet::new(vec![
            ParameterEntry {
                name: "a".to_string(),
                value: 1.5,
            },
            ParameterEntry {
                name: "b".to_string(),
                value: -0.25,
            },
            ParameterEntry {
                name: "c".to_string(),
                value: 0.75,
            },
            ParameterEntry {
                name: "d".to_string(),
                value: -2.0,
            },
            ParameterEntry {
                name: "e".to_string(),
                value: 3.0,
            },
        ])
        .unwrap();

        let system = UserDefinedDynamicalSystem::new(
            "a * x + b * y + c",
            "d * x + e * y",
            0.0,
            params,
        )
        .unwrap();

        let pos = Vector2::new(2.0, -1.0);
        let result = system.map(pos).unwrap();

        assert!((result.x - (1.5 * 2.0 + -0.25 * -1.0 + 0.75)).abs() < 1e-12);
        assert!((result.y - (-2.0 * 2.0 + 3.0 * -1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_user_defined_extended_map_uses_parameters() {
        let params = ParameterSet::new(vec![
            ParameterEntry {
                name: "a".to_string(),
                value: 2.0,
            },
            ParameterEntry {
                name: "b".to_string(),
                value: 3.0,
            },
        ])
        .unwrap();

        let epsilon = 0.1;
        let system = UserDefinedDynamicalSystem::new("a * x", "b * y", epsilon, params).unwrap();

        let state = ExtendedState {
            pos: Vector2::new(1.0, -2.0),
            normal: Vector2::new(1.0, 0.0),
        };
        let next = system.extended_map(state, 1).unwrap();

        assert!((next.pos.x - 2.1).abs() < 1e-12);
        assert!((next.pos.y - -6.0).abs() < 1e-12);
        assert!((next.normal.x - 1.0).abs() < 1e-12);
        assert!((next.normal.y - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_user_defined_jacobian() {
        // x = x^2 + y
        // y = sin(x)
        let params = ParameterSet::new(vec![
            ParameterEntry {
                name: "a".to_string(),
                value: 1.4,
            },
            ParameterEntry {
                name: "b".to_string(),
                value: 0.3,
            },
        ])
        .unwrap();
        let system = UserDefinedDynamicalSystem::new("x^2 + y", "sin(x)", 0.001, params).unwrap();
        let pos = Vector2::new(1.0, 0.5);

        let jac = system.jacobian(pos);

        // Analytical:
        // dx_new/dx = 2x = 2.0
        // dx_new/dy = 1.0
        // dy_new/dx = cos(x) = cos(1.0) approx 0.5403
        // dy_new/dy = 0.0

        assert!(
            (jac[(0, 0)] - 2.0).abs() < 1e-4,
            "Expected 2.0, got {}",
            jac[(0, 0)]
        );
        assert!(
            (jac[(0, 1)] - 1.0).abs() < 1e-4,
            "Expected 1.0, got {}",
            jac[(0, 1)]
        );
        assert!(
            (jac[(1, 0)] - 1.0f64.cos()).abs() < 1e-4,
            "Expected cos(1)={:.4}, got {:.4}",
            1.0f64.cos(),
            jac[(1, 0)]
        );
        assert!(
            (jac[(1, 1)] - 0.0).abs() < 1e-4,
            "Expected 0.0, got {}",
            jac[(1, 1)]
        );
    }

    #[test]
    fn test_lozi_map() {
        // Lozi map: x' = 1 - a*|x| + y, y' = b*x
        let params = ParameterSet::new(vec![
            ParameterEntry {
                name: "a".to_string(),
                value: 1.7,
            },
            ParameterEntry {
                name: "b".to_string(),
                value: 0.5,
            },
        ])
        .unwrap();
        let system =
            UserDefinedDynamicalSystem::new("1 - a * |x| + y", "b * x", 0.01, params).unwrap();

        let pos = Vector2::new(0.5, 0.3);
        let result = system.map(pos).unwrap();
        // x' = 1 - 1.7 * |0.5| + 0.3 = 1 - 0.85 + 0.3 = 0.45
        // y' = 0.5 * 0.5 = 0.25
        assert!((result.x - 0.45).abs() < 1e-10, "got {}", result.x);
        assert!((result.y - 0.25).abs() < 1e-10, "got {}", result.y);

        // test with negative x
        let pos2 = Vector2::new(-0.5, 0.3);
        let result2 = system.map(pos2).unwrap();
        // x' = 1 - 1.7 * |-0.5| + 0.3 = 1 - 0.85 + 0.3 = 0.45
        // y' = 0.5 * (-0.5) = -0.25
        assert!((result2.x - 0.45).abs() < 1e-10, "got {}", result2.x);
        assert!((result2.y - (-0.25)).abs() < 1e-10, "got {}", result2.y);
    }

}
