use nalgebra::{Matrix2, Matrix4, Vector2, Vector4};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use wasm_bindgen::prelude::*;


#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct DuffingParams {
    pub a: f64,
    pub b: f64,
    pub epsilon: f64
}

impl DuffingParams {
    pub fn new(a: f64, b: f64, epsilon: f64) -> Result<Self, String> {
        if !a.is_finite() || !b.is_finite() || !epsilon.is_finite() {
            return Err("Parameters must be finite number".to_string());
        }

        if b.abs() < 1e-10 {
            return Err("Parameter b cannot be zero or near zero".to_string());
        }

        if epsilon <= 0.0 {
            return Err("Epsilon must be positive".to_string());
        }

        if a.abs() > 10.0 || epsilon > 1.0 || b.abs() > 10 {
            return Err("Parameters outside reasonable range".to_string());
        }

        Ok(Self { a, b, epsilon })
    }

    /// Duffing map: x_{n+1} = y_n, y_{n+1} = -b * x_n + a*y_n - y_n^3
    pub fn duffing_map(&self, pos: &Vector2<f64>) -> Result<Vector2<f64>, String> {
        if !pos.x.is_finite() || !pos.y.is_finite() {
            return Err("Invalid position: non finite coordinates".to_string());
        }

        let new_x = pos.y; 
        let new_y = -self.b * pos.x + self.a * pos.y - pos.y.powi(3);

        if !new_x.is_finite() || !new_y.is_finite() {
            return Err("Duffing map produced non finite values".to_string());
        }

        Ok(Vector2::new(new_x, new_y))
    }

    
}
