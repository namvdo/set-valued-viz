use serde::{Deserialize, Serialize};
use std::f64;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::console;

fn log_message(s: &str) {
    #[cfg(target_arch = "wasm32")]
    console::log_1(&s.into());
    #[cfg(not(target_arch = "wasm32"))]
    println!("{}", s);
}

#[derive(Debug, Clone, Copy)]
#[wasm_bindgen]
pub struct Point {
    pub x: f64,
    pub y: f64,
}


// Extended point in the boundary space (x,y,n_x,n_y)
#[derive(Debug, Clone, Copy)]
#[wasm_bindgen]
pub struct ExtendedPoint {
    pub x: f64,
    pub y: f64,
    pub n_x: f64,
    pub n_y: f64,
}

impl ExtendedPoint {
    pub fn new(x: f64, y: f64, n_x: f64, n_y: f64) -> Self {
        Self { x, y, n_x, n_y }
    }

    pub fn from_angle(x: f64, y: f64, theta: f64) -> Self {
        Self {
            x: x, 
            y: y, 
            n_x: theta.cos(), 
            n_y: theta.sin() 
        }
    }


    pub fn normalize(&mut self) {
        let norm = (self.n_x * self.n_x + self.n_y * self.n_y);
        if norm > 1e-12 {
            self.n_x /= norm;
            self.n_y /= norm;
        }
    }


    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.n_x.is_finite() && self.n_y.is_finite()
    }

    pub fn is_bounded(&self, max_val: f64) -> bool {
        self.x.abs() < max_val && self.y.abs() < max_val
    }
}

#[derive(Debug, Clone, Copy)]
#[wasm_bindgen]
pub enum PeriodicType {
    Stable, 
    Unstable, 
    Saddle 
}

pub enum PointClassification {
    Regular,
    NearPeriodicOrbit {
        period: usize,
        stability: PeriodicType,
        distance: f64,
    }
}

// 2x2 Jacobian matrix for standard Henon map
#[derive(Debug, Clone, Copy)]
pub struct Jacobian {
    pub j11: f64,
    pub j12: f64,
    pub j21: f64,
    pub j22: f64,
}

impl Jacobian {
    pub fn new(j11: f64, j12: f64, j21: f64, j22: f64) -> Self {
        Self {
            j11: j11,
            j12: j12,
            j21: j21,
            j22: j22,
        }
    }
    
    pub fn identity() -> Self {
        Self {
            j11: 1.0,
            j12: 0.0,
            j21: 0.0,
            j22: 1.0
        }
    }

    pub fn multiply(&self, other: &Jacobian) -> Jacobian {
        Jacobian {
            j11: self.j11 * other.j11 + self.j12 * other.j12,
            j12: self.j11 * other.j21 + self.j12 * other.j22,
            j21: self.j21 * other.j11 + self.j22 * other.j22,
            j22: self.j21 * other.j12 + self.j22 * other.j22
        }
    }

    pub fn eigenvalues(&self) -> (f64, f64, bool) {
        let trace = self.j11 + self.j22;
        let det = self.j11 * self.j22 - (self.j12 + self.j21);
        let discrimant = trace * trace - 4.0 * det;
        
        if discrimant >= 0.0 {
            let sqrt_disc = discrimant.sqrt();
            ((trace + sqrt_disc) / 2.0, (trace - sqrt_disc) / 2.0, false)
        } else {
            let modulus = det.sqrt();
            (modulus, modulus, true)
        }
    }

}

// 4x4 matrix for extended boundary map 
#[derive(Copy, Clone)]
pub struct Jacobian4x4 {
    pub data: [[f64; 4]; 4]

}

impl Jacobian4x4 {
    pub fn identity() -> Self {
        Self {
            data: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        }
    }
    
    pub fn multiply(&self, other: &Jacobian4x4) -> Jacobian4x4 {
        let mut result = [[0.0;4];4];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        Jacobian4x4 { data: result }
    }

    pub fn subtract_identity(&self) -> Jacobian4x4 {
        let mut result = self.data.clone();
        for i in 0..4 {
            result[i][i] -= 1.0;
        }
        return Jacobian4x4 { data: result }
    }


    // Compute eigenvalues of 4x4 matrix using companion matrix approach    
    // Returns up to 4 eigenvalue magnitudes
    pub fn eigenvalue_magnitudes(&self) -> Vec<f64> {
        // Compute eigenvalues using QR algorithm approximation
        // For simplicity, we compute characteristic polynomial and find roots

        let a = &self.data;
        // Characteristic polynomial coefficient for 4x4 matrix
        // det(A - lamda * I) = lambda^4 - p1*lambda^3 + p2*lambda^2 - p3*lambda + p4 = 0
        
        // use trace and other invariants
        let trace = a[0][0] + a[1][1] + a[2][2] + a[3][3];

        let sum_2x2_minors = (a[0][0] * a[1][1] - a[0][1] * a[1][0])
            + (a[0][0] * a[2][2] - a[0][2] * a[2][0])
            + (a[0][0] * a[3][3] - a[0][3] * a[3][0])
            + (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            + (a[1][1] * a[3][3] - a[1][3] * a[3][1])
            + (a[2][2] * a[3][3] - a[2][3] * a[3][2]);

        // sum 3x3 principle minor    
        let det_3x3_012 = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

        let det_3x3_013 = a[0][0] * (a[1][1] * a[3][3] - a[1][3] * a[3][1])
            - a[0][1] * (a[1][0] * a[3][3] - a[1][3] * a[3][0])
            + a[0][3] * (a[1][0] * a[3][1] - a[1][1] * a[3][0]);

        let det_3x3_023 = a[0][0] * (a[2][2] * a[3][3] - a[2][3] * a[3][2])
            - a[0][2] * (a[2][0] * a[3][3] - a[2][3] * a[3][0])
            + a[0][3] * (a[2][0] * a[3][2] - a[2][2] * a[3][0]);

        let det_3x3_123 = a[1][1] * (a[2][2] * a[3][3] - a[2][3] * a[3][2])
            - a[1][2] * (a[2][1] * a[3][3] - a[2][3] * a[3][1])
            + a[1][3] * (a[2][1] * a[3][2] - a[2][2] * a[3][1]);
        let sum_3x3_minors = det_3x3_012 + det_3x3_013 + det_3x3_023 + det_3x3_123;

        // 4x4 determinant 
        let determinant = self.determinant();

        // characteristic polynomial: lambda^4 - p1*lambda^3 + p2*lambda^2 - p3*lambda + p4 = 0

        let p1 = trace;
        let p2 = sum_2x2_minors;
        let p3 = sum_3x3_minors;
        let p4 = determinant;

        // Find root numerically using companion matrix eigenvalues
        self.find_polynomial_root_quartic(p1, p2, p3, p4)
    }

    pub fn determinant(&self) -> f64 {
        let a = &self.data;
         // Laplace expansion along first row
        let minor00 = a[1][1] * (a[2][2] * a[3][3] - a[2][3] * a[3][2])
            - a[1][2] * (a[2][1] * a[3][3] - a[2][3] * a[3][1])
            + a[1][3] * (a[2][1] * a[3][2] - a[2][2] * a[3][1]);

        let minor01 = a[1][0] * (a[2][2] * a[3][3] - a[2][3] * a[3][2])
            - a[1][2] * (a[2][0] * a[3][3] - a[2][3] * a[3][0])
            + a[1][3] * (a[2][0] * a[3][2] - a[2][2] * a[3][0]);

        let minor02 = a[1][0] * (a[2][1] * a[3][3] - a[2][3] * a[3][1])
            - a[1][1] * (a[2][0] * a[3][3] - a[2][3] * a[3][0])
            + a[1][3] * (a[2][0] * a[3][1] - a[2][1] * a[3][0]);

        let minor03 = a[1][0] * (a[2][1] * a[3][2] - a[2][2] * a[3][1])
            - a[1][1] * (a[2][0] * a[3][2] - a[2][2] * a[3][0])
            + a[1][2] * (a[2][0] * a[3][1] - a[2][1] * a[3][0]);

        a[0][0] * minor00 - a[0][1] * minor01 + a[0][2] * minor02 - a[0][3] * minor03
    }

    pub fn find_polynomial_root_quartic(&self, p1: f64, p2: f64, p3: f64, p4: f64) -> Vec<f64> {
        // Finding roots of x^4 - p1*x^3 + p2*x^2 - p3*x + p4 = 0
        // Using Newton's method with multiple starting points 

        let f = |x: f64| x.powi(4) - p1 * x.powi(3) + p2 * x.powi(2) - p3 * x + p4;
        let df = |x: f64| 4.0 * x.powi(3) - 3.0 * p1 * x.powi(2) + 2.0 * p2 * x - p3;

        let mut roots = Vec::new();
        let starts = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];

        for start in starts {
            let mut x = start;
            for _ in 0..50 {
                let fx = f(x);
                let dfx = df(x);
                if dfx.abs() < 1e-12 {
                    break;
                }
                let x_new = x - fx / dfx;
                if (x_new - x).abs() < 1e-10 {
                    x = x_new;
                    break;
                }
                x = x_new;
            }

            if f(x).abs() < 1e-6 {
                // check if this is a new root
                let is_new = roots.iter().all(|&r: &f64| (r-x).abs() > 0.01);
                if is_new {
                    roots.push(x);
                }
            }
        }
        roots.iter().map(|r| r.abs()).collect()
    }

    // invert 4x4 matrix using Gaussian elimination 
    pub fn inverse(&self) -> Option<Jacobian4x4> {
        let mut a = self.data.clone(); 
        let mut inv = [[0.0;4];4];
        for i in 0..4 {
            inv[i][i] = 1.0;
        }
        
        for col in 0..4 {
            let mut max_row = col;
            for row in (col + 1)..4 {
                if a[row][col].abs() > a[max_row][col].abs() {
                    max_row = row;
                }
            }

            a.swap(col, max_row);
            inv.swap(col, max_row);

            // check for singular matrix
            if a[col][col].abs() < 1e-12 {
                return None;
            }

            let pivot = a[col][col];
            for j in 0..4 {
                a[col][j] /= pivot;
                inv[col][j] /= pivot;
            }

            for row in 0..4 {
                if row != col {
                    let factor = a[row][col];
                    for j in 0..4 {
                        a[row][j] -= factor * a[col][j];
                        inv[row][j] -= factor * inv[col][j];
                    }
                }
            }
        }
        Some(Jacobian4x4 { data: inv })
    }
}

