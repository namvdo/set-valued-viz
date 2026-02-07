use serde::{Deserialize, Serialize};
use core::f64;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::console;

use crate::StabilityType;

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
    pub nx: f64,
    pub ny: f64,
}

impl ExtendedPoint {
    pub fn new(x: f64, y: f64, n_x: f64, n_y: f64) -> Self {
        Self { x, y, nx: n_x, ny: n_y }
    }

    pub fn from_angle(x: f64, y: f64, theta: f64) -> Self {
        Self {
            x: x, 
            y: y, 
            nx: theta.cos(), 
            ny: theta.sin() 
        }
    }


    pub fn normalize(&mut self) {
        let norm = (self.nx * self.nx + self.ny * self.ny);
        if norm > 1e-12 {
            self.nx /= norm;
            self.ny /= norm;
        }
    }


    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.nx.is_finite() && self.ny.is_finite()
    }

    pub fn is_bounded(&self, max_val: f64) -> bool {
        self.x.abs() < max_val && self.y.abs() < max_val
    }
}


#[derive(Debug, Clone)]
pub struct PeriodicOrbit {
    pub points: Vec<Point>,
    pub extended_points: Vec<ExtendedPoint>,
    pub period: usize,
    pub stability: StabilityType,
    pub eigenvalues: Vec<f64>
}

#[derive(Clone, Debug)]
pub struct PeriodicOrbitDatabase {
    pub orbits: Vec<PeriodicOrbit>,
}

impl PeriodicOrbitDatabase {
    pub fn new() -> Self {
        Self { orbits: Vec::new() }
    }

    pub fn add_orbit(&mut self, orbit: PeriodicOrbit) {
        self.orbits.push(orbit);
    }

    pub fn contains_point(&self, x: f64, y: f64, tol: f64) -> bool {
        self.orbits.iter().any(|orbit| {
            orbit 
                .points
                .iter() 
                .any(|p| (p.x - x).abs() < tol && (p.y - y).abs() < tol)
        })
    }

    pub fn contains_extended_point(&self, p: &ExtendedPoint, tol: f64) -> bool {
        self.orbits.iter().any(|orbit| {
            orbit.extended_points.iter().any(|ep| {
                let dist = ((ep.x - p.x).powi(2) 
                    + (ep.y - p.y).powi(2) 
                    + (ep.nx - p.nx).powi(2) 
                    + (ep.ny - p.ny).powi(2)).sqrt();
                dist < tol
            })
        })
    }

    fn find_matching_orbit(&self, x: f64, y: f64, tol: f64) -> Option<(usize, StabilityType, f64)> {
        for orbit in &self.orbits {
            for point in &orbit.points {
                let dist = ((point.x - x).powi(2) + (point.y - y).powi(2)).sqrt();
                if dist < tol {
                    return Some((orbit.period, orbit.stability.clone(), dist));
                }
            }
        }
        None
    }

    pub fn classify_point(&self, x: f64, y: f64, tol: f64) -> PointClassification {
        if let Some((period, stability, distance)) = self.find_matching_orbit(x, y, tol) {
            PointClassification::NearPeriodicOrbit { 
                period: period,
                stability: stability, 
                distance: distance 
            }

        } else {
            PointClassification::Regular
        }
    }

    pub fn total_count(&self) -> usize {
        self.orbits.len()
    }

    pub fn get_points_of_period(&self, period: usize) -> Vec<Point> {
        self.orbits.iter() 
            .filter(|orbit| orbit.period == period)
            .flat_map(|o| o.points.clone())
            .collect()
    }

    pub fn get_extended_points_of_period(&self, period: usize) -> Vec<ExtendedPoint> {
        self.orbits.iter()
            .filter(|orbit| orbit.period == period)
            .flat_map(|o| o.extended_points.clone())
            .collect()
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
        stability: StabilityType,
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



pub struct TrajectoryPoint {
    pub x: f64,
    pub y: f64, 
    pub classification: PointClassification
}

pub fn henon_map(x: f64, y: f64, a: f64, b: f64) -> (f64, f64) {
    (1.0 - a * x * x + y, b * x)
}

fn henon_jacobian(x: f64, _y: f64, a: f64, b: f64) -> Jacobian {
    Jacobian {
        j11: -2.0 * a * x,
        j12: 1.0,
        j21: b,
        j22: 0.0,
    }
}

/// Boundary map for Henon map 
/// E(x, y, n_x, n_y) = (f_x + ep * n_x', f_y + ep * n_y', n_x', n_y')
/// where (n_x, n_y) is the outward normal vector to the boundary 
/// and (n_x', n_y') is the outward normal vector to the boundary after one iteration 
/// ep is a small parameter that controls the thickness of the boundary 
/// 
/// Normal evolution: (ñ1, ñ2) = (J^-1)^T * (n1, n2)
/// For Henon map J^-1 = [[0, 1/b], [1, 2ax/b]]
/// (J^-1)^T = [[0, 1], [1/b, 2ax/b]]
/// => ñ_x = n_y, ñy = n_x/b + 2ax*n_y/b
pub fn boundary_map(x: f64, y: f64, nx: f64, ny: f64, a: f64, b: f64, ep: f64) -> ExtendedPoint {
    let n_tilda_x = ny; 
    let n_tilda_y = nx / b + 2.0 * a * x * ny / b;
    
    let norm = (n_tilda_x * n_tilda_x + n_tilda_y * n_tilda_y).sqrt();
    if norm < 1e-12 {
        return ExtendedPoint {
            x: f64::NAN,
            y: f64::NAN,
            nx: f64::NAN,
            ny: f64::NAN,
        };
    }

    let nx_prime = n_tilda_x / norm;
    let ny_prime = n_tilda_y / norm;

    let f_x = 1.0 - a * x * x + y;
    let f_y = b * x;

    ExtendedPoint {
        x: f_x + ep * nx_prime,
        y: f_y + ep * ny_prime,
        nx: nx_prime,
        ny: ny_prime,
    }
}


// Jacobian of the boundary map E(x, y, n_x, n_y) = (f_x + ep * n_x', f_y + ep * n_y', n_x', n_y')
pub fn boundary_map_jacobian(x: f64, y: f64, nx: f64, ny: f64, a: f64, b: f64, epsilon: f64) -> Jacobian4x4 {
    // compute intermediate quantities
    let n_tilda_x = ny;
    let n_tilda_y = nx / b + 2.0 * a * x * ny / b;
    let norm_sq = n_tilda_x * n_tilda_x + n_tilda_y * n_tilda_y;
    let norm = norm_sq.sqrt();

    if norm < 1e-12 {
        return Jacobian4x4::identity();
    } 

    let nx_prime = n_tilda_x / norm;
    let ny_prime = n_tilda_y / norm;

    //  Derivatives of ñx = ny/b
    let dn_tilda_x_dx = 0.0;
    let dn_tilda_x_dy = 0.0;
    let dn_tilda_x_dn_x = 0.0;
    let dn_tilda_x_dn_y = 1.0;

    // Derivatives of ñy = nx/b + 2ax*ny/b
    let dn_tilda_y_dx = 2.0 * a * ny / b;
    let dn_tilda_y_dy = 0.0;
    let dn_tilda_y_dn_x = 1.0 / b;
    let dn_tilda_y_dn_y = 2.0 * a * x / b;
     
    // derivatives of norm = sqrt((ñx)² + (ñy)²)
    let dnorm_dx = (n_tilda_x * dn_tilda_x_dx + n_tilda_y * dn_tilda_y_dx) / norm;
    let dnorm_dy = (n_tilda_x * dn_tilda_x_dy + n_tilda_y * dn_tilda_y_dy) / norm;
    let dnorm_dn_x = (n_tilda_x * dn_tilda_x_dn_x + n_tilda_y * dn_tilda_y_dn_x) / norm;
    let dnorm_dn_y = (n_tilda_x * dn_tilda_x_dn_y + n_tilda_y * dn_tilda_y_dn_y) / norm;

    // derivative of nx' = ñx/norm using quotient rule
    let dn_x_prime_dx = (dn_tilda_x_dx * norm - n_tilda_x * dnorm_dx) / norm_sq; 
    let dn_x_prime_dy = (dn_tilda_x_dy * norm - n_tilda_x * dnorm_dy) / norm_sq;
    let dn_x_prime_dn_x = (dn_tilda_x_dn_x * norm - n_tilda_x * dnorm_dn_x) / norm_sq;
    let dn_x_prime_dn_y = (dn_tilda_x_dn_y * norm - n_tilda_x * dnorm_dn_y) / norm_sq;

    // derivative of ny' = ñy/norm using quotient rule
    let dn_y_prime_dx = (dn_tilda_y_dx * norm - n_tilda_y * dnorm_dx) / norm_sq;
    let dn_y_prime_dy = (dn_tilda_y_dy * norm - n_tilda_y * dnorm_dy) / norm_sq;
    let dn_y_prime_dn_x = (dn_tilda_y_dn_x * norm - n_tilda_y * dnorm_dn_x) / norm_sq;
    let dn_y_prime_dn_y = (dn_tilda_y_dn_y * norm - n_tilda_y * dnorm_dn_y) / norm_sq;


    // Jacobian of E 
    // E = [fx + ep*nx', fy + ep*ny', nx', ny']
    // where fx = 1 - ax^2 + y, fy = bx
    

    // dE/dx 
    let de1_dx = -2.0 * a * x + epsilon * dn_x_prime_dx;
    let de2_dx = b + epsilon * dn_y_prime_dx;
    let de3_dx = dn_x_prime_dx;
    let de4_dx = dn_y_prime_dx;
    
    // dE/dy 
    let de1_dy = 1.0 + epsilon * dn_x_prime_dy;
    let de2_dy = epsilon * dn_y_prime_dy;
    let de3_dy = dn_x_prime_dy;
    let de4_dy = dn_y_prime_dy;

    // dE/dnx 
    let de1_dnx = epsilon * dn_x_prime_dn_x;
    let de2_dnx = epsilon * dn_y_prime_dn_x; 
    let de3_dnx = dn_x_prime_dn_x;
    let de4_dnx = dn_y_prime_dn_x;
    // dE/dny 

    let de1_dny = epsilon * dn_x_prime_dn_y;
    let de2_dny = epsilon * dn_y_prime_dn_y;
    let de3_dny = dn_x_prime_dn_y;
    let de4_dny = dn_y_prime_dn_y;

    Jacobian4x4 { data: [
        [de1_dx, de1_dy, de1_dnx, de1_dny],
        [de2_dx, de2_dy, de2_dnx, de2_dny],
        [de3_dx, de3_dy, de3_dnx, de3_dny],
        [de4_dx, de4_dy, de4_dnx, de4_dny],
    ]}
}

// Compose boundary map n times and return the result with accumulated Jacobian 
pub fn compose_boundary_map_n_times(
    p: ExtendedPoint, 
    n: usize,
    a: f64, 
    b: f64, 
    epsilon: f64 
) -> (ExtendedPoint, Jacobian4x4) {
    let mut current = p; 

    if n == 0 {
        return (p, Jacobian4x4::identity());
    }

    let mut accumulated_jacobian = boundary_map_jacobian(p.x, p.y, p.nx, p.ny, a, b, epsilon);

    current = boundary_map(p.x, p.y, p.nx, p.ny, a, b, epsilon);

    for _ in 1..n {
        if !current.is_finite() || !current.is_bounded(1e10) {
            return (
                ExtendedPoint::new(f64::NAN, f64::NAN, f64::NAN, f64::NAN),
                Jacobian4x4::identity()
            );
        }

        let jac_current = boundary_map_jacobian(current.x, current.y, current.nx, current.ny, a, b, epsilon);
        accumulated_jacobian = jac_current.multiply(&accumulated_jacobian);
        current = boundary_map(current.x, current.y, current.nx, current.ny, a, b, epsilon);
    }

    (current, accumulated_jacobian)

}

// Find periodic point of the boundary map using Davidchack-Lai method
// adjust for 4d boundary map 
pub fn find_boundary_periodic_point_davidchack_lai(
    x0: f64,
    y0: f64, 
    nx_0: f64,
    ny_0: f64,
    period: usize,
    a: f64,
    b: f64,
    epsilon: f64,
    beta: Option<f64>,
    max_iter: usize,
    tol: f64,
) -> Option<ExtendedPoint> {
    let mut x = x0;
    let mut y = y0;
    let mut nx = nx_0;
    let mut ny = ny_0;

    let beta_val = beta.unwrap_or_else(|| 15.0 * 1.3_f64.powi(period as i32));

    for _ in 0..max_iter {
        if !x.is_finite() || !y.is_finite() || !nx.is_finite() || !ny.is_finite() || x.abs() > 100.0 || y.abs() > 100.0 {
            return None;
        }

        let current = ExtendedPoint::new(x, y, nx,ny);
        let (mapped, jac_fn) = compose_boundary_map_n_times(current, period, a, b, epsilon);

        if !mapped.is_finite() {
            return None;
        }

        // compute g = E^n(p) - p
        let gx = mapped.x - x;
        let gy = mapped.y - y;
        let gnx = mapped.nx - nx;
        let gny = mapped.ny - ny;

        let g_norm = (gx * gx + gy * gy + gnx * gnx + gny * gny).sqrt();

        if g_norm < 1e-10 {
            return Some(current);
        }

        // compute Jacobian of g = E^n - I
        let jac_g = jac_fn.subtract_identity();

        // Davidchank-Lai stabilization: (β||g||I - (Dg)) * Δ = g
        let scaled_beta = beta_val * g_norm;

        // modified Jacobian:  β||g||I - Jac_g

        let mut modified_jac = [[0.0;4];4];
        for i in 0..4 {
            for j in 0..4 {
                modified_jac[i][j] = -jac_g.data[i][j];
            }
            modified_jac[i][i] += scaled_beta;
        }
        let modified_jac = Jacobian4x4 { data: modified_jac };

        // Solve for Δ = (β||g||I - (Dg))^-1 * g
        let jac_inv = match modified_jac.inverse() {
            Some(inv) => inv,
            None => return None,
        };

        // delta = J^-1 * g 
        let dx = jac_inv.data[0][0] * gx
            + jac_inv.data[0][1] * gy
            + jac_inv.data[0][2] * gnx
            + jac_inv.data[0][3] * gny;
        let dy = jac_inv.data[1][0] * gx
            + jac_inv.data[1][1] * gy
            + jac_inv.data[1][2] * gnx
            + jac_inv.data[1][3] * gny;
        let dnx = jac_inv.data[2][0] * gx
            + jac_inv.data[2][1] * gy
            + jac_inv.data[2][2] * gnx
            + jac_inv.data[2][3] * gny;
        let dny = jac_inv.data[3][0] * gx
            + jac_inv.data[3][1] * gy
            + jac_inv.data[3][2] * gnx
            + jac_inv.data[3][3] * gny;

        if !dx.is_finite() || !dy.is_finite() || !dnx.is_finite() || !dny.is_finite() {
            return None;
        }

        x += dx;
        y += dy;
        nx += dnx;
        ny += dny;

        // renormalize the normal vector after each step
        let norm = (nx * nx + ny * ny).sqrt();
        nx /= norm;
        ny /= norm;

        let delta_norm = (dx * dx + dy * dy + dnx * dnx + dny * dny).sqrt();
        if delta_norm < tol {
            break;
        }
    }

    // final check if we converged 
    let final_point = ExtendedPoint::new(x, y, nx, ny);
    let (mapped, _) = compose_boundary_map_n_times(final_point, period, a, b, epsilon);

    let dist = (mapped.x - x).powi(2) 
        + (mapped.y - y).powi(2) 
        + (mapped.nx - nx).powi(2) 
        + (mapped.ny - ny).powi(2);
    
    if dist < 1e-6 {
        return Some(final_point);
    } else {
        None
    }

}

// main function for finding periodic orbits of boundary map using Davidchack-Lai
pub fn davidchack_lai_boundary_map(a: f64, b: f64, epsilon: f64, max_period: usize, grid_size: usize, theta_grid_size: usize) -> PeriodicOrbitDatabase {
    let mut database = PeriodicOrbitDatabase::new();

    // grid search in [-2.0, 2.0] x [-2.0, 2.0] x [0, 2pi]
    let x_range = (-2.0, 2.0);
    let y_range = (-2.0, 2.0);
    let theta_range = (0.0, 2.0 * PI);

    // stage 1: find orbits of all periods using 3D grid search
    for period in 1..=max_period {
        log_message(&format!("Searching for period {} orbits...", period));

        let mut found_count = 0;
        let total_points = grid_size * grid_size * theta_grid_size;
        let mut checked = 0;

        for i in 0..grid_size {
            for j in 0..grid_size {
                for k in 0..theta_grid_size {
                    checked += 1;
                    if checked % 1000 == 0 {
                        log_message(&format!(
                            "searching periodic points .. {}/{}", checked, total_points
                        ));
                    }

                    let x0 = x_range.0 + (x_range.1 - x_range.0) * (i as f64 + 0.5) / (grid_size as f64);
                    let y0 = y_range.0 + (y_range.1 - y_range.0) * (j as f64 + 0.5) / (grid_size as f64);

                    let theta = theta_range.0 + (theta_range.1 - theta_range.0) * (k as f64 + 0.5) / (theta_grid_size as f64);
                    let nx0 = theta.cos();
                    let ny0 = theta.sin();

                    if let Some(fixed_point) = find_boundary_periodic_point_davidchack_lai(
                        x0, y0, nx0, ny0, period, a, b, epsilon, None, 100, 1e-10,
                    )  {
                        if !fixed_point.is_bounded(100.0) {
                            continue;
                        }

                        if database.contains_extended_point(&fixed_point, 0.01) {
                            continue;
                        }

                        // compute orbit points 
                        let mut orbit_points = vec![Point {
                            x: fixed_point.x,
                            y: fixed_point.y,
                        }];

                        let mut extended_orbit_points = vec![fixed_point];
                        let mut current = fixed_point;

                        for _ in 1..period {
                            current = boundary_map(
                                current.x, current.y, current.nx, current.ny, a, b, epsilon,
                            );
                            orbit_points.push(Point{
                                x: current.x,
                                y: current.y,
                            });
                            extended_orbit_points.push(current);
                        }
                        
                        let (_, jac_fn) = compose_boundary_map_n_times(fixed_point, period, a, b, epsilon);
                        let (stability, eigenvalues) = classify_stability_4d(&jac_fn);

                        database.add_orbit(PeriodicOrbit {
                            points: orbit_points,
                            extended_points: extended_orbit_points,
                            period,
                            stability,
                            eigenvalues
                        });
                        found_count += 1;
                    }
                }
            }
        }
        log_message(&format!(
            "Found {} orbits of period: {}", found_count, period
        ))
    }
    database
}

/// Classify stability based on 4D Jacobian eigenvalues 
/// For boundary map, one eigenvalue is always 0 (constraint ||n|| = 1)
/// We ignore this eigenvalue and classify based on the remaining 3
pub fn classify_stability_4d(jac: &Jacobian4x4) -> (StabilityType, Vec<f64>) {
    let eigenvalues = jac.eigenvalue_magnitudes();

    // filter out non-zero eigenvalues 
    let nonzero_eigenvalues: Vec<f64> = eigenvalues.into_iter().filter(|&e| e > 1e-5).collect();
    if nonzero_eigenvalues.is_empty() {
        return (StabilityType::Stable, vec![]);
    }

    let all_stable = nonzero_eigenvalues.iter().all(|&e| e < 0.999);
    let all_unstable = nonzero_eigenvalues.iter().all(|&e| e > 1.001);

    let stability = if all_stable {
        StabilityType::Stable
    } else if all_unstable {
        StabilityType::Unstable
    } else {
        StabilityType::Saddle
    };

    (stability, nonzero_eigenvalues)
}

fn verify_minimal_period_boundary(
    point: &ExtendedPoint,
    claimed_period: usize,
    a: f64,
    b: f64, 
    epsilon: f64
) -> bool {
    for divisor in 1..claimed_period {
        if claimed_period % divisor == 0 {
            let (mapped, _) = compose_boundary_map_n_times(*point, divisor, a, b, epsilon);
            let dist = (mapped.x - point.x).powi(2) 
                + (mapped.y - point.y).powi(2) 
                + (mapped.nx - point.nx).powi(2)
                + (mapped.ny - point.ny).powi(2);
            if dist < 1e-6 {
                return false;
            }
        }
    }
    return true;
}

pub struct HenonSystemAnalysis {
    pub a: f64,
    pub b: f64, 
    pub epsilon: f64, 
    pub orbit_database: PeriodicOrbitDatabase,
    pub trajectory: Vec<TrajectoryPoint>
}

impl HenonSystemAnalysis {
    pub fn new(a: f64, b: f64, epsilon: f64, max_period: usize) -> Self {
        let grid_size = 20;
        let theta_grid_size = 20;
        
        let orbit_database = davidchack_lai_boundary_map(a, b, epsilon, max_period, grid_size, theta_grid_size);
        log_message(&format!("Total orbits found using Davidchack-Lai (boundary map): {}", orbit_database.total_count()));
        
        Self {
            a, b, epsilon, orbit_database, trajectory: Vec::new() 
        }
    }
    
    pub fn track_trajectory(&mut self, initial_x: f64, initial_y: f64, max_iterations: usize) {
        self.trajectory.clear();
        
        let mut x = initial_x;
        let mut y = initial_y;
        
        let classification = self.orbit_database.classify_point(x, y, 0.005);
        self.trajectory.push(TrajectoryPoint {
            x, y, classification
        });
        
        for iter in 1..=max_iterations {
            let (x_new, y_new) = henon_map(x, y, self.a, self.b);
            
            if !x_new.is_finite() || !y_new.is_finite() || x_new.abs() > 100.0 || y_new.abs() > 100.0 {
                log_message(&format!("Point diverged at iteration {}", iter));
                break;
            }
            let classification = self.orbit_database.classify_point(x_new, y_new, 1e-4);
            self.trajectory.push(TrajectoryPoint {
                x, y, classification
            });
            
            x = x_new;
            y = y_new;
        }
        
        log_message(&format!("Trajectory complete. Total points: {}", self.trajectory.len()));
    }
}


#[derive(Serialize, Deserialize)]
pub struct TrajectoryPointJS {
    pub x: f64,
    pub y: f64,
    pub classification: String,
    pub period: Option<usize>,
    pub stability: Option<String>
}

#[derive(Serialize, Deserialize)]
pub struct PeriodicOrbitJS {
    pub points: Vec<(f64, f64)>,
    pub extended_points: Vec<(f64, f64, f64, f64)>,
    pub period: usize,
    pub stability: String,
    pub eigenvalues: Vec<f64>
}

impl From<&StabilityType> for String {
    fn from(stability: &StabilityType) -> Self {
        match stability {
            StabilityType::Stable => "stable".to_string(),
            StabilityType::Unstable => "unstable".to_string(),
            StabilityType::Saddle => "saddle".to_string(),
        }
    }
}
impl From<&TrajectoryPoint> for TrajectoryPointJS {
    fn from(point: &TrajectoryPoint) -> Self {
        match &point.classification {
            PointClassification::Regular => TrajectoryPointJS {
                x: point.x,
                y: point.y, 
                classification: "regular".to_string(),
                period: None,
                stability: None
            } ,
            PointClassification::NearPeriodicOrbit {
                period,
                stability,
                distance: _,
            } => TrajectoryPointJS {
                x: point.x,
                y: point.y,
                classification: "periodic".to_string(),
                period: Some(*period),
                stability: Some(String::from(stability),)
            }
            
        } 
    }
}


#[wasm_bindgen]
pub struct HenonSystemWasm {
    system: HenonSystemAnalysis,
    current_iteration: usize,
}

#[wasm_bindgen]
impl HenonSystemWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(a: f64, b: f64, epsilon: f64, max_period: usize) -> Result<HenonSystemWasm, JsValue> {
        console_error_panic_hook::set_once();
        
        let system = HenonSystemAnalysis::new(a, b, epsilon, max_period);
        Ok (Self {
            system,
            current_iteration: 0
        })
    }
    
    #[wasm_bindgen(js_name = getPeriodicOrbits)]
    pub fn get_periodic_orbits(&self, x: f64, y: f64, max_iterations: usize) -> Result<JsValue, JsValue> {
        let orbits: Vec<PeriodicOrbitJS> = self
            .system
            .orbit_database
            .orbits
            .iter()
            .map(|orbit| PeriodicOrbitJS {
                points: orbit.points.iter().map(|p| (p.x, p.y)).collect(),
                extended_points: orbit
                    .extended_points
                    .iter()
                    .map(|p|(p.x, p.y, p.nx, p.ny))
                    .collect(),
                period: orbit.period,
                stability: String::from(&orbit.stability),
                eigenvalues: orbit.eigenvalues.clone(),
            }).collect();
        serde_wasm_bindgen::to_value(&orbits).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }
    #[wasm_bindgen(js_name = "trackTrajectory")]
    pub fn track_trajectory(&mut self, initial_x: f64, initial_y: f64, max_iterations: usize) {
        self.system.track_trajectory(initial_x, initial_y, max_iterations);
        self.current_iteration = 0;
    }
    
    #[wasm_bindgen(js_name = "getCurrentPoint")]
    pub fn get_current_point(&self) -> Result<JsValue, JsValue>{
        if self.current_iteration < self.system.trajectory.len() {
            let point = &self.system.trajectory[self.current_iteration];
            let point_js = TrajectoryPointJS::from(point);
            
            serde_wasm_bindgen::to_value(&point_js)
                .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
        } else {
            Ok(JsValue::NULL)
        }
    }
    
    #[wasm_bindgen(js_name = "getTrajectory")] 
    pub fn get_trajectory(&self, start: usize, end: usize) -> Result<JsValue, JsValue> {
        let end = end.min(self.system.trajectory.len());
        let points: Vec<TrajectoryPointJS> = self.system.trajectory[start..end]
            .iter()
            .map(TrajectoryPointJS::from)
            .collect();
        
        serde_wasm_bindgen::to_value(&points).map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }
    
    #[wasm_bindgen()]
    pub fn step(&mut self) -> bool {
        if self.current_iteration + 1 < self.system.trajectory.len() {
            self.current_iteration += 1;
            true
        } else {
            false
        }
    }
    
    #[wasm_bindgen]
    pub fn reset(&mut self) {self.current_iteration = 0; }
    
    #[wasm_bindgen(js_name = "getTotalIterations")]
    pub fn get_total_iterations(&self) -> usize { self.system.trajectory.len() }
    
    #[wasm_bindgen(js_name = "getCurrentIteration")]
    pub fn get_current_iteration(&self) -> usize { self.current_iteration }
    
    #[wasm_bindgen(js_name = "getOrbitCount")]
    pub fn get_orbit_count(&self) -> usize { self.system.orbit_database.total_count() }
    
    #[wasm_bindgen(js_name = "getEpsilon")]
    pub fn get_epsilon(&self) -> f64 { self.system.epsilon }
}





