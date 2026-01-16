use std::f64;
use kdtree::KDTree;
#[derive(Debug, Clone)]
pub struct Point {
    pub x: f64, 
    pub y: f64
}

#[derive(Debug, Clone)]
pub struct Jacobian {
    pub j11: f64,
    pub j12: f64, 
    pub j21: f64, 
    pub j22: f64
}


pub struct FixedPoint {
    pub x: f64, 
    pub y: f64, 
    pub period: usize, 
    pub stability: StabilityType,
}

pub enum StabilityType {
    Stable,
    Unstable,
    Saddle
}



impl Jacobian {
    fn inverse(&self) -> Option<Jacobian> {
        let det = self.j11 * self.j22 - self.j12 * self.j21;

        if det.abs() < 1e-12 {
            return None
        }

        Some(Jacobian {
            j11: self.j22 / det, 
            j12: -self.j12 / det,
            j21: -self.j21 / det,
            j22: self.j11 / det
        })
    }

    fn multiply(&self, other: &Jacobian) -> Jacobian {
        Jacobian {
            j11: self.j11 * other.j11 + self.j12 * other.j21, 
            j12: self.j11 * other.j12 + self.j12 * other.j22, 
            j21: self.j21 * other.j11 + self.j22 * other.j21, 
            j22: self.j21 * other.j12 + self.j22 * other.j22
        }
    }
}

fn henon_map(x: f64, y: f64, a: f64, b: f64) -> (f64, f64) {
    (
        1.0 - a * x * x + y,
        b * x
    )
}

fn henon_jacobian(x: f64, _y: f64, a: f64, b: f64) -> Jacobian {
    Jacobian {
        j11: -2.0 * a * x,
        j12: 1.0,
        j21: b,
        j22: 0.0
    }
}

fn compose_henon_n_times(x0: f64, y0: f64, a: f64, b: f64, n: usize) -> (f64, f64, Jacobian) {
    let mut x = x0;
    let mut y = y0; 

    let mut total_jac = Jacobian {
        j11: 1.0, 
        j12: 0.0,
        j21: 0.0,
        j22: 1.0
    };

    for _ in 0..n {
        let local_jac = henon_jacobian(x, y, a, b);
        (x, y) = henon_map(x, y, a, b);
        total_jac = local_jac.multiply(&total_jac);
    }

    (x, y, total_jac)
}

fn proper_divisors(n: usize) -> Vec<usize> {
    if n <= 1 {
        return vec![];
    }
    (1..n).filter(|&d| n % d == 0).collect()
}

fn verify_minimal_period(x: f64, y: f64, a: f64, b: f64, period: usize) -> bool {
    let mut xn = x;
    let mut yn = y;
    for _ in 0..period {
        (xn, yn) = henon_map(xn, yn, a, b);
    }
    let dist = ((xn - x).powi(2) + (yn - y).powi(2)).sqrt();
    if dist >= 1e-8 {
        return false;
    }
    
    for divisor in proper_divisors(period) {
        let mut xt = x;
        let mut yt = y;
        for _ in 0..divisor {
            (xt, yt) = henon_map(xt, yt, a, b);
        }
        let dist = ((xt - x).powi(2) + (yt - y).powi(2)).sqrt();
        if dist < 1e-8 {
            return false;
        }
    }
    
    true
}

pub fn find_periodic_point_near(x_guess: f64, y_guess: f64, a: f64, b: f64, period: usize) -> Option<Point> {
    let mut x = x_guess;
    let mut y = y_guess; 

    const MAX_ITERATIONS: usize = 100;
    const TOLERANCE: f64 = 1e-10;

    for _ in 0..MAX_ITERATIONS {
        let (fx, fy, jacobian) = compose_henon_n_times(x, y, a, b, period);

        let rx = fx - x; 
        let ry = fy - y;

        let error = (rx * rx + ry * ry).sqrt();

        if error < TOLERANCE {
            if verify_minimal_period(x, y, a, b, period) {
                return Some(Point { x, y });
            } else {
                return None;
            }
        }

        let jac_g = Jacobian {
            j11: jacobian.j11 - 1.0,
            j12: jacobian.j12,
            j21: jacobian.j21,
            j22: jacobian.j22 - 1.0
        };
        
        let jac_g_inv = match jac_g.inverse() {
            Some(inv) => inv, 
            None => return None
        };

        let dx = jac_g_inv.j11 * (-rx) + jac_g_inv.j12 * (-ry);
        let dy = jac_g_inv.j21 * (-rx) + jac_g_inv.j22 * (-ry);

        x += dx; 
        y += dy;

        if x.abs() > 100.0 || y.abs() > 100.0 {
            return None
        }
    } 

    None
}

fn main() {
    let a = 1.4;
    let b = 0.3;

    println!("Searching for period-1 fixed points:");
    for i in 0..20 {
        for j in 0..20 {
            let x_guess = -2.0 + 4.0 * (i as f64) / 20.0;
            let y_guess = -2.0 + 4.0 * (j as f64) / 20.0;

            if let Some(fp) = find_periodic_point_near(x_guess, y_guess, a, b, 1) {
                println!("Period-1: ({:.10}, {:.10})", fp.x, fp.y);
            }
        }
    }

    for i in 0..20 {
        for j in 0..20 {
            let x_guess = -2.0 + 4.0 * (i as f64) / 20.0;
            let y_guess = -2.0 + 4.0 * (j as f64) / 20.0;

            if let Some(fp) = find_periodic_point_near(x_guess, y_guess, a, b, 2) {
                println!("Period-2: ({:.10}, {:.10})", fp.x, fp.y);
            }
        }
    }
}


pub fn precompute_periodic_orbits(a: f64, b: f64) -> PeriodicOrbitDatabase {
    let mut database = PeriodicOrbitDatabase::new();
}

pub struct PeriodicOrbitDatabase {
    orbits: Vec<PeriodicOrbit>,
    spatial_index: KDTree,
}

impl PeriodicOrbitDatabase {
    pub fn find_matching_orbit(&self, x: f64, y: f64, tolerance: f64) -> Option<&PeriodicOrbit> {
        for orbit in &self.orbits {
            for point in &orbit.points {
                let distance = ((x - point.x).powi(2) + (y - point.y).powi(2)).sqrt();

                if distance < tolerance {
                    return Some(orbit);
                }
            }
        }

        None
    }

    
}

pub struct PeriodicOrbit {
    pub points: Vec<Point>,
    pub period: usize,
    pub stability: StabilityType
}

