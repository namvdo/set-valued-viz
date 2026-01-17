use std::char::MAX;
use std::cmp::max;
use std::f64;
#[derive(Debug, Clone)]
pub struct Point {
    pub x: f64, 
    pub y: f64,
}

#[derive(Debug, Clone)]
pub struct Jacobian {
    pub j11: f64,
    pub j12: f64, 
    pub j21: f64, 
    pub j22: f64
}

#[derive(Debug, Clone)]
pub struct FixedPoint {
    pub x: f64, 
    pub y: f64, 
    pub period: usize, 
    pub stability: StabilityType,
}



#[derive(Debug, Clone)]
pub enum StabilityType {
    Stable,
    Unstable,
    Saddle
}


#[derive(Debug, Clone)]
pub struct BoundaryPoint {
    pub x: f64,
    pub y: f64,
    pub nx: f64,
    pub ny: f64,
    pub classification: PointClassification
}


#[derive(Debug, Clone)]
pub enum PointClassification {
    Regular,
    NearPeriodicOrbit {
        period: usize,
        stability: StabilityType,
        distance: f64
    }
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

    fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        (self.j11 * x + self.j12 * y, self.j21 * x + self.j22 * y)
    }

    fn multiply(&self, other: &Jacobian) -> Jacobian {
        Jacobian {
            j11: self.j11 * other.j11 + self.j12 * other.j21, 
            j12: self.j11 * other.j12 + self.j12 * other.j22, 
            j21: self.j21 * other.j11 + self.j22 * other.j21, 
            j22: self.j21 * other.j12 + self.j22 * other.j22
        }
    }

    fn eigenvalues(&self) -> (f64, f64, bool) {
        let trace = self.j11 + self.j22;
        let det = self.j11 * self.j22 - self.j12 * self.j21;
        let discriminant = trace * trace - 4.0 * det;

        if discriminant >= 0.0 {
            let sqrt_disc = discriminant.sqrt();
            let lambda1 = (trace + sqrt_disc) / 2.0;
            let lambda2 = (trace - sqrt_disc) / 2.0;
            (lambda1, lambda2, false)
        } else {
            let modulus = det.sqrt();
            (modulus, modulus, true)
        }
    }
}


fn classify_stability(jac: &Jacobian) -> StabilityType {
    let (lambda1, lambda2, is_complex) = jac.eigenvalues();

    if is_complex {
        if lambda1 < 1.0 {
            StabilityType::Stable
        } else {
            StabilityType::Unstable
        }
    } else {
        let l1_stable = lambda1.abs() < 1.0;
        let l2_stable = lambda2.abs() < 1.0;

        if l1_stable && l2_stable {
            StabilityType::Stable
        } else if !l1_stable && !l2_stable {
            StabilityType::Unstable
        } else {
            StabilityType::Saddle
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

pub fn find_periodic_point_near(x_guess: f64, y_guess: f64, a: f64, b: f64, period: usize) -> Option<FixedPoint> {
    let mut x = x_guess;
    let mut y = y_guess;

    const MAX_ITERATIONS: usize = 100;
    const TOLERANCE: f64 = 1e-10;

    for _iter in 0..MAX_ITERATIONS {
        let (fx, fy, jac_fn) = compose_henon_n_times(x, y, a, b, period);

        let rx = fx - x;
        let ry = fy - y;

        let error = (rx * rx + ry * ry).sqrt();

        if error < TOLERANCE {
            if verify_minimal_period(x, y, a, b, period) {
                let stability = classify_stability(&jac_fn);

                return Some(FixedPoint {
                    x,
                    y,
                    period,
                    stability
                });
            } else {
                return None;
            }
        }
        let jac_g = Jacobian {
            j11: jac_fn.j11 - 1.0,
            j12: jac_fn.j12,
            j21: jac_fn.j21,
            j22: jac_fn.j22 - 1.0,
        };

        let jac_g_inv = match jac_g.inverse() {
            Some(inv) => inv,
            None => return None,
        };

        let (dx, dy) = jac_g_inv.apply(-rx, -ry);
        x += dx;
        y += dy;

        if x.abs() > 100.0 || y.abs() > 100.0 {
            return None;
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


pub fn precompute_periodic_orbits(a: f64, b: f64, max_period: usize) -> PeriodicOrbitDatabase {
    let mut database = PeriodicOrbitDatabase::new(a, b);

    for period in 1..=max_period {
        let mut found_this_period = 0;

        let grid_size = if period == 1 { 30 } else { 50 };

        for i in 0..grid_size {
            for j in 0..grid_size {
                let x_guess = -2.0 + 4.0 * (i as f64) / grid_size as f64;
                let y_guess = -2.0 + 4.0 * (j as f64) / grid_size as f64;

                if let Some(fixed_point) = find_periodic_point_near(x_guess, y_guess, a, b, period) {
                    let before_count = database.total_count();
                    database.add_if_new(fixed_point)
                }
            }
        }
    }
    database
}

pub struct PeriodicOrbitDatabase {
    orbits: Vec<PeriodicOrbit>,
    a: f64,
    b: f64
}

impl PeriodicOrbitDatabase {
    pub fn new(a: f64, b: f64) -> Self {
        Self {
            orbits: Vec::new(),
            a,
            b,
        }
    }
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

    pub fn add_if_new(&mut self, fixed_point: FixedPoint) {
        const DUPLICATE_TOLERANCE: f64 = 1e-6;

        for existing_orbit in &self.orbits {
            if existing_orbit.period != fixed_point.period {
                continue;
            }
            for existing_point in &existing_orbit.points {
                let distance = ((existing_point.x - fixed_point.x).powi(2) + (existing_point.y - fixed_point.y).powi(2)).sqrt();
                if distance < DUPLICATE_TOLERANCE {
                    return;
                }
            }
        }
        let orbit = self.construct_full_orbit(fixed_point);
        self.orbits.push(orbit);
    }

    pub fn total_count(&self) -> usize {
        self.orbits.len()
    }

    fn construct_full_orbit(&self, start_point: FixedPoint) -> PeriodicOrbit {
        let mut points = vec![Point{
            x: start_point.x,
            y: start_point.y,
        }];

        let mut x = start_point.x;
        let mut y = start_point.y;

        for _ in 1..start_point.period {
            (x, y) = henon_map(x, y, self.a, self.b);
            points.push(Point {x, y})
        }

        PeriodicOrbit {
            points,
            period: start_point.period,
            stability: start_point.stability
        }
    }

}

pub struct PeriodicOrbit {
    pub points: Vec<Point>,
    pub period: usize,
    pub stability: StabilityType
}

impl PeriodicOrbit {
    fn representative_point(&self) -> &Point {
        &self.points[0]
    }
}


fn initialize_circular_boundary(
    center: (f64, f64),
    epsilon: f64,
    num_points: usize
) -> Vec<BoundaryPoint> {
    (0..num_points)
        .map(|i| {
            let theta = 2.0 * std::f64::consts::PI * (i as f64) / (num_points as f64);
            let x = center.0 + epsilon * theta.cos();
            let y = center.1 + epsilon * theta.sin();
            let nx = theta.cos();
            let ny = theta.sin();

            BoundaryPoint {
                x,
                y,
                nx,
                ny,
                classification: PointClassification::Regular
            }
        }).collect()
}


fn evolve_boundary_step(
    boundary: &[BoundaryPoint],
    a: f64,
    b: f64,
    epsilon: f64
) -> Vec<BoundaryPoint> {
    boundary
        .iter()
        .map(|point| {
            let (fx, fy) = henon_map(point.x as f64, point.y as f64, a, b);
            let jac = henon_jacobian(point.x, point.y, a, b);

            let jac_inv = match jac.inverse() {
                Some(inv) => inv,
                None => {
                    return BoundaryPoint {
                        x: fx,
                        y: fy,
                        nx: point.nx,
                        ny: point.ny,
                        classification: PointClassification::Regular
                    };
                }
            };

            let nx_new = jac_inv.j11 * point.nx + jac_inv.j21 * point.ny;
            let ny_new = jac_inv.j12 * point.nx + jac_inv.j22 * point.ny;

            let norm = (nx_new * nx_new + ny_new * ny_new).sqrt();

            let (nx_norm, ny_norm) = if norm > 1e-12 {
                (nx_new / norm, ny_new / norm)
            } else {
                (point.nx, point.ny)
            };

            let x_new = fx + epsilon * nx_norm;
            let y_new = fy + epsilon * ny_norm;

            BoundaryPoint {
                x: x_new,
                y: y_new,
                nx: nx_norm,
                ny: ny_norm,
                classification: PointClassification::Regular
            }
        }).collect()
}
