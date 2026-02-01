use crate::unstable_manifold::HenonParams;
use nalgebra::Vector2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

macro_rules! console_log {
    ($($t:tt)*) => {
        log(&format!($($t)*))
    }
}

/// A hyperrectangular box in 2D space
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HyperRect {
    pub center: (f64, f64),
    pub radius: (f64, f64),
}

impl HyperRect {
    /// Check if this box intersects with an epsilon-ball centered at a point
    pub fn intersects_ball(&self, point: (f64, f64), epsilon: f64) -> bool {
        // Check if the box intersects with a ball of radius epsilon centered at point
        // Using the standard box-circle intersection test
        let dx = (point.0 - self.center.0).abs();
        let dy = (point.1 - self.center.1).abs();

        // If the point is too far away in either axis, no intersection
        if dx > self.radius.0 + epsilon {
            return false;
        }
        if dy > self.radius.1 + epsilon {
            return false;
        }

        // If the point is close enough in both axes, definitely intersects
        if dx <= self.radius.0 || dy <= self.radius.1 {
            return true;
        }

        // Check corner case using circle equation
        let corner_dist_sq = (dx - self.radius.0).powi(2) + (dy - self.radius.1).powi(2);
        corner_dist_sq <= epsilon * epsilon
    }

    /// Check if a point is inside this box
    pub fn contains(&self, point: (f64, f64)) -> bool {
        (point.0 - self.center.0).abs() <= self.radius.0
            && (point.1 - self.center.1).abs() <= self.radius.1
    }
}

/// Grid structure for the Ulam method
#[derive(Clone)]
pub struct Grid {
    pub boxes: Vec<HyperRect>,
    pub domain_min: Vector2<f64>,
    pub domain_max: Vector2<f64>,
    pub step: Vector2<f64>,
    pub dims: (usize, usize),
}

impl Grid {
    pub fn new(min: Vector2<f64>, max: Vector2<f64>, subdivisions: usize) -> Self {
        let step = (max - min) / (subdivisions as f64);
        let mut boxes = Vec::with_capacity(subdivisions * subdivisions);

        for j in 0..subdivisions {
            for i in 0..subdivisions {
                let center = Vector2::new(
                    min.x + step.x * (i as f64 + 0.5),
                    min.y + step.y * (j as f64 + 0.5),
                );

                boxes.push(HyperRect {
                    center: (center.x, center.y),
                    radius: (step.x / 2.0, step.y / 2.0),
                })
            }
        }

        Grid {
            boxes,
            domain_min: min,
            domain_max: max,
            step,
            dims: (subdivisions, subdivisions),
        }
    }

    /// Find the box index containing a point
    pub fn search(&self, point: &Vector2<f64>) -> Option<usize> {
        let rel = point - self.domain_min;

        if rel.x < 0.0 || rel.y < 0.0 {
            return None;
        }

        let ix = (rel.x / self.step.x).floor() as usize;
        let iy = (rel.y / self.step.y).floor() as usize;

        if ix >= self.dims.0 || iy >= self.dims.1 {
            return None;
        }
        Some(iy * self.dims.0 + ix)
    }

    /// Find all boxes that intersect with an epsilon-ball centered at a point
    /// This implements the GAIO-style epsilon inflation
    pub fn find_intersecting_boxes(&self, point: &Vector2<f64>, epsilon: f64) -> Vec<usize> {
        let mut result = Vec::new();

        // Calculate the bounding box of potential intersections
        let search_min_x = point.x - epsilon;
        let search_max_x = point.x + epsilon;
        let search_min_y = point.y - epsilon;
        let search_max_y = point.y + epsilon;

        // Convert to grid indices (with bounds checking)
        let rel_min = Vector2::new(search_min_x, search_min_y) - self.domain_min;
        let rel_max = Vector2::new(search_max_x, search_max_y) - self.domain_min;

        let ix_min = if rel_min.x < 0.0 {
            0
        } else {
            (rel_min.x / self.step.x).floor() as usize
        };
        let iy_min = if rel_min.y < 0.0 {
            0
        } else {
            (rel_min.y / self.step.y).floor() as usize
        };
        let ix_max = ((rel_max.x / self.step.x).ceil() as usize).min(self.dims.0);
        let iy_max = ((rel_max.y / self.step.y).ceil() as usize).min(self.dims.1);

        // Check each box in the potential range
        for iy in iy_min..iy_max {
            for ix in ix_min..ix_max {
                let idx = iy * self.dims.0 + ix;
                if idx < self.boxes.len() {
                    let box_ref = &self.boxes[idx];
                    if box_ref.intersects_ball((point.x, point.y), epsilon) {
                        result.push(idx);
                    }
                }
            }
        }

        result
    }
}

/// UlamComputer computes the transition matrix and invariant measures
/// using the Ulam/GAIO method with epsilon-inflation
#[wasm_bindgen]
pub struct UlamComputer {
    grid: Grid,
    transitions: HashMap<usize, Vec<(usize, f64)>>,
    right_eigenvector: Vec<f64>, // Invariant measure (forward dynamics)
    left_eigenvector: Vec<f64>,  // Invariant measure (backward dynamics)
    epsilon: f64,
}

#[wasm_bindgen]
impl UlamComputer {
    /// Create a new UlamComputer with the given parameters
    ///
    /// # Arguments
    /// * `a` - Henon map parameter a
    /// * `b` - Henon map parameter b  
    /// * `subdivisions` - Number of grid subdivisions in each dimension
    /// * `points_per_box` - Number of sample points per box (will be squared for grid)
    /// * `epsilon` - Epsilon parameter for ball inflation (boundary detection)
    #[wasm_bindgen(constructor)]
    pub fn new(
        a: f64,
        b: f64,
        subdivisions: usize,
        points_per_box: usize,
        epsilon: f64,
    ) -> Result<UlamComputer, String> {
        let params = HenonParams::new(a, b, 0.001)?;

        // Domain for Henon map visualization
        let min = Vector2::new(-2.0, -1.5);
        let max = Vector2::new(2.0, 1.5);
        let grid = Grid::new(min, max, subdivisions);

        let n_boxes = grid.boxes.len();

        // Determine grid sampling pattern
        // points_per_box controls density, we'll use sqrt(points_per_box) x sqrt(points_per_box) grid
        let samples_per_dim = (points_per_box as f64).sqrt().ceil() as usize;
        let actual_samples = samples_per_dim * samples_per_dim;

        console_log!(
            "Ulam: {} boxes, {}x{} samples/box = {} samples, epsilon = {}",
            n_boxes,
            samples_per_dim,
            samples_per_dim,
            actual_samples,
            epsilon
        );

        // Build transition matrix using set-valued approach with epsilon-inflation
        let mut transitions: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();

        for i in 0..n_boxes {
            let rect = &grid.boxes[i];
            let center = Vector2::new(rect.center.0, rect.center.1);
            let radius = Vector2::new(rect.radius.0, rect.radius.1);

            // Count transitions to each target box
            let mut counts: HashMap<usize, usize> = HashMap::new();
            let mut total_valid = 0usize;

            // Sample points uniformly on a grid within the box
            for sy in 0..samples_per_dim {
                for sx in 0..samples_per_dim {
                    // Map [0, samples_per_dim-1] to [-1, 1] range within box
                    let tx = if samples_per_dim > 1 {
                        -1.0 + 2.0 * (sx as f64) / ((samples_per_dim - 1) as f64)
                    } else {
                        0.0
                    };
                    let ty = if samples_per_dim > 1 {
                        -1.0 + 2.0 * (sy as f64) / ((samples_per_dim - 1) as f64)
                    } else {
                        0.0
                    };

                    let pt = Vector2::new(center.x + tx * radius.x, center.y + ty * radius.y);

                    // Map the point through Henon map
                    if let Ok(mapped) = params.henon_map(&pt) {
                        // Find all boxes intersecting the epsilon-ball around the mapped point
                        // This is the key GAIO/set-valued enhancement
                        let intersecting = grid.find_intersecting_boxes(&mapped, epsilon);

                        if !intersecting.is_empty() {
                            total_valid += 1;
                            // Each intersecting box gets a count (normalized later)
                            for target_idx in intersecting {
                                *counts.entry(target_idx).or_insert(0) += 1;
                            }
                        }
                    }
                }
            }

            // Normalize to get transition probabilities
            if total_valid > 0 {
                let mut probs = Vec::with_capacity(counts.len());
                let total = counts.values().sum::<usize>() as f64;
                for (target, count) in counts {
                    probs.push((target, (count as f64) / total));
                }
                transitions.insert(i, probs);
            }
        }

        // Compute both eigenvectors using power iteration
        let right_eigenvector = Self::compute_right_eigenvector(&transitions, n_boxes, 100);
        let left_eigenvector = Self::compute_left_eigenvector(&transitions, n_boxes, 100);

        console_log!(
            "Ulam computation complete. Right EV sum: {:.6}, Left EV sum: {:.6}",
            right_eigenvector.iter().sum::<f64>(),
            left_eigenvector.iter().sum::<f64>()
        );

        Ok(UlamComputer {
            grid,
            transitions,
            right_eigenvector,
            left_eigenvector,
            epsilon,
        })
    }

    /// Compute the right eigenvector (invariant measure for forward dynamics)
    /// This is the stationary distribution: mu * P = mu
    fn compute_right_eigenvector(
        transitions: &HashMap<usize, Vec<(usize, f64)>>,
        n_boxes: usize,
        iterations: usize,
    ) -> Vec<f64> {
        let mut measure = vec![1.0 / (n_boxes as f64); n_boxes];
        let mut next_measure = vec![0.0; n_boxes];

        for _ in 0..iterations {
            next_measure.fill(0.0);

            for i in 0..n_boxes {
                if measure[i] < 1e-15 {
                    continue;
                }

                if let Some(targets) = transitions.get(&i) {
                    for (tgt, prob) in targets {
                        if *tgt < n_boxes {
                            next_measure[*tgt] += measure[i] * prob;
                        }
                    }
                }
            }

            let total_mass: f64 = next_measure.iter().sum();
            if total_mass > 1e-15 {
                let scale = 1.0 / total_mass;
                for x in next_measure.iter_mut() {
                    *x *= scale;
                }
                measure.copy_from_slice(&next_measure);
            } else {
                break;
            }
        }

        measure
    }

    /// Compute the left eigenvector (invariant measure for backward dynamics)
    /// This is computed on the transpose: P^T * v = v
    fn compute_left_eigenvector(
        transitions: &HashMap<usize, Vec<(usize, f64)>>,
        n_boxes: usize,
        iterations: usize,
    ) -> Vec<f64> {
        // Build transpose of the transition matrix
        let mut transpose: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();

        for (from, targets) in transitions.iter() {
            for (to, prob) in targets {
                transpose
                    .entry(*to)
                    .or_insert_with(Vec::new)
                    .push((*from, *prob));
            }
        }

        // Now apply power iteration on transpose
        let mut measure = vec![1.0 / (n_boxes as f64); n_boxes];
        let mut next_measure = vec![0.0; n_boxes];

        for _ in 0..iterations {
            next_measure.fill(0.0);

            for i in 0..n_boxes {
                if measure[i] < 1e-15 {
                    continue;
                }

                if let Some(targets) = transpose.get(&i) {
                    for (tgt, prob) in targets {
                        if *tgt < n_boxes {
                            next_measure[*tgt] += measure[i] * prob;
                        }
                    }
                }
            }

            let total_mass: f64 = next_measure.iter().sum();
            if total_mass > 1e-15 {
                let scale = 1.0 / total_mass;
                for x in next_measure.iter_mut() {
                    *x *= scale;
                }
                measure.copy_from_slice(&next_measure);
            } else {
                break;
            }
        }

        measure
    }

    /// Get the grid boxes as a serialized array
    pub fn get_grid_boxes(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.grid.boxes).unwrap()
    }

    /// Get the transitions from a specific box index
    pub fn get_transitions(&self, from_box_idx: usize) -> JsValue {
        if let Some(probs) = self.transitions.get(&from_box_idx) {
            let result: Vec<serde_json::Value> = probs
                .iter()
                .map(|(idx, p)| {
                    serde_json::json!({
                        "index": idx,
                        "probability": p
                    })
                })
                .collect();
            serde_wasm_bindgen::to_value(&result).unwrap()
        } else {
            JsValue::NULL
        }
    }

    /// Get the right eigenvector (invariant measure, forward dynamics)
    pub fn get_invariant_measure(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.right_eigenvector).unwrap()
    }

    /// Get the left eigenvector (backward invariant measure)
    pub fn get_left_eigenvector(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.left_eigenvector).unwrap()
    }

    /// Get the epsilon parameter used for this computation
    pub fn get_epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Get the grid step size (useful for UI scaling)
    pub fn get_grid_step(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&(self.grid.step.x, self.grid.step.y)).unwrap()
    }

    pub fn get_box_index(&self, x: f64, y: f64) -> isize {
        match self.grid.search(&Vector2::new(x, y)) {
            Some(idx) => idx as isize,
            None => -1,
        }
    }

    pub fn get_intersecting_boxes(&self, x: f64, y: f64) -> JsValue {
        let point = Vector2::new(x, y);
        let boxes = self.grid.find_intersecting_boxes(&point, self.epsilon);
        serde_wasm_bindgen::to_value(&boxes).unwrap()
    }

    pub fn get_dimensions(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.grid.dims).unwrap()
    }
}
