use nalgebra::{DVector, Vector2};
use nalgebra_sparse::{CscMatrix, CooMatrix};
use rayon::prelude::*;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy)]
struct HyperRect {
    center: Vector2<f64>,
    radius: Vector2<f64> 
}

impl HyperRect {
    // correspond to P = kron(E, X) * diag(r) + kron(c', ones(...))
    fn generate_points(&self, sample_template: &[Vector2<f64>]) -> Vec<Vector2<f64>> {
        sample_template 
            .iter() 
            .map(|pt| {
                Vector2::new(
                    self.center.x + pt.x * self.radius.x, 
                    self.center.y + pt.y * self.radius.y,
                )
            })
            .collect()
    }
} 

struct Grid {
    boxes: Vec<HyperRect>,
    domain_min: Vector2<f64>,
    step: Vector2<f64>,
    dims: (usize, usize)
}

impl Grid {
    fn new(min: Vector2<f64>, max: Vector2<f64>, subdivisions: usize) -> Self {
        let step = (max - min) / subdivisions as f64;
        let mut boxes = Vec::with_capacity(subdivisions * subdivisions);

        for j in 0..subdivisions {
            for i in 0..subdivisions {
                let center = Vector2::new(
                    min.x + step.x * (i as f64 + 0.5),
                    min.y + step.y * (j as f64 + 0.5),
                );

                boxes.push(HyperRect {
                    center, 
                    radius: step / 2.0,
                })
            }
        }

        Grid {
            boxes,
            domain_min: min,
            step,
            dims: (subdivisions, subdivisions)
        }
    }


    // correspond to t.search(eP', depth) in matlab implementation

    fn search(&self, point: &Vector2<f64>) -> Option<usize> {
        let rel = point - self.domain_min;

        if (rel.x < 0.0 || rel.y < 0.0) { return None; }

        let ix = (rel.x / self.step.x).floor() as usize;
        let iy = (rel.y / self.step.y).floor() as usize;

        if ix >= self.dims.0 || iy >= self.dims.1 {
            return None;
        }
        Some(iy * self.dims.0 + ix)
    }
}


    