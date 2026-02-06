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

    

}