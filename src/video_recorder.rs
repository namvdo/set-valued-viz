use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecordingStatus {
    Idle,
    Recording,
    Encoding,
    Complete,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameInfo {
    pub frame_index: u32,
    pub timestamp_ms: f64,
    pub parameter_value: f64,
}

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct VideoConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub crf: u32,
}

#[wasm_bindgen]
impl VideoConfig {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32, fps: u32, crf: u32) -> Self {
        Self {
            width,
            height,
            fps,
            crf,
        }
    }

    #[wasm_bindgen]
    pub fn default_config() -> Self {
        Self {
            width: 1280,
            height: 720,
            fps: 30,
            crf: 23,
        }
    }
}

/// Video recorder state machine for coordinating frame capture
#[wasm_bindgen]
pub struct VideoRecorder {
    status: RecordingStatus,
    config: VideoConfig,
    frame_count: u32,
    start_time_ms: f64,

    // Parameters for filename
    dynamic_name: String,
    param_a: f64,
    param_b: f64,
    param_epsilon: f64,
    animated_param: String,
    range_start: f64,
    range_end: f64,
}

#[wasm_bindgen]
impl VideoRecorder {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            status: RecordingStatus::Idle,
            config: VideoConfig::default_config(),
            frame_count: 0,
            start_time_ms: 0.0,
            dynamic_name: "henon".to_string(),
            param_a: 0.0,
            param_b: 0.0,
            param_epsilon: 0.0,
            animated_param: "a".to_string(),
            range_start: 0.0,
            range_end: 0.0,
        }
    }

    /// Start recording with current parameters
    #[wasm_bindgen]
    pub fn start_recording(
        &mut self,
        a: f64,
        b: f64,
        epsilon: f64,
        animated_param: &str,
        range_start: f64,
        range_end: f64,
    ) -> bool {
        if self.status != RecordingStatus::Idle {
            return false;
        }

        self.param_a = a;
        self.param_b = b;
        self.param_epsilon = epsilon;
        self.animated_param = animated_param.to_string();
        self.range_start = range_start;
        self.range_end = range_end;
        self.frame_count = 0;
        self.start_time_ms = js_sys::Date::now();
        self.status = RecordingStatus::Recording;

        true
    }

    /// Record a frame timestamp
    #[wasm_bindgen]
    pub fn add_frame(&mut self, _parameter_value: f64) -> u32 {
        if self.status != RecordingStatus::Recording {
            return 0;
        }

        self.frame_count += 1;
        self.frame_count
    }

    /// Get current frame count
    #[wasm_bindgen]
    pub fn get_frame_count(&self) -> u32 {
        self.frame_count
    }

    /// Get recording status
    #[wasm_bindgen]
    pub fn get_status(&self) -> RecordingStatus {
        self.status
    }

    /// Set status to encoding
    #[wasm_bindgen]
    pub fn start_encoding(&mut self) {
        if self.status == RecordingStatus::Recording {
            self.status = RecordingStatus::Encoding;
        }
    }

    /// Set status to complete
    #[wasm_bindgen]
    pub fn finish_encoding(&mut self) {
        self.status = RecordingStatus::Complete;
    }

    /// Set status to error
    #[wasm_bindgen]
    pub fn set_error(&mut self) {
        self.status = RecordingStatus::Error;
    }

    /// Reset recorder to idle
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.status = RecordingStatus::Idle;
        self.frame_count = 0;
    }

    /// Generate filename based on parameters
    /// Format: henon_a{a}_b{b}_eps{eps}_{animated}_{start}to{end}.mp4
    #[wasm_bindgen]
    pub fn generate_filename(&self) -> String {
        let a_str = format!("{:.3}", self.param_a).replace(".", "p");
        let b_str = format!("{:.3}", self.param_b).replace(".", "p");
        let eps_str = format!("{:.4}", self.param_epsilon).replace(".", "p");
        let start_str = format!("{:.3}", self.range_start)
            .replace(".", "p")
            .replace("-", "m");
        let end_str = format!("{:.3}", self.range_end)
            .replace(".", "p")
            .replace("-", "m");

        format!(
            "{}_{}_a{}_b{}_eps{}_{}_to_{}.mp4",
            self.dynamic_name, self.animated_param, a_str, b_str, eps_str, start_str, end_str
        )
    }

    /// Get video config
    #[wasm_bindgen]
    pub fn get_config(&self) -> VideoConfig {
        self.config.clone()
    }

    /// Set video config
    #[wasm_bindgen]
    pub fn set_config(&mut self, config: VideoConfig) {
        self.config = config;
    }

    /// Get expected duration in seconds based on frame count and fps
    #[wasm_bindgen]
    pub fn get_expected_duration_secs(&self) -> f64 {
        self.frame_count as f64 / self.config.fps as f64
    }

    /// Check if currently recording
    #[wasm_bindgen]
    pub fn is_recording(&self) -> bool {
        self.status == RecordingStatus::Recording
    }

    /// Check if encoding
    #[wasm_bindgen]
    pub fn is_encoding(&self) -> bool {
        self.status == RecordingStatus::Encoding
    }

    /// Get parameter overlay text for current frame
    #[wasm_bindgen]
    pub fn get_overlay_text(&self, current_param_value: f64) -> String {
        format!(
            "a = {:.4}  b = {:.4}  ε = {:.4}",
            if self.animated_param == "a" {
                current_param_value
            } else {
                self.param_a
            },
            if self.animated_param == "b" {
                current_param_value
            } else {
                self.param_b
            },
            if self.animated_param == "epsilon" {
                current_param_value
            } else {
                self.param_epsilon
            }
        )
    }
}

impl Default for VideoRecorder {
    fn default() -> Self {
        Self::new()
    }
}
