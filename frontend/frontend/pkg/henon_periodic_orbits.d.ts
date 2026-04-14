/* tslint:disable */
/* eslint-disable */
export function compute_manifold_simple(a: number, b: number, epsilon: number, x_min: number, x_max: number, y_min: number, y_max: number): any;
export function compute_manifold_js(a: number, b: number, epsilon: number, saddle_x: number, saddle_y: number, period: number, eigenvector_x: number, eigenvector_y: number, eigenvalue: number, is_dual_repeller: boolean): any;
export function compute_user_defined_manifold(x_eq: string, y_eq: string, params: any, epsilon: number, x_min: number, x_max: number, y_min: number, y_max: number): any;
/**
 * orbits_js: Array of {points: [[x,y],...], period: number, stability: "stable"|"saddle"|"unstable"}
 */
export function compute_manifold_from_orbits(a: number, b: number, epsilon: number, x_min: number, x_max: number, y_min: number, y_max: number, orbits_js: any): any;
export function evaluate_user_defined_map(x: number, y: number, x_eq: string, y_eq: string, params: any, epsilon: number): any;
export function compute_stable_and_unstable_manifolds(a: number, b: number, epsilon: number, x_min: number, x_max: number, y_min: number, y_max: number, orbits_js: any, intersection_threshold: number): any;
export function compute_manifold_from_orbits_user_defined(x_eq: string, y_eq: string, params: any, epsilon: number, x_min: number, x_max: number, y_min: number, y_max: number, orbits_js: any): any;
export function compute_stable_and_unstable_manifolds_user_defined(x_eq: string, y_eq: string, params: any, epsilon: number, x_min: number, x_max: number, y_min: number, y_max: number, orbits_js: any, intersection_threshold: number): any;
export function boundary_map_duffing_ode(x: number, y: number, nx: number, ny: number, delta: number, h: number, epsilon: number): any;
export function evaluate_user_defined_ode(x: number, y: number, x_eq: string, y_eq: string, params: any): any;
export function boundary_map_user_defined_ode(x: number, y: number, nx: number, ny: number, x_eq: string, y_eq: string, params: any, h: number, epsilon: number): any;
export function boundary_map_user_defined(x: number, y: number, nx: number, ny: number, x_eq: string, y_eq: string, params: any, epsilon: number): any;
/**
 * Legacy Hénon-specific sweep (kept for backward compat, routes through generic pipeline)
 */
export function parameterSweep(b: number, epsilon: number, a_min: number, a_max: number, num_samples: number, max_period: number, x_min: number, x_max: number, y_min: number, y_max: number): any;
/**
 * Unified parameter sweep: works for any system type + any parameter.
 */
export function parameterSweepGeneric(system_type: string, x_eq: string, y_eq: string, params_js: any, sweep_param_name: string, sweep_min: number, sweep_max: number, num_samples: number, epsilon: number, max_period: number, x_min: number, x_max: number, y_min: number, y_max: number): any;
export function parameterSweepCsv(b: number, epsilon: number, a_min: number, a_max: number, num_samples: number, max_period: number, x_min: number, x_max: number, y_min: number, y_max: number): string;
export function compute_duffing_manifold_simple(a: number, b: number, epsilon: number, x_min: number, x_max: number, y_min: number, y_max: number): any;
export function compute_hausdorff_distance_between_manifolds(unstable_plus_js: any, unstable_minus_js: any, stable_plus_js: any, stable_minus_js: any, num_closest_pairs: number): any;
export function compute_bifurcation_hausdorff(b: number, epsilon: number, a_min: number, a_max: number, num_samples: number): any;
export enum DuffingStabilityType {
  Stable = 0,
  Unstable = 1,
  Saddle = 2,
}
export enum PeriodicType {
  Stable = 0,
  Unstable = 1,
  Saddle = 2,
}
export enum RecordingStatus {
  Idle = 0,
  Recording = 1,
  Encoding = 2,
  Complete = 3,
  Error = 4,
}
export enum StabilityType {
  Stable = 0,
  Unstable = 1,
  Saddle = 2,
}
export class BdeSimulatorUserDefinedWasm {
  free(): void;
  constructor(x_eq: string, y_eq: string, params: any, epsilon: number, cx: number, cy: number, r: number, num_points: number);
  step(h: number): any;
  get_points(): any;
  reparameterize(): void;
  has_self_intersection(gap: number): number;
  get_fold_indices(speed_threshold: number): any;
}
export class BdeSimulatorWasm {
  free(): void;
  constructor(delta: number, epsilon: number, cx: number, cy: number, r: number, num_points: number);
  step(h: number): any;
  get_points(): any;
  /**
   * arc-length reparameterize: redistribute points evenly along the curve.
   */
  reparameterize(): void;
  has_self_intersection(gap: number): number;
  get_fold_indices(speed_threshold: number): any;
}
export class BoundaryHenonSystemWasm {
  free(): void;
  constructor(a: number, b: number, epsilon: number, max_period: number, x_min: number, x_max: number, y_min: number, y_max: number);
  getPeriodicOrbits(): any;
  trackTrajectory(initial_x: number, initial_y: number, initial_nx: number, initial_ny: number, max_iterations: number): void;
  getCurrentPoint(): any;
  getTrajectory(start: number, end: number): any;
  step(): boolean;
  reset(): void;
  getTotalIterations(): number;
  getCurrentIteration(): number;
  getOrbitCount(): number;
  getEpsilon(): number;
}
export class BoundaryUserDefinedSystemWasm {
  free(): void;
  constructor(x_eq: string, y_eq: string, params: any, epsilon: number, max_period: number, x_min: number, x_max: number, y_min: number, y_max: number);
  getPeriodicOrbits(): any;
  getOrbitCount(): number;
  getEpsilon(): number;
}
export class DuffingParams {
  private constructor();
  free(): void;
  a: number;
  b: number;
  epsilon: number;
}
export class DuffingSystemWasm {
  free(): void;
  constructor(a: number, b: number, max_period: number);
  getPeriodicOrbits(): any;
  trackTrajectory(initial_x: number, initial_y: number, max_iterations: number): void;
  getCurrentPoint(): any;
  getTrajectory(start: number, end: number): any;
  step(): boolean;
  reset(): void;
  getTotalIterations(): number;
  getCurrentIteration(): number;
  getOrbitCount(): number;
}
export class EulerMapSystemWasm {
  free(): void;
  constructor(delta: number, h: number, epsilon: number, max_period: number);
  getPeriodicOrbits(): any;
  trackTrajectory(_initial_x: number, _initial_y: number, _max_iterations: number): void;
  getCurrentPoint(): any;
  getTrajectory(_start: number, _end: number): any;
  step(): boolean;
  reset(): void;
  getTotalIterations(): number;
  getCurrentIteration(): number;
  getOrbitCount(): number;
}
export class ExtendedPoint {
  private constructor();
  free(): void;
  x: number;
  y: number;
  nx: number;
  ny: number;
}
export class HenonParams {
  private constructor();
  free(): void;
  a: number;
  b: number;
  epsilon: number;
}
/**
 * UlamComputer computes the transition matrix and invariant measures
 * using the Ulam/GAIO method with epsilon-inflation
 */
export class UlamComputer {
  free(): void;
  /**
   * Create a new UlamComputer with the given parameters
   *
   * # Arguments
   * * `a` - Henon map parameter a
   * * `b` - Henon map parameter b  
   * * `subdivisions` - Number of grid subdivisions in each dimension
   * * `points_per_box` - Number of sample points per box (will be squared for grid)
   * * `epsilon` - Epsilon parameter for ball inflation (boundary detection)
   */
  constructor(a: number, b: number, subdivisions: number, points_per_box: number, epsilon: number, x_min: number, x_max: number, y_min: number, y_max: number);
  /**
   * Get the grid boxes as a serialized array
   */
  get_grid_boxes(): any;
  get_transitions(from_box_idx: number): any;
  /**
   * Get the right eigenvector (invariant measure, forward dynamics)
   */
  get_invariant_measure(): any;
  /**
   * Get the left eigenvector (backward invariant measure)
   */
  get_left_eigenvector(): any;
  /**
   * Get the epsilon parameter used for this computation
   */
  get_epsilon(): number;
  /**
   * Get the grid step size (useful for UI scaling)
   */
  get_grid_step(): any;
  get_box_index(x: number, y: number): number;
  get_intersecting_boxes(x: number, y: number): any;
  get_dimensions(): any;
}
export class UlamComputerContinuous {
  free(): void;
  /**
   * Build the Ulam matrix for the Duffing ODE  ẋ=y, ẏ=x−x³−δy
   * using the time-T flow map as the generating discrete map.
   *
   * Arguments
   * * `delta`        – damping δ
   * * `capital_t`    – integration time T per discrete step
   * * `subdivisions` – grid cells per axis
   * * `points_per_box` – sample density
   * * `epsilon`      – epsilon ball inflation for set-valued images
   */
  constructor(delta: number, capital_t: number, subdivisions: number, points_per_box: number, epsilon: number, x_min: number, x_max: number, y_min: number, y_max: number);
  get_grid_boxes(): any;
  get_transitions(from_box_idx: number): any;
  get_invariant_measure(): any;
  get_left_eigenvector(): any;
  get_epsilon(): number;
  get_grid_step(): any;
  get_dimensions(): any;
  get_box_index(x: number, y: number): number;
}
export class UlamComputerContinuousUserDefined {
  free(): void;
  /**
   * Build the Ulam matrix for a user-defined ODE using the time-T flow map.
   *
   * Arguments
   * * `x_eq`, `y_eq` – vector field components ẋ, ẏ
   * * `params`      – parameter list (name/value)
   * * `capital_t`   – integration time T per discrete step
   * * `subdivisions` – grid cells per axis
   * * `points_per_box` – sample density
   * * `epsilon`     – epsilon ball inflation for set-valued images
   */
  constructor(x_eq: string, y_eq: string, params: any, capital_t: number, subdivisions: number, points_per_box: number, epsilon: number, x_min: number, x_max: number, y_min: number, y_max: number);
  get_grid_boxes(): any;
  get_transitions(from_box_idx: number): any;
  get_invariant_measure(): any;
  get_left_eigenvector(): any;
  get_epsilon(): number;
  get_grid_step(): any;
  get_dimensions(): any;
  get_box_index(x: number, y: number): number;
}
export class UlamComputerUserDefined {
  free(): void;
  constructor(x_eq: string, y_eq: string, params: any, subdivisions: number, points_per_box: number, epsilon: number, x_min: number, x_max: number, y_min: number, y_max: number);
  get_grid_boxes(): any;
  get_transitions(from_box_idx: number): any;
  get_invariant_measure(): any;
  get_left_eigenvector(): any;
  get_epsilon(): number;
  get_grid_step(): any;
  get_box_index(x: number, y: number): number;
  get_intersecting_boxes(x: number, y: number): any;
  get_dimensions(): any;
}
export class VideoConfig {
  free(): void;
  constructor(width: number, height: number, fps: number, crf: number);
  static default_config(): VideoConfig;
  width: number;
  height: number;
  fps: number;
  crf: number;
}
/**
 * Video recorder state machine for coordinating frame capture
 */
export class VideoRecorder {
  free(): void;
  constructor();
  /**
   * Start recording with current parameters
   */
  start_recording(a: number, b: number, epsilon: number, animated_param: string, range_start: number, range_end: number): boolean;
  /**
   * Record a frame timestamp
   */
  add_frame(_parameter_value: number): number;
  /**
   * Get current frame count
   */
  get_frame_count(): number;
  /**
   * Get recording status
   */
  get_status(): RecordingStatus;
  /**
   * Set status to encoding
   */
  start_encoding(): void;
  /**
   * Set status to complete
   */
  finish_encoding(): void;
  /**
   * Set status to error
   */
  set_error(): void;
  /**
   * Reset recorder to idle
   */
  reset(): void;
  /**
   * Generate filename based on parameters
   * Format: henon_a{a}_b{b}_eps{eps}_{animated}_{start}to{end}.mp4
   */
  generate_filename(): string;
  /**
   * Get video config
   */
  get_config(): VideoConfig;
  /**
   * Set video config
   */
  set_config(config: VideoConfig): void;
  /**
   * Get expected duration in seconds based on frame count and fps
   */
  get_expected_duration_secs(): number;
  /**
   * Check if currently recording
   */
  is_recording(): boolean;
  /**
   * Check if encoding
   */
  is_encoding(): boolean;
  /**
   * Get parameter overlay text for current frame
   */
  get_overlay_text(current_param_value: number): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_henonparams_free: (a: number, b: number) => void;
  readonly __wbg_get_henonparams_a: (a: number) => number;
  readonly __wbg_set_henonparams_a: (a: number, b: number) => void;
  readonly __wbg_get_henonparams_b: (a: number) => number;
  readonly __wbg_set_henonparams_b: (a: number, b: number) => void;
  readonly __wbg_get_henonparams_epsilon: (a: number) => number;
  readonly __wbg_set_henonparams_epsilon: (a: number, b: number) => void;
  readonly compute_manifold_simple: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
  readonly compute_manifold_js: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => void;
  readonly compute_user_defined_manifold: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => void;
  readonly compute_manifold_from_orbits: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
  readonly evaluate_user_defined_map: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
  readonly compute_stable_and_unstable_manifolds: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => void;
  readonly compute_manifold_from_orbits_user_defined: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => void;
  readonly compute_stable_and_unstable_manifolds_user_defined: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => void;
  readonly __wbg_eulermapsystemwasm_free: (a: number, b: number) => void;
  readonly eulermapsystemwasm_new: (a: number, b: number, c: number, d: number, e: number) => void;
  readonly eulermapsystemwasm_getPeriodicOrbits: (a: number, b: number) => void;
  readonly eulermapsystemwasm_trackTrajectory: (a: number, b: number, c: number, d: number) => void;
  readonly eulermapsystemwasm_getCurrentPoint: (a: number, b: number) => void;
  readonly eulermapsystemwasm_getTrajectory: (a: number, b: number, c: number, d: number) => void;
  readonly eulermapsystemwasm_step: (a: number) => number;
  readonly eulermapsystemwasm_reset: (a: number) => void;
  readonly eulermapsystemwasm_getCurrentIteration: (a: number) => number;
  readonly __wbg_bdesimulatorwasm_free: (a: number, b: number) => void;
  readonly bdesimulatorwasm_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
  readonly bdesimulatorwasm_step: (a: number, b: number) => number;
  readonly bdesimulatorwasm_get_points: (a: number) => number;
  readonly bdesimulatorwasm_reparameterize: (a: number) => void;
  readonly bdesimulatorwasm_has_self_intersection: (a: number, b: number) => number;
  readonly bdesimulatorwasm_get_fold_indices: (a: number, b: number) => number;
  readonly __wbg_bdesimulatoruserdefinedwasm_free: (a: number, b: number) => void;
  readonly bdesimulatoruserdefinedwasm_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => void;
  readonly bdesimulatoruserdefinedwasm_step: (a: number, b: number) => number;
  readonly bdesimulatoruserdefinedwasm_get_points: (a: number) => number;
  readonly bdesimulatoruserdefinedwasm_reparameterize: (a: number) => void;
  readonly bdesimulatoruserdefinedwasm_has_self_intersection: (a: number, b: number) => number;
  readonly bdesimulatoruserdefinedwasm_get_fold_indices: (a: number, b: number) => number;
  readonly boundary_map_duffing_ode: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => number;
  readonly evaluate_user_defined_ode: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
  readonly boundary_map_user_defined_ode: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => void;
  readonly eulermapsystemwasm_getTotalIterations: (a: number) => number;
  readonly eulermapsystemwasm_getOrbitCount: (a: number) => number;
  readonly __wbg_extendedpoint_free: (a: number, b: number) => void;
  readonly __wbg_get_extendedpoint_x: (a: number) => number;
  readonly __wbg_set_extendedpoint_x: (a: number, b: number) => void;
  readonly __wbg_get_extendedpoint_y: (a: number) => number;
  readonly __wbg_set_extendedpoint_y: (a: number, b: number) => void;
  readonly __wbg_get_extendedpoint_nx: (a: number) => number;
  readonly __wbg_set_extendedpoint_nx: (a: number, b: number) => void;
  readonly __wbg_get_extendedpoint_ny: (a: number) => number;
  readonly __wbg_set_extendedpoint_ny: (a: number, b: number) => void;
  readonly __wbg_boundaryhenonsystemwasm_free: (a: number, b: number) => void;
  readonly boundaryhenonsystemwasm_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
  readonly boundaryhenonsystemwasm_getPeriodicOrbits: (a: number, b: number) => void;
  readonly boundaryhenonsystemwasm_trackTrajectory: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly boundaryhenonsystemwasm_getCurrentPoint: (a: number, b: number) => void;
  readonly boundaryhenonsystemwasm_getTrajectory: (a: number, b: number, c: number, d: number) => void;
  readonly boundaryhenonsystemwasm_step: (a: number) => number;
  readonly boundaryhenonsystemwasm_reset: (a: number) => void;
  readonly boundaryhenonsystemwasm_getTotalIterations: (a: number) => number;
  readonly boundaryhenonsystemwasm_getCurrentIteration: (a: number) => number;
  readonly boundaryhenonsystemwasm_getOrbitCount: (a: number) => number;
  readonly boundaryhenonsystemwasm_getEpsilon: (a: number) => number;
  readonly __wbg_boundaryuserdefinedsystemwasm_free: (a: number, b: number) => void;
  readonly boundaryuserdefinedsystemwasm_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number) => void;
  readonly boundaryuserdefinedsystemwasm_getPeriodicOrbits: (a: number, b: number) => void;
  readonly boundaryuserdefinedsystemwasm_getOrbitCount: (a: number) => number;
  readonly boundaryuserdefinedsystemwasm_getEpsilon: (a: number) => number;
  readonly boundary_map_user_defined: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => void;
  readonly parameterSweep: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => void;
  readonly parameterSweepGeneric: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number, o: number, p: number, q: number, r: number, s: number) => void;
  readonly parameterSweepCsv: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => void;
  readonly compute_duffing_manifold_simple: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
  readonly __wbg_duffingparams_free: (a: number, b: number) => void;
  readonly __wbg_get_duffingparams_a: (a: number) => number;
  readonly __wbg_set_duffingparams_a: (a: number, b: number) => void;
  readonly __wbg_get_duffingparams_b: (a: number) => number;
  readonly __wbg_set_duffingparams_b: (a: number, b: number) => void;
  readonly __wbg_get_duffingparams_epsilon: (a: number) => number;
  readonly __wbg_set_duffingparams_epsilon: (a: number, b: number) => void;
  readonly __wbg_ulamcomputer_free: (a: number, b: number) => void;
  readonly ulamcomputer_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => void;
  readonly ulamcomputer_get_grid_boxes: (a: number) => number;
  readonly ulamcomputer_get_transitions: (a: number, b: number) => number;
  readonly ulamcomputer_get_invariant_measure: (a: number) => number;
  readonly ulamcomputer_get_left_eigenvector: (a: number) => number;
  readonly ulamcomputer_get_epsilon: (a: number) => number;
  readonly ulamcomputer_get_grid_step: (a: number) => number;
  readonly ulamcomputer_get_box_index: (a: number, b: number, c: number) => number;
  readonly ulamcomputer_get_intersecting_boxes: (a: number, b: number, c: number) => number;
  readonly ulamcomputer_get_dimensions: (a: number) => number;
  readonly __wbg_ulamcomputeruserdefined_free: (a: number, b: number) => void;
  readonly ulamcomputeruserdefined_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number) => void;
  readonly ulamcomputeruserdefined_get_grid_boxes: (a: number) => number;
  readonly ulamcomputeruserdefined_get_transitions: (a: number, b: number) => number;
  readonly ulamcomputeruserdefined_get_invariant_measure: (a: number) => number;
  readonly ulamcomputeruserdefined_get_left_eigenvector: (a: number) => number;
  readonly ulamcomputeruserdefined_get_epsilon: (a: number) => number;
  readonly ulamcomputeruserdefined_get_grid_step: (a: number) => number;
  readonly ulamcomputeruserdefined_get_box_index: (a: number, b: number, c: number) => number;
  readonly ulamcomputeruserdefined_get_intersecting_boxes: (a: number, b: number, c: number) => number;
  readonly ulamcomputeruserdefined_get_dimensions: (a: number) => number;
  readonly __wbg_ulamcomputercontinuous_free: (a: number, b: number) => void;
  readonly ulamcomputercontinuous_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => void;
  readonly ulamcomputercontinuous_get_grid_boxes: (a: number) => number;
  readonly ulamcomputercontinuous_get_transitions: (a: number, b: number) => number;
  readonly ulamcomputercontinuous_get_invariant_measure: (a: number) => number;
  readonly ulamcomputercontinuous_get_left_eigenvector: (a: number) => number;
  readonly ulamcomputercontinuous_get_epsilon: (a: number) => number;
  readonly ulamcomputercontinuous_get_grid_step: (a: number) => number;
  readonly ulamcomputercontinuous_get_dimensions: (a: number) => number;
  readonly ulamcomputercontinuous_get_box_index: (a: number, b: number, c: number) => number;
  readonly __wbg_ulamcomputercontinuoususerdefined_free: (a: number, b: number) => void;
  readonly ulamcomputercontinuoususerdefined_new: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number, l: number, m: number, n: number) => void;
  readonly ulamcomputercontinuoususerdefined_get_grid_boxes: (a: number) => number;
  readonly ulamcomputercontinuoususerdefined_get_transitions: (a: number, b: number) => number;
  readonly ulamcomputercontinuoususerdefined_get_invariant_measure: (a: number) => number;
  readonly ulamcomputercontinuoususerdefined_get_left_eigenvector: (a: number) => number;
  readonly ulamcomputercontinuoususerdefined_get_epsilon: (a: number) => number;
  readonly ulamcomputercontinuoususerdefined_get_grid_step: (a: number) => number;
  readonly ulamcomputercontinuoususerdefined_get_dimensions: (a: number) => number;
  readonly ulamcomputercontinuoususerdefined_get_box_index: (a: number, b: number, c: number) => number;
  readonly __wbg_duffingsystemwasm_free: (a: number, b: number) => void;
  readonly duffingsystemwasm_new: (a: number, b: number, c: number, d: number) => void;
  readonly duffingsystemwasm_getPeriodicOrbits: (a: number, b: number) => void;
  readonly duffingsystemwasm_trackTrajectory: (a: number, b: number, c: number, d: number) => void;
  readonly duffingsystemwasm_getCurrentPoint: (a: number, b: number) => void;
  readonly duffingsystemwasm_getTrajectory: (a: number, b: number, c: number, d: number) => void;
  readonly duffingsystemwasm_step: (a: number) => number;
  readonly duffingsystemwasm_reset: (a: number) => void;
  readonly duffingsystemwasm_getTotalIterations: (a: number) => number;
  readonly duffingsystemwasm_getCurrentIteration: (a: number) => number;
  readonly duffingsystemwasm_getOrbitCount: (a: number) => number;
  readonly __wbg_videoconfig_free: (a: number, b: number) => void;
  readonly __wbg_get_videoconfig_width: (a: number) => number;
  readonly __wbg_set_videoconfig_width: (a: number, b: number) => void;
  readonly __wbg_get_videoconfig_height: (a: number) => number;
  readonly __wbg_set_videoconfig_height: (a: number, b: number) => void;
  readonly __wbg_get_videoconfig_fps: (a: number) => number;
  readonly __wbg_set_videoconfig_fps: (a: number, b: number) => void;
  readonly __wbg_get_videoconfig_crf: (a: number) => number;
  readonly __wbg_set_videoconfig_crf: (a: number, b: number) => void;
  readonly videoconfig_new: (a: number, b: number, c: number, d: number) => number;
  readonly videoconfig_default_config: () => number;
  readonly __wbg_videorecorder_free: (a: number, b: number) => void;
  readonly videorecorder_new: () => number;
  readonly videorecorder_start_recording: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => number;
  readonly videorecorder_add_frame: (a: number, b: number) => number;
  readonly videorecorder_get_frame_count: (a: number) => number;
  readonly videorecorder_get_status: (a: number) => number;
  readonly videorecorder_start_encoding: (a: number) => void;
  readonly videorecorder_finish_encoding: (a: number) => void;
  readonly videorecorder_set_error: (a: number) => void;
  readonly videorecorder_reset: (a: number) => void;
  readonly videorecorder_generate_filename: (a: number, b: number) => void;
  readonly videorecorder_get_config: (a: number) => number;
  readonly videorecorder_set_config: (a: number, b: number) => void;
  readonly videorecorder_get_expected_duration_secs: (a: number) => number;
  readonly videorecorder_is_recording: (a: number) => number;
  readonly videorecorder_is_encoding: (a: number) => number;
  readonly videorecorder_get_overlay_text: (a: number, b: number, c: number) => void;
  readonly compute_hausdorff_distance_between_manifolds: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly compute_bifurcation_hausdorff: (a: number, b: number, c: number, d: number, e: number, f: number) => void;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;
/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
