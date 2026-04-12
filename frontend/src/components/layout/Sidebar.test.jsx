import React from 'react';
import { render, screen } from '@testing-library/react';
import { vi } from 'vitest';

vi.mock('../sidebar/SystemPicker', () => ({ SystemPicker: () => <div data-testid="system-picker" /> }));
vi.mock('../sidebar/EquationDisplay', () => ({ EquationDisplay: () => <div data-testid="equation-display" /> }));
vi.mock('../sidebar/ParametersPanel', () => ({ ParametersPanel: () => <div data-testid="parameters-panel" /> }));
vi.mock('../sidebar/ManifoldsPanel', () => ({ ManifoldsPanel: () => <div data-testid="manifolds-panel" /> }));
vi.mock('../sidebar/VisualizationPanel', () => ({ VisualizationPanel: () => <div data-testid="visualization-panel" /> }));
vi.mock('../sidebar/StartingPoint', () => ({ StartingPoint: () => <div data-testid="starting-point" /> }));
vi.mock('../sidebar/PeriodicOrbitsPanel', () => ({ PeriodicOrbitsPanel: () => <div data-testid="periodic-orbits" /> }));
vi.mock('../sidebar/UlamPanel', () => ({ UlamPanel: () => <div data-testid="ulam-panel" /> }));
vi.mock('../sidebar/AnimationPanel', () => ({ AnimationPanel: () => <div data-testid="animation-panel" /> }));
vi.mock('../sidebar/BifurcationPanel', () => ({ BifurcationPanel: () => <div data-testid="bifurcation-panel" /> }));
vi.mock('./InfoStrip', () => ({ InfoStrip: () => <div data-testid="info-strip" /> }));
vi.mock('./ControlsBar', () => ({ ControlsBar: () => <div data-testid="controls-bar" /> }));

import { Sidebar } from './Sidebar';

const baseProps = {
  type: 'continuous',
  setType: vi.fn(),
  dynamicSystem: 'duffing_ode',
  setDynamicSystem: vi.fn(),
  SYSTEMS: { continuous: [], discrete: [] },
  customEquations: {},
  setCustomEquations: vi.fn(),
  equationError: null,
  params: {},
  setParams: vi.fn(),
  applyPreset: vi.fn(),
  customParams: [],
  setCustomParams: vi.fn(),
  paramErrors: [],
  viewRange: { xMin: -2, xMax: 2, yMin: -1.5, yMax: 1.5 },
  setViewRange: vi.fn(),
  rangeLimit: 10,
  resetViewRange: vi.fn(),
  manifoldState: {},
  setManifoldState: vi.fn(),
  ORBIT_COLORS: {},
  filters: {},
  setFilters: vi.fn(),
  periodicState: {},
  updateStartPoint: vi.fn(),
  animationState: {},
  setAnimationState: vi.fn(),
  recordingState: {},
  startAnimation: vi.fn(),
  stopAnimation: vi.fn(),
  toggleRecording: vi.fn(),
  ulamState: {},
  setUlamState: vi.fn(),
  bifurcationState: { data: [], isComputing: false, error: null, aMin: 0.1, aMax: 2.0, numSamples: 30, threshold: 0.2, intersectionCount: 0, criticalValues: [] },
  setBifurcationState: vi.fn(),
  wasmModule: null,
  bdeState: {},
  stepForwardManifold: vi.fn(),
  runToConvergenceManifold: vi.fn(),
  resetManifold: vi.fn(),
  toggleBdeFlow: vi.fn(),
  resetBdeFlow: vi.fn()
};

describe('Sidebar', () => {
  it('shows the starting point panel for continuous systems', () => {
    render(<Sidebar {...baseProps} type="continuous" dynamicSystem="duffing_ode" />);
    expect(screen.getByTestId('starting-point')).toBeInTheDocument();
  });

  it('hides the starting point panel for discrete systems', () => {
    render(<Sidebar {...baseProps} type="discrete" dynamicSystem="henon" />);
    expect(screen.queryByTestId('starting-point')).toBeNull();
  });
});
