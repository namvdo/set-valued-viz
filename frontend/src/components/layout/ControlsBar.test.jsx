import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { vi } from 'vitest';
import { ControlsBar } from './ControlsBar';

const baseProps = {
  dynamicSystem: 'henon',
  manifoldState: {
    isReady: true,
    isRunning: false
  },
  bdeState: {
    isRunning: false
  },
  stepForwardManifold: vi.fn(),
  runToConvergenceManifold: vi.fn(),
  resetManifold: vi.fn(),
  toggleBdeFlow: vi.fn(),
  resetBdeFlow: vi.fn()
};

describe('ControlsBar', () => {
  it('keeps Play enabled while running so it can stop (discrete)', () => {
    render(
      <ControlsBar
        {...baseProps}
        dynamicSystem="henon"
        manifoldState={{ ...baseProps.manifoldState, isRunning: true }}
      />
    );

    const play = screen.getByRole('button', { name: /stop|play/i });
    expect(play).not.toBeDisabled();
    fireEvent.click(play);
    expect(baseProps.runToConvergenceManifold).toHaveBeenCalled();
  });

  it('keeps Play enabled while running for continuous duffing_ode', () => {
    render(
      <ControlsBar
        {...baseProps}
        dynamicSystem="duffing_ode"
        manifoldState={{ ...baseProps.manifoldState, isRunning: true }}
      />
    );

    const play = screen.getByRole('button', { name: /stop|play/i });
    expect(play).not.toBeDisabled();
  });
});
