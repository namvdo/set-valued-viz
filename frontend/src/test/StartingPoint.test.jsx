import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { vi } from 'vitest';
import { StartingPoint } from '../components/sidebar/StartingPoint';

describe('StartingPoint', () => {
  const basePoint = { x: 1.23456, y: -2.5, nx: 0.6, ny: -0.2 };

  it('renders position and velocity inputs for continuous systems', () => {
    const updateStartPoint = vi.fn();
    render(
      <StartingPoint
        type="continuous"
        startPoint={basePoint}
        updateStartPoint={updateStartPoint}
      />
    );

    expect(screen.getByLabelText('x0-position')).toHaveValue(basePoint.x.toFixed(4));
    expect(screen.getByLabelText('v0-velocity')).toHaveValue(basePoint.y.toFixed(4));
    expect(screen.queryByLabelText('nx-normal')).toBeNull();
    expect(screen.queryByLabelText('ny-normal')).toBeNull();
  });

  it('renders position and normal inputs for discrete systems', () => {
    const updateStartPoint = vi.fn();
    render(
      <StartingPoint
        type="discrete"
        startPoint={basePoint}
        updateStartPoint={updateStartPoint}
      />
    );

    expect(screen.getByLabelText('x0-position')).toHaveValue(basePoint.x.toFixed(4));
    expect(screen.getByLabelText('y0-position')).toHaveValue(basePoint.y.toFixed(4));
    expect(screen.getByLabelText('nx-normal')).toHaveValue(basePoint.nx.toFixed(4));
    expect(screen.getByLabelText('ny-normal')).toHaveValue(basePoint.ny.toFixed(4));
  });

  it('updates velocity through the y-component for continuous systems', () => {
    const updateStartPoint = vi.fn();
    render(
      <StartingPoint
        type="continuous"
        startPoint={basePoint}
        updateStartPoint={updateStartPoint}
      />
    );

    fireEvent.change(screen.getByLabelText('v0-velocity'), { target: { value: '3.125' } });
    expect(updateStartPoint).toHaveBeenCalledWith('y', 3.125);
  });
});
