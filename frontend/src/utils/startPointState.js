export const applyStartPointUpdate = (prev, newStart) => ({
  ...prev,
  startPoint: newStart,
  currentPoint: newStart,
  trajectoryPoints: [],
  iteration: 0,
  hasStarted: false,
  isRunning: false
});
