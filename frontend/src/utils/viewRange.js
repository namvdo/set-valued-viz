export const RANGE_LIMIT = 10;

export const DEFAULT_VIEW_RANGE = {
  xMin: -2,
  xMax: 2,
  yMin: -1.5,
  yMax: 1.5
};

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

export const normalizeViewRange = (range, limit = RANGE_LIMIT) => {
  let xMin = Number.isFinite(range.xMin) ? range.xMin : DEFAULT_VIEW_RANGE.xMin;
  let xMax = Number.isFinite(range.xMax) ? range.xMax : DEFAULT_VIEW_RANGE.xMax;
  let yMin = Number.isFinite(range.yMin) ? range.yMin : DEFAULT_VIEW_RANGE.yMin;
  let yMax = Number.isFinite(range.yMax) ? range.yMax : DEFAULT_VIEW_RANGE.yMax;

  let loX = Math.min(xMin, xMax);
  let hiX = Math.max(xMin, xMax);
  let loY = Math.min(yMin, yMax);
  let hiY = Math.max(yMin, yMax);

  loX = clamp(loX, -limit, limit);
  hiX = clamp(hiX, -limit, limit);
  loY = clamp(loY, -limit, limit);
  hiY = clamp(hiY, -limit, limit);

  if (Math.abs(hiX - loX) < 1e-6) {
    const center = (hiX + loX) / 2;
    loX = clamp(center - 1, -limit, limit);
    hiX = clamp(center + 1, -limit, limit);
  }

  if (Math.abs(hiY - loY) < 1e-6) {
    const center = (hiY + loY) / 2;
    loY = clamp(center - 1, -limit, limit);
    hiY = clamp(center + 1, -limit, limit);
  }

  return { xMin: loX, xMax: hiX, yMin: loY, yMax: hiY };
};
