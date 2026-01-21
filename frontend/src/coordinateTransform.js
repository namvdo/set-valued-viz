// src/visualization/coordinateTransform.js

export class CoordinateTransform {
  constructor(mathBounds, viewportSize) {
    this.xMin = mathBounds.xRange[0];
    this.xMax = mathBounds.xRange[1];
    this.yMin = mathBounds.yRange[0];
    this.yMax = mathBounds.yRange[1];

    this.viewportWidth = viewportSize.width;
    this.viewportHeight = viewportSize.height;

    const xScale = this.viewportWidth / (this.xMax - this.xMin);
    const yScale = this.viewportHeight / (this.yMax - this.yMin);
    
    this.scale = Math.min(xScale, yScale) * 0.9; // 0.9 for padding

    this.xOffset = -((this.xMax + this.xMin) / 2) * this.scale;
    this.yOffset = -((this.yMax + this.yMin) / 2) * this.scale;
  }

  toThreeJS(mathX, mathY) {
    return [
      mathX * this.scale + this.xOffset,
      mathY * this.scale + this.yOffset,
      0
    ];
  }

  toMath(threeX, threeY) {
    return [
      (threeX - this.xOffset) / this.scale,
      (threeY - this.yOffset) / this.scale
    ];
  }

  toThreeJSBatch(mathPoints) {
    return mathPoints.map(([x, y]) => this.toThreeJS(x, y));
  }
}