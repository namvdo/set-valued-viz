import React, { useEffect, useRef } from 'react';

const Algorithm1Viz = ({
  boundaryHistory,
  currentIteration,
  epsilon,
  nBoundaryPoints,
  isConverged,
  stepData,
  visualizationMode,
  showDetailedViz,
  isDiverged
}) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // Show instructions if no data
    if (!showDetailedViz && (!boundaryHistory || boundaryHistory.length === 0)) {
      ctx.fillStyle = '#64b5f6';
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Click "Step Forward" to start visualization', width / 2, height / 2);
      return;
    }

    // Determine what data to visualize
    let dataToVisualize = null;
    let allPoints = [];

    if (showDetailedViz && stepData) {
      // Detailed step-by-step visualization
      dataToVisualize = stepData;
      
      // Collect all points for bounds calculation
      if (stepData.mapped_points) {
        for (let i = 0; i < stepData.mapped_points.length; i += 2) {
          allPoints.push([stepData.mapped_points[i], stepData.mapped_points[i + 1]]);
        }
      }
      if (stepData.projected_points) {
        for (let i = 0; i < stepData.projected_points.length; i += 2) {
          allPoints.push([stepData.projected_points[i], stepData.projected_points[i + 1]]);
        }
      }
    } else if (boundaryHistory && boundaryHistory.length > 0) {
      // Full evolution visualization
      for (const boundary of boundaryHistory) {
        for (let i = 0; i < boundary.length; i += 2) {
          allPoints.push([boundary[i], boundary[i + 1]]);
        }
      }
    }

    if (allPoints.length === 0) return;

    // Calculate adaptive bounds with smooth scaling
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    for (const [x, y] of allPoints) {
      if (isFinite(x) && isFinite(y)) {
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
      }
    }

    // Handle edge cases
    if (!isFinite(minX) || !isFinite(maxX) || !isFinite(minY) || !isFinite(maxY)) {
      ctx.fillStyle = '#ff6b6b';
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Invalid data bounds', width / 2, height / 2);
      return;
    }

    // Add adaptive padding (15% of range, or minimum 0.5 if range is small)
    const rangeX = maxX - minX;
    const rangeY = maxY - minY;
    const paddingX = Math.max(rangeX * 0.15, 0.5);
    const paddingY = Math.max(rangeY * 0.15, 0.5);

    minX -= paddingX;
    maxX += paddingX;
    minY -= paddingY;
    maxY += paddingY;

    // Calculate axis spacing
    const finalRangeX = maxX - minX;
    const finalRangeY = maxY - minY;

    // Reserve space for axes and labels
    const marginLeft = 60;
    const marginRight = 20;
    const marginTop = 20;
    const marginBottom = 50;

    const plotWidth = width - marginLeft - marginRight;
    const plotHeight = height - marginTop - marginBottom;

    // CRITICAL: Apply EQUAL ASPECT RATIO
    // Choose the scaling that fits both dimensions
    const scaleFactorX = plotWidth / finalRangeX;
    const scaleFactorY = plotHeight / finalRangeY;
    
    // Use the SMALLER scale factor to ensure everything fits
    const scaleFactor = Math.min(scaleFactorX, scaleFactorY);
    
    // Calculate actual used dimensions
    const usedWidth = finalRangeX * scaleFactor;
    const usedHeight = finalRangeY * scaleFactor;
    
    // Center the plot in available space
    const offsetX = marginLeft + (plotWidth - usedWidth) / 2;
    const offsetY = marginTop + (plotHeight - usedHeight) / 2;

    // Coordinate transformation functions with EQUAL scaling
    const scaleX = (x) => offsetX + (x - minX) * scaleFactor;
    const scaleY = (y) => offsetY + usedHeight - (y - minY) * scaleFactor;
    const scaleRadius = (r) => r * scaleFactor;

    // Draw grid and axes
    drawGridAndAxes(ctx, width, height, minX, maxX, minY, maxY, 
                    marginLeft, marginRight, marginTop, marginBottom,
                    scaleX, scaleY);

    // Draw visualization based on mode
    if (showDetailedViz && dataToVisualize) {
      drawDetailedStepVisualization(ctx, dataToVisualize, visualizationMode, 
                                    epsilon, scaleX, scaleY, scaleRadius);
    } else if (boundaryHistory && boundaryHistory.length > 0) {
      drawBoundaryEvolution(ctx, boundaryHistory, scaleX, scaleY);
    }

    // Draw status information
    drawStatusInfo(ctx, width, height, currentIteration, nBoundaryPoints, 
                   epsilon, isConverged, isDiverged, dataToVisualize);

  }, [boundaryHistory, currentIteration, epsilon, nBoundaryPoints, 
      isConverged, stepData, visualizationMode, showDetailedViz, isDiverged]);

  return (
    <canvas 
      ref={canvasRef} 
      width={1200} 
      height={700}
      style={{ 
        width: '100%', 
        height: 'auto',
        borderRadius: '8px',
        background: '#000'
      }}
    />
  );
};

// Helper function: Draw grid and axes
function drawGridAndAxes(ctx, width, height, minX, maxX, minY, maxY,
                        marginLeft, marginRight, marginTop, marginBottom,
                        scaleX, scaleY) {
  const plotWidth = width - marginLeft - marginRight;
  const plotHeight = height - marginTop - marginBottom;

  // Calculate nice tick spacing
  const xRange = maxX - minX;
  const yRange = maxY - minY;
  const xTickSpacing = getNiceTickSpacing(xRange, 8);
  const yTickSpacing = getNiceTickSpacing(yRange, 6);

  // Draw vertical grid lines (x-axis)
  ctx.strokeStyle = '#2a3150';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#909090';
  ctx.font = '11px Arial';
  ctx.textAlign = 'center';

  let xTick = Math.ceil(minX / xTickSpacing) * xTickSpacing;
  while (xTick <= maxX) {
    const x = scaleX(xTick);
    
    // Grid line
    ctx.beginPath();
    ctx.moveTo(x, marginTop);
    ctx.lineTo(x, height - marginBottom);
    ctx.stroke();

    // Tick label
    ctx.fillText(xTick.toFixed(2), x, height - marginBottom + 20);
    
    xTick += xTickSpacing;
  }

  // Draw horizontal grid lines (y-axis)
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';

  let yTick = Math.ceil(minY / yTickSpacing) * yTickSpacing;
  while (yTick <= maxY) {
    const y = scaleY(yTick);
    
    // Grid line
    ctx.beginPath();
    ctx.moveTo(marginLeft, y);
    ctx.lineTo(width - marginRight, y);
    ctx.stroke();

    // Tick label
    ctx.fillText(yTick.toFixed(2), marginLeft - 10, y);
    
    yTick += yTickSpacing;
  }

  // Draw main axes
  ctx.strokeStyle = '#64b5f6';
  ctx.lineWidth = 2;

  // Y-axis
  ctx.beginPath();
  ctx.moveTo(marginLeft, marginTop);
  ctx.lineTo(marginLeft, height - marginBottom);
  ctx.stroke();

  // X-axis
  ctx.beginPath();
  ctx.moveTo(marginLeft, height - marginBottom);
  ctx.lineTo(width - marginRight, height - marginBottom);
  ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#64b5f6';
  ctx.font = 'bold 13px Arial';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('x', width / 2, height - marginBottom + 35);

  ctx.save();
  ctx.translate(marginLeft - 45, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('y', 0, 0);
  ctx.restore();
}

// Helper function: Calculate nice tick spacing
function getNiceTickSpacing(range, targetTicks) {
  const roughSpacing = range / targetTicks;
  const magnitude = Math.pow(10, Math.floor(Math.log10(roughSpacing)));
  const normalized = roughSpacing / magnitude;

  let niceSpacing;
  if (normalized < 1.5) niceSpacing = 1;
  else if (normalized < 3) niceSpacing = 2;
  else if (normalized < 7) niceSpacing = 5;
  else niceSpacing = 10;

  return niceSpacing * magnitude;
}

// Helper function: Draw detailed step visualization
function drawDetailedStepVisualization(ctx, stepData, mode, epsilon, scaleX, scaleY, scaleRadius) {
  const { mapped_points, projected_points, normals } = stepData;

  // Step 1: Draw noise circles (true circles now with equal aspect ratio!)
  if (mode === 'circles' || mode === 'all') {
    ctx.strokeStyle = '#ffaa00';
    ctx.fillStyle = 'rgba(255, 170, 0, 0.1)';
    ctx.lineWidth = 1.5;

    // For each mapped point, draw circle of radius epsilon
    // Since we have equal aspect ratio, we can use arc() directly
    const radiusPixels = scaleRadius(epsilon);

    for (let i = 0; i < mapped_points.length; i += 2) {
      const cx = scaleX(mapped_points[i]);
      const cy = scaleY(mapped_points[i + 1]);

      ctx.beginPath();
      ctx.arc(cx, cy, radiusPixels, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  }

  // Step 2: Draw mapped points f(zk)
  if (mode === 'mapped' || mode === 'all') {
    ctx.fillStyle = '#00ffff';
    for (let i = 0; i < mapped_points.length; i += 2) {
      const x = scaleX(mapped_points[i]);
      const y = scaleY(mapped_points[i + 1]);

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  // Step 3: Draw normal vectors from mapped point to projected point
  if (mode === 'all' && normals && normals.length > 0 && projected_points && projected_points.length > 0) {
    ctx.strokeStyle = '#00ff00';
    ctx.lineWidth = 2;

    // Draw arrow from each mapped point to its corresponding projected point
    for (let i = 0; i < mapped_points.length; i += 2) {
      const mappedX = scaleX(mapped_points[i]);
      const mappedY = scaleY(mapped_points[i + 1]);
      
      const projectedX = scaleX(projected_points[i]);
      const projectedY = scaleY(projected_points[i + 1]);

      // Draw line from mapped to projected
      ctx.beginPath();
      ctx.moveTo(mappedX, mappedY);
      ctx.lineTo(projectedX, projectedY);
      ctx.stroke();

      // Draw small arrowhead at the projected point
      const angle = Math.atan2(projectedY - mappedY, projectedX - mappedX);
      const arrowSize = 8;
      
      ctx.beginPath();
      ctx.moveTo(projectedX, projectedY);
      ctx.lineTo(
        projectedX - arrowSize * Math.cos(angle - Math.PI / 6),
        projectedY - arrowSize * Math.sin(angle - Math.PI / 6)
      );
      ctx.moveTo(projectedX, projectedY);
      ctx.lineTo(
        projectedX - arrowSize * Math.cos(angle + Math.PI / 6),
        projectedY - arrowSize * Math.sin(angle + Math.PI / 6)
      );
      ctx.stroke();
    }
  }

  // Step 4: Draw projected boundary
  if (mode === 'projected' || mode === 'all') {
    if (projected_points && projected_points.length > 0) {
      // Draw boundary curve
      ctx.strokeStyle = '#ff00ff';
      ctx.lineWidth = 2.5;
      ctx.beginPath();

      for (let i = 0; i < projected_points.length; i += 2) {
        const x = scaleX(projected_points[i]);
        const y = scaleY(projected_points[i + 1]);

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.closePath();
      ctx.stroke();

      // Fill interior
      ctx.fillStyle = 'rgba(255, 0, 255, 0.08)';
      ctx.fill();

      // Draw boundary points - these are exactly on the circle boundary
      ctx.fillStyle = '#ff00ff';
      for (let i = 0; i < projected_points.length; i += 2) {
        const x = scaleX(projected_points[i]);
        const y = scaleY(projected_points[i + 1]);

        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI);
        ctx.fill();
        
        // Add white outline to make them more visible
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }
}

// Helper function: Draw boundary evolution
function drawBoundaryEvolution(ctx, boundaryHistory, scaleX, scaleY) {
  const maxIterations = boundaryHistory.length;

  for (let iter = 0; iter < maxIterations; iter++) {
    const boundary = boundaryHistory[iter];
    if (!boundary || boundary.length === 0) continue;

    // Color gradient from blue to red
    const intensity = iter / Math.max(1, maxIterations - 1);
    const r = Math.floor(255 * Math.pow(intensity, 0.7));
    const g = Math.floor(255 * (1 - Math.abs(2 * intensity - 1)));
    const b = Math.floor(255 * Math.pow(1 - intensity, 0.7));
    const alpha = 0.3 + 0.7 * intensity;

    ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
    ctx.lineWidth = iter === maxIterations - 1 ? 3 : 1.5;

    // Draw boundary polygon
    ctx.beginPath();
    for (let i = 0; i < boundary.length; i += 2) {
      const x = scaleX(boundary[i]);
      const y = scaleY(boundary[i + 1]);

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.closePath();
    ctx.stroke();

    // Draw points for final iteration
    if (iter === maxIterations - 1) {
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      for (let i = 0; i < boundary.length; i += 2) {
        const x = scaleX(boundary[i]);
        const y = scaleY(boundary[i + 1]);
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }
}

// Helper function: Draw status information
function drawStatusInfo(ctx, width, height, currentIteration, nBoundaryPoints, 
                       epsilon, isConverged, isDiverged, stepData) {
  // Status panel background
  ctx.fillStyle = 'rgba(26, 31, 58, 0.9)';
  const statusHeight = isDiverged ? 140 : 120;
  ctx.fillRect(10, 10, 240, statusHeight);

  // Status text
  ctx.fillStyle = '#ffffff';
  ctx.font = '13px Arial';
  ctx.textAlign = 'left';
  
  ctx.fillText(`Iteration: ${currentIteration}`, 20, 30);
  ctx.fillText(`Boundary Points: ${nBoundaryPoints}`, 20, 50);
  ctx.fillText(`ε = ${epsilon.toFixed(3)}`, 20, 70);

  // Verify mathematical relationship if we have detailed data
  if (stepData && stepData.mapped_points && stepData.projected_points && stepData.normals) {
    let allCorrect = true;
    let maxError = 0;
    
    for (let i = 0; i < stepData.mapped_points.length; i += 2) {
      const mx = stepData.mapped_points[i];
      const my = stepData.mapped_points[i + 1];
      const px = stepData.projected_points[i];
      const py = stepData.projected_points[i + 1];
      const nx = stepData.normals[i];
      const ny = stepData.normals[i + 1];
      
      // Calculate expected projected point
      const expectedPx = mx + epsilon * nx;
      const expectedPy = my + epsilon * ny;
      
      // Calculate error
      const error = Math.sqrt((px - expectedPx) ** 2 + (py - expectedPy) ** 2);
      maxError = Math.max(maxError, error);
      
      if (error > 1e-6) {
        allCorrect = false;
      }
    }
    
    ctx.fillStyle = allCorrect ? '#00ff00' : '#ffaa00';
    ctx.fillText(`Max error: ${maxError.toExponential(2)}`, 20, 90);
  }

  if (isDiverged) {
    ctx.fillStyle = '#ff6b6b';
    ctx.font = 'bold 13px Arial';
    ctx.fillText('DIVERGED', 20, 110);
  } else if (isConverged) {
    ctx.fillStyle = '#00ff00';
    ctx.font = 'bold 13px Arial';
    ctx.fillText('CONVERGED', 20, 110);
  }

  // Legend (if detailed view)
  if (stepData) {
    const legendX = width - 250;
    const legendY = 10;

    ctx.fillStyle = 'rgba(26, 31, 58, 0.9)';
    ctx.fillRect(legendX, legendY, 240, 110);

    ctx.font = '12px Arial';
    ctx.fillStyle = '#00ffff';
    ctx.fillRect(legendX + 10, legendY + 10, 15, 15);
    ctx.fillStyle = '#ffffff';
    ctx.fillText('Mapped f(zₖ)', legendX + 30, legendY + 22);

    ctx.fillStyle = '#ffaa00';
    ctx.fillRect(legendX + 10, legendY + 32, 15, 15);
    ctx.fillStyle = '#ffffff';
    ctx.fillText('Noise circle Bε(f(zₖ))', legendX + 30, legendY + 44);

    ctx.fillStyle = '#00ff00';
    ctx.fillRect(legendX + 10, legendY + 54, 15, 15);
    ctx.fillStyle = '#ffffff';
    ctx.fillText('Normal vector (ε·n)', legendX + 30, legendY + 66);

    ctx.fillStyle = '#ff00ff';
    ctx.fillRect(legendX + 10, legendY + 76, 15, 15);
    ctx.fillStyle = '#ffffff';
    ctx.fillText('Boundary ∂F(z)', legendX + 30, legendY + 88);
  }
}

export default Algorithm1Viz;