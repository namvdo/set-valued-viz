const MIS_SUPPORT_THRESHOLD = 1e-10;
const MIS_FILTER_SUBDIVISIONS = 64;
const MIS_FILTER_POINTS_PER_BOX = 64;

let wasmPromise = null;
let cachedUlamComputer = null;

const ensureWasm = async () => {
  if (!wasmPromise) {
    wasmPromise = import('../pkg/henon_periodic_orbits.js').then(async (mod) => {
      await mod.default();
      return mod;
    });
  }
  return wasmPromise;
};

const cleanupCachedUlamComputer = () => {
  if (cachedUlamComputer && typeof cachedUlamComputer.free === 'function') {
    cachedUlamComputer.free();
  }
  cachedUlamComputer = null;
};

const getSupportIndex = (x, y, support) => {
  if (!support) return -1;
  const { xMin, xMax, yMin, yMax, subdivisions } = support;
  if (x < xMin || x > xMax || y < yMin || y > yMax) return -1;

  const dx = (xMax - xMin) / subdivisions;
  const dy = (yMax - yMin) / subdivisions;
  if (!Number.isFinite(dx) || !Number.isFinite(dy) || dx <= 0 || dy <= 0) {
    return -1;
  }

  let ix = Math.floor((x - xMin) / dx);
  let iy = Math.floor((y - yMin) / dy);
  if (ix >= subdivisions) ix -= 1;
  if (iy >= subdivisions) iy -= 1;
  if (ix < 0 || iy < 0) return -1;
  return iy * subdivisions + ix;
};

const isSupportedPoint = (x, y, support) => {
  if (!support) return true;
  const idx = getSupportIndex(x, y, support);
  if (idx < 0) return false;
  return (support.invariantMeasure?.[idx] ?? 0) > support.threshold;
};

const filterOrbitsBySupport = (orbits, support) => {
  return (orbits || []).filter((orbit) =>
    (orbit.points || []).every(([x, y]) => isSupportedPoint(x, y, support))
  );
};

const computePeriodic = async (payload) => {
  const wasm = await ensureWasm();
  const { dynamicSystem, params, viewRange, periodicSearchSettings } = payload;

  if (dynamicSystem === 'custom' || dynamicSystem === 'custom_ode') {
    return { orbits: [], support: null };
  }

  let system = null;
  let supportComputer = null;

  try {
    if (dynamicSystem === 'duffing') {
      system = new wasm.DuffingSystemWasm(params.a, params.b, params.maxPeriod);
    } else if (dynamicSystem === 'duffing_ode') {
      system = new wasm.EulerMapSystemWasm(
        params.delta,
        params.h,
        params.epsilon,
        params.maxPeriod
      );
    } else {
      system = new wasm.BoundaryHenonSystemWasm(
        params.a,
        params.b,
        params.epsilon,
        params.maxPeriod,
        viewRange.xMin,
        viewRange.xMax,
        viewRange.yMin,
        viewRange.yMax,
        periodicSearchSettings.gridSize,
        periodicSearchSettings.thetaGridSize,
        periodicSearchSettings.residualThreshold
      );
    }

    let orbits = system.getPeriodicOrbits();
    let support = null;

    if (dynamicSystem === 'henon') {
      supportComputer = new wasm.UlamComputer(
        params.a,
        params.b,
        MIS_FILTER_SUBDIVISIONS,
        MIS_FILTER_POINTS_PER_BOX,
        params.epsilon,
        viewRange.xMin,
        viewRange.xMax,
        viewRange.yMin,
        viewRange.yMax
      );

      support = {
        invariantMeasure: supportComputer.get_invariant_measure(),
        subdivisions: MIS_FILTER_SUBDIVISIONS,
        xMin: viewRange.xMin,
        xMax: viewRange.xMax,
        yMin: viewRange.yMin,
        yMax: viewRange.yMax,
        threshold: MIS_SUPPORT_THRESHOLD
      };

      orbits = filterOrbitsBySupport(orbits, support);
    }

    return { orbits, support };
  } finally {
    if (supportComputer && typeof supportComputer.free === 'function') {
      supportComputer.free();
    }
    if (system && typeof system.free === 'function') {
      system.free();
    }
  }
};

const computeManifolds = async (payload) => {
  const wasm = await ensureWasm();
  const {
    dynamicSystem,
    params,
    viewRange,
    periodicOrbits,
    customEquations,
    customParams,
    showStableManifold,
    showUnstableManifold,
    intersectionThreshold
  } = payload;

  if (dynamicSystem === 'duffing') {
    const result = wasm.compute_duffing_manifold_simple(
      params.a,
      params.b,
      params.epsilon,
      viewRange.xMin,
      viewRange.xMax,
      viewRange.yMin,
      viewRange.yMax
    );
    return {
      manifolds: result.manifolds || [],
      stableManifolds: [],
      fixedPoints: result.fixed_points || [],
      intersections: []
    };
  }

  if (dynamicSystem === 'custom') {
    if ((periodicOrbits || []).length > 0) {
      if (showStableManifold || showUnstableManifold) {
        const result = wasm.compute_stable_and_unstable_manifolds_user_defined(
          customEquations.xEq,
          customEquations.yEq,
          customParams,
          params.epsilon,
          viewRange.xMin,
          viewRange.xMax,
          viewRange.yMin,
          viewRange.yMax,
          periodicOrbits,
          intersectionThreshold
        );
        return {
          manifolds: result.unstable_manifolds || [],
          stableManifolds: result.stable_manifolds || [],
          fixedPoints: result.fixed_points || [],
          intersections: result.intersections || []
        };
      }
      return {
        manifolds: [],
        stableManifolds: [],
        fixedPoints: [],
        intersections: []
      };
    }

    const result = wasm.compute_user_defined_manifold(
      customEquations.xEq,
      customEquations.yEq,
      customParams,
      params.epsilon,
      viewRange.xMin,
      viewRange.xMax,
      viewRange.yMin,
      viewRange.yMax
    );

    return {
      manifolds: result.manifolds || [],
      stableManifolds: [],
      fixedPoints: result.fixed_points || [],
      intersections: []
    };
  }

  if ((periodicOrbits || []).length > 0) {
    if (showStableManifold || showUnstableManifold) {
      const result = wasm.compute_stable_and_unstable_manifolds(
        params.a,
        params.b,
        params.epsilon,
        viewRange.xMin,
        viewRange.xMax,
        viewRange.yMin,
        viewRange.yMax,
        periodicOrbits,
        intersectionThreshold
      );
      return {
        manifolds: result.unstable_manifolds || [],
        stableManifolds: result.stable_manifolds || [],
        fixedPoints: result.fixed_points || [],
        intersections: result.intersections || []
      };
    }
    return {
      manifolds: [],
      stableManifolds: [],
      fixedPoints: [],
      intersections: []
    };
  }

  const result = wasm.compute_manifold_simple(
    params.a,
    params.b,
    params.epsilon,
    viewRange.xMin,
    viewRange.xMax,
    viewRange.yMin,
    viewRange.yMax
  );

  return {
    manifolds: result.manifolds || [],
    stableManifolds: [],
    fixedPoints: result.fixed_points || [],
    intersections: []
  };
};

const buildUlamComputer = (wasm, payload) => {
  const {
    dynamicSystem,
    params,
    viewRange,
    ulam,
    customEquations,
    customParams
  } = payload;

  if (dynamicSystem === 'custom') {
    return new wasm.UlamComputerUserDefined(
      customEquations.xEq,
      customEquations.yEq,
      customParams,
      ulam.subdivisions,
      ulam.pointsPerBox,
      ulam.epsilon,
      viewRange.xMin,
      viewRange.xMax,
      viewRange.yMin,
      viewRange.yMax
    );
  }

  if (dynamicSystem === 'custom_ode') {
    const capitalT = Math.max(params.h * 10, 0.5);
    return new wasm.UlamComputerContinuousUserDefined(
      customEquations.xEq,
      customEquations.yEq,
      customParams,
      capitalT,
      ulam.subdivisions,
      ulam.pointsPerBox,
      ulam.epsilon,
      viewRange.xMin,
      viewRange.xMax,
      viewRange.yMin,
      viewRange.yMax
    );
  }

  if (dynamicSystem === 'duffing_ode') {
    const capitalT = Math.max(params.h * 10, 0.5);
    return new wasm.UlamComputerContinuous(
      params.delta,
      capitalT,
      ulam.subdivisions,
      ulam.pointsPerBox,
      ulam.epsilon,
      viewRange.xMin,
      viewRange.xMax,
      viewRange.yMin,
      viewRange.yMax
    );
  }

  return new wasm.UlamComputer(
    params.a,
    params.b,
    ulam.subdivisions,
    ulam.pointsPerBox,
    ulam.epsilon,
    viewRange.xMin,
    viewRange.xMax,
    viewRange.yMin,
    viewRange.yMax
  );
};

const computeUlam = async (payload) => {
  const wasm = await ensureWasm();
  cleanupCachedUlamComputer();
  cachedUlamComputer = buildUlamComputer(wasm, payload);

  const boxes = cachedUlamComputer.get_grid_boxes();
  const invariantMeasure = cachedUlamComputer.get_invariant_measure();
  const leftEigenvector = cachedUlamComputer.get_left_eigenvector();

  let currentBoxIndex = -1;
  if (payload.currentPoint) {
    currentBoxIndex = cachedUlamComputer.get_box_index(
      payload.currentPoint.x,
      payload.currentPoint.y
    );
  }

  return {
    boxes,
    invariantMeasure,
    leftEigenvector,
    currentBoxIndex
  };
};

const getUlamTransitions = async (payload) => {
  if (!cachedUlamComputer) {
    return [];
  }
  return cachedUlamComputer.get_transitions(payload.index) || [];
};

self.onmessage = async (event) => {
  const { id, kind, payload } = event.data || {};
  if (!kind) return;

  try {
    let result = null;
    if (kind === 'computePeriodic') {
      result = await computePeriodic(payload);
    } else if (kind === 'computeManifolds') {
      result = await computeManifolds(payload);
    } else if (kind === 'computeUlam') {
      result = await computeUlam(payload);
    } else if (kind === 'getUlamTransitions') {
      result = await getUlamTransitions(payload);
    } else {
      throw new Error(`Unknown worker task: ${kind}`);
    }

    self.postMessage({ id, ok: true, kind, result });
  } catch (err) {
    self.postMessage({
      id,
      ok: false,
      kind,
      error: err instanceof Error ? err.message : String(err)
    });
  }
};
