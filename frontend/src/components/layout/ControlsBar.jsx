import React from 'react';

export const ControlsBar = ({ dynamicSystem, manifoldState, bdeState, stepForwardManifold, runToConvergenceManifold, resetManifold, toggleBdeFlow, resetBdeFlow }) => {
  return (
    <div className="ctrl-bar">
      {dynamicSystem === 'duffing_ode' ? (
        <div className="ctrl-row">
          <button 
            className="ctrl-btn primary" 
            onClick={runToConvergenceManifold} 
            disabled={!manifoldState.isReady || bdeState.isRunning}
          >
            {manifoldState.isRunning ? '⏹ Stop' : '▶ Play'}
          </button>
        </div>
      ) : (
        <>
          <div className="ctrl-row">
            <button 
              className="ctrl-btn primary" 
              onClick={runToConvergenceManifold} 
              disabled={!manifoldState.isReady || manifoldState.isRunning || bdeState.isRunning}
            >
              {manifoldState.isRunning ? '⏹ Stop' : '▶ Play trajectory'}
            </button>
            <button 
              className="ctrl-btn" 
              style={{ flex: 'none', width: '34px' }}
              onClick={stepForwardManifold} 
              disabled={!manifoldState.isReady || manifoldState.isRunning || bdeState.isRunning}
              title="Step forward"
            >
              ⏭
            </button>
          </div>
          {dynamicSystem === 'duffing_ode' && (
            <div className="ctrl-row">
              <button 
                className="ctrl-btn primary" 
                style={{ backgroundColor: bdeState.isRunning ? 'rgba(168,82,82,.07)' : '', color: bdeState.isRunning ? 'var(--red)' : '', borderColor: bdeState.isRunning ? 'var(--red)' : '' }}
                onClick={toggleBdeFlow} 
                disabled={!manifoldState.isReady || manifoldState.isRunning}
              >
                {bdeState.isRunning ? '⏹ Stop bound flow' : '▶ Boundary flow'}
              </button>
            </div>
          )}
        </>
      )}
      <button 
        className="ctrl-btn danger" 
        onClick={() => { resetManifold(); if (dynamicSystem === 'duffing_ode') resetBdeFlow(); }} 
        disabled={manifoldState.isRunning || bdeState.isRunning}
      >
        ↺ Reset all
      </button>
    </div>
  );
};
