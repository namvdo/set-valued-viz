import React from 'react';
import { SystemPicker } from '../sidebar/SystemPicker';
import { EquationDisplay } from '../sidebar/EquationDisplay';
import { ParametersPanel } from '../sidebar/ParametersPanel';
import { ManifoldsPanel } from '../sidebar/ManifoldsPanel';
import { VisualizationPanel } from '../sidebar/VisualizationPanel';
import { StartingPoint } from '../sidebar/StartingPoint';
import { PeriodicOrbitsPanel } from '../sidebar/PeriodicOrbitsPanel';
import { UlamPanel } from '../sidebar/UlamPanel';
import { AnimationPanel } from '../sidebar/AnimationPanel';
import { InfoStrip } from './InfoStrip';
import { ControlsBar } from './ControlsBar';

export const Sidebar = (props) => {
  const { ORBIT_COLORS } = props;

  return (
    <div className="sidebar">
      <div className="app-name">
        <div className="app-name-mark">∿</div>
        SVDSOULU
      </div>

      <div className="sidebar-scroll">
        <SystemPicker
          type={props.type}
          setType={props.setType}
          systemId={props.dynamicSystem}
          setSystemId={props.setDynamicSystem}
          systems={props.SYSTEMS}
        />

        <EquationDisplay
          systemId={props.dynamicSystem}
          customEquations={props.customEquations}
          setCustomEquations={props.setCustomEquations}
          equationError={props.equationError}
          disabled={props.manifoldState.isRunning}
        />

        <ParametersPanel
          systemId={props.dynamicSystem}
          params={props.params}
          setParams={props.setParams}
          disabled={props.manifoldState.isRunning}
          systems={props.SYSTEMS}
          applyPreset={props.applyPreset}
        />

        {props.type === 'discrete' && (
          <>
            <ManifoldsPanel
              manifoldState={props.manifoldState}
              setManifoldState={props.setManifoldState}
              ORBIT_COLORS={ORBIT_COLORS}
            />
            <PeriodicOrbitsPanel
              manifoldState={props.manifoldState}
              setManifoldState={props.setManifoldState}
              filters={props.filters}
              setFilters={props.setFilters}
              periodicState={props.periodicState}
            />
          </>
        )}

        {props.type === 'continuous' && (
          null
        )}

        {props.type === 'discrete' && (
          <>
            <StartingPoint
              type={props.type}
              startPoint={props.manifoldState.startPoint}
              updateStartPoint={props.updateStartPoint}
            />

            <AnimationPanel
              animationState={props.animationState}
              setAnimationState={props.setAnimationState}
              manifoldState={props.manifoldState}
              recordingState={props.recordingState}
              startAnimation={props.startAnimation}
              stopAnimation={props.stopAnimation}
              toggleRecording={props.toggleRecording}
            />

            <UlamPanel
              ulamState={props.ulamState}
              setUlamState={props.setUlamState}
            />
          </>
        )}

        {props.type === 'continuous' && (
          <StartingPoint
            type={props.type}
            startPoint={props.manifoldState.startPoint}
            updateStartPoint={props.updateStartPoint}
          />
        )}

      </div>

      <InfoStrip
        type={props.type}
        manifoldState={props.manifoldState}
        ulamState={props.ulamState}
        params={props.params}
        periodicState={props.periodicState}
      />

      <ControlsBar
        dynamicSystem={props.dynamicSystem}
        manifoldState={props.manifoldState}
        bdeState={props.bdeState}
        stepForwardManifold={props.stepForwardManifold}
        runToConvergenceManifold={props.runToConvergenceManifold}
        resetManifold={props.resetManifold}
        toggleBdeFlow={props.toggleBdeFlow}
        resetBdeFlow={props.resetBdeFlow}
      />
    </div>
  );
};
