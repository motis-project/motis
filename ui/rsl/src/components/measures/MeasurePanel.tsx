import { PrimitiveAtom, useAtom } from "jotai";
import { useUpdateAtom } from "jotai/utils";
import { useCallback } from "react";

import {
  MeasureUnion,
  currentEditorMeasureAtom,
  measuresAtom,
} from "@/data/measures";

import useRenderCount from "@/util/useRenderCount";

import MeasureEditor from "@/components/measures/MeasureEditor";
import MeasureList from "@/components/measures/MeasureList";

export type MeasurePanelProps = {
  onSimulationFinished: () => void;
};

function MeasurePanel({
  onSimulationFinished,
}: MeasurePanelProps): JSX.Element {
  const setMeasureAtoms = useUpdateAtom(measuresAtom);
  const [currentMeasureAtom, setCurrentMeasureAtom] = useAtom(
    currentEditorMeasureAtom
  );
  const renderCount = useRenderCount();

  const deleteMeasure = useCallback(
    (measureAtom: PrimitiveAtom<MeasureUnion>) => {
      setMeasureAtoms((prev) => prev.filter((e) => e !== measureAtom));
      setCurrentMeasureAtom((ma) => (ma === measureAtom ? null : ma));
    },
    [setMeasureAtoms, setCurrentMeasureAtom]
  );

  const closeEditor = useCallback(
    () => setCurrentMeasureAtom(null),
    [setCurrentMeasureAtom]
  );

  return (
    <div className="pb-5">
      <div>MeasurePanel Render Count: {renderCount}</div>
      {currentMeasureAtom ? (
        <MeasureEditor
          measureAtom={currentMeasureAtom}
          deleteMeasure={deleteMeasure}
          closeEditor={closeEditor}
        />
      ) : (
        <MeasureList onSimulationFinished={onSimulationFinished} />
      )}
    </div>
  );
}

export default MeasurePanel;
