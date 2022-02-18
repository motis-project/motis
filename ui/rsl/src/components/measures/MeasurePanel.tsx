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

function MeasurePanel(): JSX.Element {
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
    <div>
      <div>MeasurePanel Render Count: {renderCount}</div>
      {currentMeasureAtom ? (
        <MeasureEditor
          measureAtom={currentMeasureAtom}
          deleteMeasure={deleteMeasure}
          closeEditor={closeEditor}
        />
      ) : (
        <MeasureList />
      )}
    </div>
  );
}

export default MeasurePanel;
