import { PrimitiveAtom, useAtom } from "jotai";
import { focusAtom } from "jotai/optics";
import { useUpdateAtom } from "jotai/utils";
import { ReactNode, useMemo, useState } from "react";

import { MeasureUnion } from "@/data/measures";
import { selectedTripAtom } from "@/data/selectedTrip";
import { showLegacyMeasureTypesAtom } from "@/data/settings";

import RtUpdateMeasureEditor from "@/components/measures/RtUpdateMeasureEditor";
import SharedDataEditor from "@/components/measures/SharedDataEditor";
import TripLoadInfoMeasureEditor from "@/components/measures/TripLoadInfoMeasureEditor";
import TripLoadRecommendationMeasureEditor from "@/components/measures/TripLoadRecommendationMeasureEditor";
import TripRecommendationMeasureEditor from "@/components/measures/TripRecommendationMeasureEditor";
import ModalDialog from "@/components/util/ModalDialog";

export type MeasureEditorProps = {
  measureAtom: PrimitiveAtom<MeasureUnion>;
  deleteMeasure: (measureAtom: PrimitiveAtom<MeasureUnion>) => void;
  closeEditor: () => void;
};

function MeasureEditor({
  measureAtom,
  deleteMeasure,
  closeEditor,
}: MeasureEditorProps): JSX.Element {
  const typeAtom = useMemo(
    () => focusAtom(measureAtom, (optic) => optic.prop("type")),
    [measureAtom]
  );
  const [measureType] = useAtom(typeAtom);
  const setMeasure = useUpdateAtom(measureAtom);
  const [changeTypeDialogOpen, setChangeTypeDialogOpen] = useState(false);

  const onChangeTypeDialogClose = (cancel: boolean) => {
    if (!cancel) {
      setMeasure((m) => {
        return { type: "Empty", shared: m.shared };
      });
    }
    setChangeTypeDialogOpen(false);
  };

  const measureEditor = (e: JSX.Element) => (
    <div>
      <div className="flex justify-between">
        <span className="text-xl">Maßnahme bearbeiten</span>
        <button
          onClick={() => setChangeTypeDialogOpen(true)}
          className="px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white text-sm rounded"
        >
          Typ ändern
        </button>
      </div>
      <SharedDataEditor measureAtom={measureAtom} />
      {e}
      <ModalDialog
        isOpen={changeTypeDialogOpen}
        onClose={onChangeTypeDialogClose}
        title={"Maßnahmentyp ändern"}
        cancelButton={"Abbrechen"}
        okButton={"OK"}
      >
        Wirklich den Maßnahmentyp ändern? Dabei gehen alle bisherigen
        Einstellungen der Maßnahme verloren.
      </ModalDialog>
    </div>
  );

  switch (measureType) {
    case "Empty":
      return (
        <EmptyMeasureEditor
          measureAtom={measureAtom}
          deleteMeasure={deleteMeasure}
          closeEditor={closeEditor}
          key={measureAtom.toString()}
        />
      );
    case "TripLoadInfoMeasure":
      return measureEditor(
        <TripLoadInfoMeasureEditor
          measureAtom={measureAtom}
          closeEditor={closeEditor}
          key={measureAtom.toString()}
        />
      );
    case "TripRecommendationMeasure":
      return measureEditor(
        <TripRecommendationMeasureEditor
          measureAtom={measureAtom}
          closeEditor={closeEditor}
          key={measureAtom.toString()}
        />
      );
    case "TripLoadRecommendationMeasure":
      return measureEditor(
        <TripLoadRecommendationMeasureEditor
          measureAtom={measureAtom}
          closeEditor={closeEditor}
          key={measureAtom.toString()}
        />
      );
    case "RtUpdateMeasure":
      return measureEditor(
        <RtUpdateMeasureEditor
          measureAtom={measureAtom}
          closeEditor={closeEditor}
          key={measureAtom.toString()}
        />
      );
  }
}

interface MeasureTypeOptionProps {
  title: string;
  children: ReactNode;
  onClick: () => void;
}

function MeasureTypeOption({
  title,
  children,
  onClick,
}: MeasureTypeOptionProps) {
  return (
    <div
      className="group bg-white hover:bg-db-red-700 rounded-lg shadow-md px-5 py-4 cursor-pointer"
      onClick={onClick}
    >
      <div className="font-medium text-gray-900 group-hover:text-white">
        {title}
      </div>
      <div className="text-gray-500 group-hover:text-db-red-100">
        {children}
      </div>
    </div>
  );
}

function EmptyMeasureEditor({
  measureAtom,
  deleteMeasure,
}: MeasureEditorProps) {
  const setMeasure = useUpdateAtom(measureAtom);
  const [selectedTrip] = useAtom(selectedTripAtom);
  const [showLegacyMeasureTypes] = useAtom(showLegacyMeasureTypesAtom);

  const setTripLoadInfo = () => {
    setMeasure((m) => {
      return {
        type: "TripLoadInfoMeasure",
        shared: m.shared,
        data: { trip: selectedTrip, level: "Full" },
      };
    });
  };

  const setTripRecommendation = () => {
    setMeasure((m) => {
      return {
        type: "TripRecommendationMeasure",
        shared: m.shared,
        data: {
          planned_destination: undefined,
          recommended_trip: selectedTrip,
          interchange_station: undefined,
        },
      };
    });
  };

  const setTripLoadRecommendation = () => {
    setMeasure((m) => {
      return {
        type: "TripLoadRecommendationMeasure",
        shared: m.shared,
        data: {
          planned_destination: undefined,
          full_trip: { trip: selectedTrip, level: "Full" },
          recommended_trips: [{ trip: undefined, level: "Low" }],
        },
      };
    });
  };

  const setRtUpdate = () => {
    setMeasure((m) => {
      return {
        type: "RtUpdateMeasure",
        shared: m.shared,
        data: { trip: undefined, ribasis: undefined },
      };
    });
  };

  return (
    <div>
      <div>Maßnahmentyp wählen:</div>
      <div className="flex flex-col gap-3 py-3">
        <MeasureTypeOption
          title="Alternativenempfehlung mit Auslastungsinformation"
          onClick={setTripLoadRecommendation}
        >
          Empfehlung an Reisende in einem Zug oder an einer Station, statt einem
          überfüllten Zug eine weniger ausgelastete Alternative zu verwenden
        </MeasureTypeOption>
        {showLegacyMeasureTypes && (
          <>
            <MeasureTypeOption
              title="Auslastungsinformation"
              onClick={setTripLoadInfo}
            >
              Ansage oder Anzeige der erwarteten Zugauslastung
            </MeasureTypeOption>
            <MeasureTypeOption
              title="Zugempfehlung"
              onClick={setTripRecommendation}
            >
              Empfehlung an Reisende in einem Zug oder an einer Station,
              Verbindungen mit einem empfohlenen Zug zu verwenden
            </MeasureTypeOption>
          </>
        )}
        <MeasureTypeOption title="Echtzeitupdate" onClick={setRtUpdate}>
          Zugverlauf bearbeiten (Verspätungen, Umleitungen, Gleisänderungen)
        </MeasureTypeOption>
        <button
          onClick={() => deleteMeasure(measureAtom)}
          className="px-2 py-1 bg-db-red-500 hover:bg-db-red-600 text-white text-sm rounded"
        >
          Abbrechen
        </button>
      </div>
    </div>
  );
}

export default MeasureEditor;
