import { PrimitiveAtom, useAtom, useSetAtom } from "jotai";
import { focusAtom } from "jotai-optics";
import { ReactNode, useMemo, useState } from "react";

import { MeasureUnion } from "@/data/measures";
import { mostRecentlySelectedTripAtom } from "@/data/selectedTrip";
import { showLegacyMeasureTypesAtom } from "@/data/settings";

import RtCancelMeasureEditor from "@/components/sim/measures/RtCancelMeasureEditor";
import RtUpdateMeasureEditor from "@/components/sim/measures/RtUpdateMeasureEditor";
import SharedDataEditor from "@/components/sim/measures/SharedDataEditor";
import TripLoadInfoMeasureEditor from "@/components/sim/measures/TripLoadInfoMeasureEditor";
import TripLoadRecommendationMeasureEditor from "@/components/sim/measures/TripLoadRecommendationMeasureEditor";
import TripRecommendationMeasureEditor from "@/components/sim/measures/TripRecommendationMeasureEditor";
import UpdateCapacityMeasureEditor from "@/components/sim/measures/UpdateCapacityMeasureEditor";
import ModalDialog from "@/components/util/ModalDialog";

export interface MeasureEditorProps {
  measureAtom: PrimitiveAtom<MeasureUnion>;
  deleteMeasure: (measureAtom: PrimitiveAtom<MeasureUnion>) => void;
  closeEditor: () => void;
}

function MeasureEditor({
  measureAtom,
  deleteMeasure,
  closeEditor,
}: MeasureEditorProps): JSX.Element {
  const typeAtom = useMemo(
    () => focusAtom(measureAtom, (optic) => optic.prop("type")),
    [measureAtom],
  );
  const [measureType] = useAtom(typeAtom);
  const setMeasure = useSetAtom(measureAtom);
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
        />,
      );
    case "TripRecommendationMeasure":
      return measureEditor(
        <TripRecommendationMeasureEditor
          measureAtom={measureAtom}
          closeEditor={closeEditor}
          key={measureAtom.toString()}
        />,
      );
    case "TripLoadRecommendationMeasure":
      return measureEditor(
        <TripLoadRecommendationMeasureEditor
          measureAtom={measureAtom}
          closeEditor={closeEditor}
          key={measureAtom.toString()}
        />,
      );
    case "RtUpdateMeasure":
      return measureEditor(
        <RtUpdateMeasureEditor
          measureAtom={measureAtom}
          closeEditor={closeEditor}
          key={measureAtom.toString()}
        />,
      );
    case "RtCancelMeasure":
      return measureEditor(
        <RtCancelMeasureEditor
          measureAtom={measureAtom}
          closeEditor={closeEditor}
          deleteMeasure={deleteMeasure}
          key={measureAtom.toString()}
        />,
      );
    case "UpdateCapacitiesMeasure":
      return measureEditor(
        <UpdateCapacityMeasureEditor
          measureAtom={measureAtom}
          closeEditor={closeEditor}
          deleteMeasure={deleteMeasure}
          key={measureAtom.toString()}
        />,
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
  const setMeasure = useSetAtom(measureAtom);
  const [selectedTrip] = useAtom(mostRecentlySelectedTripAtom);
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
        data: { trip: selectedTrip, ribasis: undefined },
      };
    });
  };

  const setRtCancel = () => {
    setMeasure((m) => {
      return {
        type: "RtCancelMeasure",
        shared: m.shared,
        data: {
          trip: selectedTrip,
          original_ribasis: undefined,
          canceled_stops: [],
          allow_reroute: true,
        },
      };
    });
  };

  const setUpdateCapacity = () => {
    setMeasure((m) => {
      return {
        type: "UpdateCapacitiesMeasure",
        shared: m.shared,
        data: {
          trip: selectedTrip,
          seats: 0,
        },
      };
    });
  };

  return (
    <div>
      <div className="text-xl">Neue Maßnahme hinzufügen</div>
      <div className="flex flex-col gap-3 py-3">
        <div className="text-lg">Nachfrageeinflussende Maßnahmen</div>
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
        <div className="text-lg">Angebotsbeeinflussende Maßnahmen</div>
        <MeasureTypeOption title="Ausfall/Teilausfall" onClick={setRtCancel}>
          Ausfall aller oder einzelner Halte eines Zuges simulieren
        </MeasureTypeOption>
        <MeasureTypeOption title="Echtzeitupdate" onClick={setRtUpdate}>
          Beliebige Änderungen am Zugverlauf (Verspätungen, Umleitungen,
          Gleisänderungen, Ausfälle) simulieren
        </MeasureTypeOption>
        <MeasureTypeOption
          title="Kapazitätsänderung"
          onClick={setUpdateCapacity}
        >
          Änderung der Kapazität eines Zuges simulieren
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
