import { ReactElement } from "react";

import { usePaxMonStatusQuery } from "@/api/paxmon.ts";

import { formatNumber } from "@/data/numberFormat.ts";

function DatasetStatus(): ReactElement {
  const { data: paxmonStatus } = usePaxMonStatusQuery(0);

  return (
    <div className="py-3">
      <h2 className="text-lg font-semibold">
        Überwachte Züge und Reisende (im Hauptuniversum)
      </h2>
      {paxmonStatus && (
        <div>
          <div>
            Überwachte Reisendengruppen:{" "}
            {formatNumber(paxmonStatus.active_groups)}
          </div>
          <div>Überwachte Züge: {formatNumber(paxmonStatus.trip_count)}</div>
        </div>
      )}
    </div>
  );
}

export default DatasetStatus;

//   const { data: paxmonStatus } = usePaxMonStatusQuery(universe);
