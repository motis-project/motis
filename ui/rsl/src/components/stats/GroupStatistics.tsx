import { useAtom } from "jotai";

import { PaxMonGroupStatisticsResponse } from "@/api/protocol/motis/paxmon";

import { usePaxMonGroupStatisticsQuery } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";
import { formatNumber } from "@/data/numberFormat";

import Histogram from "@/components/stats/Histogram";

function GroupStatistics(): JSX.Element {
  const [universe] = useAtom(universeAtom);
  const { data } = usePaxMonGroupStatisticsQuery(universe);

  if (!data) {
    return <div>Gruppenstatistiken werden geladen...</div>;
  }

  return (
    <div className="p-3 grow overflow-y-auto">
      <h1 className="text-xl font-semibold">Gruppenstatistiken</h1>
      <p>Reisendengruppen: {formatNumber(data.group_count)} </p>
      <p>
        Reiseketten: {formatNumber(data.total_group_route_count)} gesamt,{" "}
        {formatNumber(data.active_group_route_count)} aktive
      </p>
      <GroupHistograms data={data} />
    </div>
  );
}

function formatDurationTick(mins: number): string {
  if (mins >= 60 || mins <= -60) {
    return `${Math.floor(mins / 60)}h${Math.abs(mins % 60)}m`;
  }
  return `${mins}m`;
}

type GroupHistogramsProps = {
  data: PaxMonGroupStatisticsResponse;
};

function GroupHistograms({ data }: GroupHistogramsProps): JSX.Element {
  return (
    <div className="flex flex-col gap-4 mt-4">
      <div>
        <div>Routen pro Gruppe</div>
        <div className="h-96">
          <Histogram data={data.routes_per_group} />
        </div>
      </div>
      <div>
        <div>Aktive Routen pro Gruppe</div>
        <div className="h-96">
          <Histogram data={data.active_routes_per_group} />
        </div>
      </div>
      <div>
        <div>Umleitungen pro Gruppe</div>
        <div className="h-96">
          <Histogram data={data.reroutes_per_group} />
        </div>
      </div>
      <div>
        <div>Erwartete Zielverspätung in Minuten</div>
        <div className="h-96">
          <Histogram
            data={data.expected_estimated_delay}
            xTickFormat={formatDurationTick}
          />
        </div>
      </div>
      <div>
        <div>Minimale Zielverspätung in Minuten</div>
        <div className="h-96">
          <Histogram
            data={data.min_estimated_delay}
            xTickFormat={formatDurationTick}
          />
        </div>
      </div>
      <div>
        <div>Maximale Zielverspätung in Minuten</div>
        <div className="h-96">
          <Histogram
            data={data.max_estimated_delay}
            xTickFormat={formatDurationTick}
          />
        </div>
      </div>
    </div>
  );
}

export default GroupStatistics;
