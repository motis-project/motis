import { useAtom } from "jotai";

import { PaxMonGroupStatisticsResponse } from "@/api/protocol/motis/paxmon";

import { usePaxMonGroupStatisticsQuery } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";
import { formatNumber } from "@/data/numberFormat";

import Histogram, {
  ResponsiveHistogramProps,
} from "@/components/stats/Histogram";

function GroupStatistics(): JSX.Element {
  const [universe] = useAtom(universeAtom);
  const { data } = usePaxMonGroupStatisticsQuery({
    universe,
    count_passengers: true,
  });

  if (!data) {
    return <div>Gruppenstatistiken werden geladen...</div>;
  }

  return (
    <div className="p-3 grow overflow-y-auto">
      <h1 className="text-xl font-semibold">Gruppenstatistiken</h1>
      <p>
        Reisendengruppen: {formatNumber(data.group_count)}, Reisende:{" "}
        {formatNumber(data.total_pax_count)}, Möglicherweise gestrandete
        Reisendengruppen:{" "}
        {formatNumber(data.groups_with_unreachable_destination)}
      </p>
      <p>
        Reiseketten: {formatNumber(data.total_group_route_count)} gesamt,{" "}
        {formatNumber(data.active_group_route_count)} aktive
      </p>
      <GroupHistograms data={data} />
    </div>
  );
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
        <div>Reisekettenwahrscheinlichkeit</div>
        <div className="h-96">
          <Histogram
            data={data.group_route_probabilities}
            xTickFormat={formatPercentageTick}
          />
        </div>
      </div>
      <div>
        <div>Erwartete Zielverspätung</div>
        <div className="h-96">
          <DurationHistogram data={data.expected_estimated_delay} />
        </div>
      </div>
      <div>
        <div>Minimale Zielverspätung</div>
        <div className="h-96">
          <DurationHistogram data={data.min_estimated_delay} />
        </div>
      </div>
      <div>
        <div>Maximale Zielverspätung</div>
        <div className="h-96">
          <DurationHistogram data={data.max_estimated_delay} />
        </div>
      </div>
    </div>
  );
}

function formatDurationTick(mins: number): string {
  if (mins >= 60 || mins <= -60) {
    const hrs = Math.floor(mins / 60);
    mins = Math.abs(mins % 60);
    return mins === 0 ? `${hrs}h` : `${hrs}h${mins}m`;
  }
  return `${mins}m`;
}

function formatPercentageTick(percent: number): string {
  return `${percent} %`;
}

type DurationHistogramProps = Omit<
  ResponsiveHistogramProps,
  "xTickFormat" | "xTickValues" | "xNumTicks"
>;

function DurationHistogram({
  data,
  ...rest
}: DurationHistogramProps): JSX.Element {
  const totalRange = data.max_value - data.min_value;
  const maxTickCount = 15;
  let tickStep = totalRange > 10 * 60 ? 60 : totalRange > 5 * 60 ? 30 : 10;
  while (totalRange / tickStep > maxTickCount) {
    tickStep *= 2;
  }
  const xTickValues: number[] = [0];
  for (
    let x = Math.ceil(data.min_value / 60) * 60;
    x < data.max_value;
    x += tickStep
  ) {
    if (x !== 0) {
      xTickValues.push(x);
    }
  }
  return (
    <Histogram
      data={data}
      xTickFormat={formatDurationTick}
      xTickValues={xTickValues}
      {...rest}
    />
  );
}

export default GroupStatistics;
