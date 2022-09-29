import { useAtom } from "jotai";

import { usePaxMonGroupStatisticsQuery } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";

function GroupStatistics(): JSX.Element {
  const [universe] = useAtom(universeAtom);
  const { data } = usePaxMonGroupStatisticsQuery(universe);

  if (!data) {
    return <div>Gruppenstatistiken werden geladen...</div>;
  }

  return (
    <div>
      <p>GroupStatistics</p>
      <p>Reisendengruppen: {data.group_count} </p>
      <p>
        Reiseketten: {data.total_group_route_count} gesamt,{" "}
        {data.active_group_route_count} aktive
      </p>
    </div>
  );
}

export default GroupStatistics;
