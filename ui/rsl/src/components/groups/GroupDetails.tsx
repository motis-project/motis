import { UsersIcon } from "@heroicons/react/24/outline";
import { useAtom, useSetAtom } from "jotai";
import React, { ReactNode, useEffect } from "react";
import { useParams } from "react-router-dom";

import { usePaxMonGetGroupsRequest } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";
import { mostRecentlySelectedGroupAtom } from "@/data/selectedGroup";

import { formatDateTime } from "@/util/dateFormat";

import GroupRouteTree from "@/components/groups/GroupRouteTree";
import { GroupRoutes } from "@/components/groups/GroupRoutes.tsx";
import { RerouteLog } from "@/components/groups/RerouteLog.tsx";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs.tsx";

interface GroupDetailsProps {
  groupId: number;
}

function GroupDetails({ groupId }: GroupDetailsProps): ReactNode {
  const [universe] = useAtom(universeAtom);
  const { data, isPending, error } = usePaxMonGetGroupsRequest({
    universe,
    ids: [groupId],
    sources: [],
    include_reroute_log: true,
  });

  const setMostRecentlySelectedGroup = useSetAtom(
    mostRecentlySelectedGroupAtom,
  );
  useEffect(() => {
    setMostRecentlySelectedGroup(groupId);
  }, [groupId, setMostRecentlySelectedGroup]);

  if (!data) {
    if (isPending) {
      return <div>Gruppeninformationen werden geladen...</div>;
    } else {
      return (
        <div>
          Fehler beim Laden der Gruppeninformationen:{" "}
          {error instanceof Error ? error.message : `Unbekannter Fehler`}
        </div>
      );
    }
  }
  if (data.groups.length === 0) {
    return <div>Gruppe {groupId} nicht gefunden.</div>;
  }

  const group = data.groups[0];

  if (group.routes.length === 0) {
    return <div>Gruppe {groupId} ist ungültig (keine Routen).</div>;
  }

  return (
    <div>
      <div className="flex gap-10 text-xl">
        <div>
          Reisendengruppe <span>{group.source.primary_ref}</span>
          <span className="text-db-cool-gray-400">
            .{group.source.secondary_ref}
          </span>
        </div>
        <div>Interne ID: {group.id}</div>
        <div className="flex items-center gap-x-1">
          <UsersIcon
            className="h-4 w-4 text-db-cool-gray-500"
            aria-hidden="true"
          />
          {group.passenger_count}
          <span className="sr-only">Reisende</span>
        </div>
      </div>
      <div className="">
        Planmäßige Ankunftszeit:{" "}
        {formatDateTime(group.routes[0].planned_arrival_time)}
      </div>
      <Tabs defaultValue="routes" className="mt-2">
        <TabsList>
          <TabsTrigger value="routes">Routen</TabsTrigger>
          <TabsTrigger value="log">Änderungsprotokoll</TabsTrigger>
          <TabsTrigger value="tree">Verbindungsbaum</TabsTrigger>
        </TabsList>
        <TabsContent value="routes">
          <GroupRoutes group={group} />
        </TabsContent>
        <TabsContent value="log">
          <RerouteLog group={group} />
        </TabsContent>
        <TabsContent value="tree">
          <div className="py-5">
            {group.routes.length <= 10 && group.reroute_log.length <= 10 ? (
              <GroupRouteTree group={group} />
            ) : (
              <div>
                Der Verbindungsbaum ist zu groß, um angezeigt zu werden.
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}

export function GroupDetailsFromRoute() {
  const params = useParams();
  const groupId = Number.parseInt(params.groupId ?? "");
  if (!Number.isNaN(groupId)) {
    return <GroupDetails groupId={groupId} key={groupId} />;
  } else {
    return <></>;
  }
}

export default GroupDetails;
