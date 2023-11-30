import { ReactElement } from "react";

import CapacityStatus from "@/components/status/CapacityStatus";
import DatasetStatus from "@/components/status/DatasetStatus.tsx";
import RISStatus from "@/components/status/RISStatus";
import RslStatus from "@/components/status/RslStatus";
import RtStatus from "@/components/status/RtStatus";
import StatusOverview from "@/components/status/overview/StatusOverview.tsx";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs.tsx";

function StatusMain(): ReactElement {
  return (
    <div className="grow overflow-y-auto p-3">
      <Tabs defaultValue="overview" className="">
        <TabsList>
          <TabsTrigger value="overview">Übersicht</TabsTrigger>
          <TabsTrigger value="dataset">Datensätze</TabsTrigger>
          <TabsTrigger value="rt">
            Echtzeitmeldungen &amp; Wagenreihungen
          </TabsTrigger>
          <TabsTrigger value="capacity">Kapazitäten</TabsTrigger>
          <TabsTrigger value="forecast">RSL-Vorhersage</TabsTrigger>
        </TabsList>
        <TabsContent value="overview">
          <StatusOverview />
        </TabsContent>
        <TabsContent value="dataset">
          <DatasetStatus />
        </TabsContent>
        <TabsContent value="rt">
          <RISStatus />
          <RtStatus />
        </TabsContent>
        <TabsContent value="capacity">
          <CapacityStatus />
        </TabsContent>
        <TabsContent value="forecast">
          <RslStatus />
        </TabsContent>
      </Tabs>
    </div>
  );
}

export default StatusMain;
