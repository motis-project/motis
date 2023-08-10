import { useAtom } from "jotai";
import { useCallback, useState } from "react";

import { hasSimResultsAtom } from "@/data/simulation";

import SimResultsPanel from "@/components/sim/SimResultsPanel";
import MeasurePanel from "@/components/sim/measures/MeasurePanel";

import { cn } from "@/lib/utils";

type TabId = "measures" | "results";

interface TabProps {
  id: TabId;
  label: string;
  selectedTab: TabId;
  setSelectedTab: (id: TabId) => void;
  disabled?: boolean;
}

function Tab({
  id,
  label,
  selectedTab,
  setSelectedTab,
  disabled,
}: TabProps): JSX.Element {
  return (
    <button
      className={cn(
        "w-full py-2.5 text-sm leading-5 font-medium rounded-lg",
        "focus:outline-none focus:ring-2 ring-db-red-800",
        selectedTab === id
          ? "bg-db-red-500 text-white shadow"
          : disabled
          ? "text-db-cool-gray-200"
          : "text-db-cool-gray-100 hover:bg-white/[0.2] hover:text-white",
      )}
      onClick={() => setSelectedTab(id)}
      disabled={disabled}
    >
      {label}
    </button>
  );
}

function SimPanel(): JSX.Element {
  const [selectedTab, setSelectedTab] = useState<TabId>("measures");
  const [hasSimResults] = useAtom(hasSimResultsAtom);

  const onSimulationFinished = useCallback(() => {
    setSelectedTab("results");
  }, [setSelectedTab]);

  return (
    <div className="flex flex-col w-full h-full">
      <div className="flex p-1 space-x-1 bg-db-cool-gray-400 rounded-xl mb-3">
        <Tab
          id="measures"
          label="Maßnahmen"
          selectedTab={selectedTab}
          setSelectedTab={setSelectedTab}
        />
        <Tab
          id="results"
          label="Ergebnisse"
          selectedTab={selectedTab}
          setSelectedTab={setSelectedTab}
          disabled={!hasSimResults}
        />
      </div>
      <div
        className={cn(
          selectedTab === "measures" ? "block" : "hidden",
          "grow overflow-y-auto",
        )}
      >
        <MeasurePanel onSimulationFinished={onSimulationFinished} />
      </div>
      <div
        className={cn(
          selectedTab === "results" ? "block" : "hidden",
          "grow overflow-y-auto",
        )}
      >
        <SimResultsPanel />
      </div>
    </div>
  );
}

export default SimPanel;
