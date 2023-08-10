import { SparklesIcon } from "@heroicons/react/20/solid";
import { atom, useAtom, useSetAtom } from "jotai";
import { useCallback, useEffect, useRef, useState } from "react";

import { TripId } from "@/api/protocol/motis";

import { getApiEndpoint } from "@/api/endpoint";
import { sendPaxMonDestroyUniverseRequest } from "@/api/paxmon";

import { measuresAtom } from "@/data/measures";
import { scheduleAtom, universeAtom } from "@/data/multiverse";
import { showOptimizationDebugLogAtom } from "@/data/settings";

import { cn } from "@/lib/utils";
import OptimizationWorker from "@/optimization/worker?worker";
import { WorkerRequest, WorkerUpdate } from "@/optimization/workerMessages";

export interface TripOptimizationProps {
  tripId: TripId;
  optimizationAvailable: boolean;
}

function TripOptimization({
  tripId,
  optimizationAvailable,
}: TripOptimizationProps): JSX.Element {
  const textAreaRef = useRef<HTMLTextAreaElement>(null);
  const worker = useRef<Worker | null>(null);
  const simUniverses = useRef<number[]>([]);
  const [running, setRunning] = useState(false);

  const [universe] = useAtom(universeAtom);
  const [schedule] = useAtom(scheduleAtom);
  const setMeasureAtoms = useSetAtom(measuresAtom);
  const [showOptimizationDebugLog] = useAtom(showOptimizationDebugLogAtom);

  const startOptimization = useCallback(() => {
    if (!textAreaRef.current) {
      return;
    }
    setRunning(true);
    if (!worker.current) {
      worker.current = new OptimizationWorker();
      worker.current?.postMessage({
        action: "Init",
        apiEndpoint: getApiEndpoint(),
      } as WorkerRequest);
    }
    textAreaRef.current.value = "Optimierung wird gestartet...\n";
    worker.current.onmessage = (msg) => {
      if (textAreaRef.current != null) {
        const update = msg.data as WorkerUpdate;
        switch (update.type) {
          case "Log": {
            textAreaRef.current.value += `${update.msg}\n`;
            break;
          }
          case "UniverseForked": {
            simUniverses.current.push(update.universe);
            break;
          }
          case "UniverseDestroyed": {
            simUniverses.current = simUniverses.current.filter(
              (u) => u != update.universe,
            );
            break;
          }
          case "MeasuresAdded": {
            const newMeasures = update.measures.map((mu) => atom(mu));
            setMeasureAtoms((prev) => [...prev, ...newMeasures]);
            break;
          }
          case "OptimizationComplete": {
            setRunning(false);
            break;
          }
        }
      }
    };
    worker.current.postMessage({
      action: "Start",
      universe,
      schedule,
      tripId,
      optType: "V1",
    } as WorkerRequest);
  }, [tripId, universe, schedule, setMeasureAtoms]);

  const stopOptimization = useCallback(async () => {
    setRunning(false);
    if (!textAreaRef.current || !worker.current) {
      return;
    }
    textAreaRef.current.value += "Optimierung wird gestoppt...\n";
    worker.current?.terminate();
    worker.current = null;
    const universesToDestroy = simUniverses.current;
    if (universesToDestroy.length > 0) {
      textAreaRef.current.value += `Universen werden freigegeben: ${universesToDestroy.join(
        ", ",
      )}\n`;
      simUniverses.current = [];
      const destructionCalls = universesToDestroy.map((u) =>
        sendPaxMonDestroyUniverseRequest({ universe: u }),
      );
      await Promise.allSettled(destructionCalls);
      textAreaRef.current.value += "Universen freigegeben.\n";
    }
  }, []);

  const toggleOptimization = useCallback(() => {
    if (worker.current && running) {
      stopOptimization().catch(console.error);
    } else {
      startOptimization();
    }
  }, [startOptimization, stopOptimization, running]);

  // cancel on onmount
  useEffect(() => {
    return () => {
      stopOptimization().catch(console.error);
    };
  }, [stopOptimization]);

  return (
    <div
      className={cn(
        "max-w-7xl mx-auto pb-2",
        optimizationAvailable ? "visible" : "invisible",
      )}
    >
      <div className="flex justify-center my-2 gap-2">
        <button
          type="button"
          onClick={toggleOptimization}
          className={cn(
            "flex justify-center items-center gap-1 px-3 py-1 rounded text-sm",
            running
              ? "bg-db-red-300 text-db-red-100 cursor-wait"
              : "bg-db-red-500 hover:bg-db-red-600 text-white",
          )}
        >
          <SparklesIcon className="w-5 h-5" aria-hidden="true" />
          Maßnahmen zur Auslastungsreduktion ermitteln
        </button>
      </div>
      <div className={cn(showOptimizationDebugLog ? "block" : "hidden")}>
        <textarea
          ref={textAreaRef}
          rows={10}
          className="
                    mt-1
                    block
                    w-full
                    rounded-md
                    border-gray-300
                    shadow-sm
                    focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50
                  "
        />
      </div>
    </div>
  );
}

export default TripOptimization;
