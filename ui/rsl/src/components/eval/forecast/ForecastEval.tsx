import { Download, Rocket, XOctagon } from "lucide-react";
import { ReactNode, useCallback, useEffect, useRef, useState } from "react";

import { getApiEndpoint } from "@/api/endpoint.ts";

import { formatFileNameTime } from "@/util/dateFormat.ts";
import { saveAsCSV } from "@/util/download.ts";

import ForecastEvalWorker from "@/components/eval/forecast/worker.ts?worker";
import {
  EvalResult,
  WorkerRequest,
  WorkerUpdate,
} from "@/components/eval/forecast/workerMessages.ts";
import { Button } from "@/components/ui/button.tsx";
import { Progress } from "@/components/ui/progress.tsx";

import { cn } from "@/lib/utils.ts";

function getDownloadFileName(
  evalResult: EvalResult,
  prefix: string,
  suffix = ".csv",
) {
  return `${prefix}_${formatFileNameTime(evalResult.intervalStart)}_${formatFileNameTime(evalResult.intervalEnd)}${suffix}`;
}

export function ForecastEval(): ReactNode {
  const worker = useRef<Worker | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMax, setProgressMax] = useState(0);
  const [evalResult, setEvalResult] = useState<EvalResult | null>(null);

  const startEvaluation = useCallback(() => {
    setRunning(true);
    setProgress(0);
    if (!worker.current) {
      worker.current = new ForecastEvalWorker();
      worker.current.onmessage = (msg) => {
        const update = msg.data as WorkerUpdate;
        switch (update.type) {
          case "TripCount":
            setProgressMax(update.totalTrips);
            break;
          case "TripInfo":
            setProgress(update.progress);
            break;
          case "Done":
            setProgress(update.progress);
            setRunning(false);
            setEvalResult(update.result);
            break;
        }
      };
    }
    worker.current?.postMessage({
      action: "Start",
      apiEndpoint: getApiEndpoint(),
    } as WorkerRequest);
  }, []);

  const stopEvaluation = useCallback(() => {
    setRunning(false);
    if (!worker.current) {
      return;
    }
    worker.current?.terminate();
    worker.current = null;
  }, []);

  const toggleEvaluation = useCallback(() => {
    if (worker.current && running) {
      stopEvaluation();
    } else {
      startEvaluation();
    }
  }, [startEvaluation, stopEvaluation, running]);

  // cancel on onmount
  useEffect(() => {
    return () => {
      stopEvaluation();
    };
  }, [stopEvaluation]);

  const downloadTripCsv = useCallback(() => {
    if (!evalResult) {
      return;
    }
    saveAsCSV(evalResult.tripCsv, getDownloadFileName(evalResult, "trip_eval"));
  }, [evalResult]);

  const downloadSectionCsv = useCallback(() => {
    if (!evalResult) {
      return;
    }
    saveAsCSV(
      evalResult.sectionCsv,
      getDownloadFileName(evalResult, "section_eval"),
    );
  }, [evalResult]);

  return (
    <div className="grow overflow-y-auto p-3">
      <h1 className="text-xl font-semibold">
        Vergleich der RSL-Prognose mit Reisendenzähldaten
      </h1>
      <div className="my-4 flex gap-2">
        <div className="min-w-64">
          <Button
            variant={running ? "destructive" : "default"}
            className={cn("gap-1", running && "cursor-wait")}
            onClick={toggleEvaluation}
          >
            {running ? (
              <>
                <XOctagon className="h-5 w-5" aria-hidden="true" /> Auswertung
                abbrechen
              </>
            ) : (
              <>
                <Rocket className="h-5 w-5" aria-hidden="true" /> Auswertung
                erstellen
              </>
            )}
          </Button>
        </div>
        <Button
          disabled={evalResult == null}
          className="gap-1"
          onClick={downloadTripCsv}
        >
          <Download className="h-5 w-5" aria-hidden="true" />
          Auswertung nach Zügen (CSV)
        </Button>
        <Button
          disabled={evalResult == null}
          className="gap-1"
          onClick={downloadSectionCsv}
        >
          <Download className="h-5 w-5" aria-hidden="true" />
          Auswertung nach Fahrtabschnitten (CSV)
        </Button>
      </div>
      <div className="mb-8">
        <Progress
          value={progressMax != 0 ? (progress / progressMax) * 100 : 0}
        />
      </div>
    </div>
  );
}
