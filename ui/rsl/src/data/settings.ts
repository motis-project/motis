import { atomWithStorage } from "jotai/utils";

import { SectionLoadGraphPlotType } from "@/components/trips/SectionLoadGraph";

export const sectionGraphPlotTypeAtom =
  atomWithStorage<SectionLoadGraphPlotType>(
    "sectionLoadGraphPlotType",
    "SimpleBox",
  );

export const showLegacyLoadForecastChartAtom = atomWithStorage(
  "showLegacyLoadForecastChart",
  false,
);

export const showLegacyMeasureTypesAtom = atomWithStorage(
  "showLegacyMeasureTypes",
  false,
);

export const showOptimizationDebugLogAtom = atomWithStorage(
  "showOptimizationDebugLog",
  false,
);

export const showCapacityInfoAtom = atomWithStorage("showCapacityInfo", false);
