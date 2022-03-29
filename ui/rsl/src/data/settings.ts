import { atomWithStorage } from "jotai/utils";

import { SectionLoadGraphPlotType } from "@/components/SectionLoadGraph";

export const sectionGraphPlotTypeAtom =
  atomWithStorage<SectionLoadGraphPlotType>(
    "sectionLoadGraphPlotType",
    "SimpleBox"
  );

export const showLegacyLoadForecastChartAtom = atomWithStorage<boolean>(
  "showLegacyLoadForecastChart",
  false
);

export const showLegacyMeasureTypesAtom = atomWithStorage<boolean>(
  "showLegacyMeasureTypes",
  false
);
