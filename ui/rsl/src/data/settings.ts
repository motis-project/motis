import { atomWithStorage } from "jotai/utils";

import { SectionLoadGraphPlotType } from "@/components/SectionLoadGraph";

export const sectionGraphPlotTypeAtom =
  atomWithStorage<SectionLoadGraphPlotType>(
    "sectionLoadGraphPlotType",
    "SimpleBox"
  );
