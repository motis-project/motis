import { atom } from "jotai";

export const mostRecentlySelectedGroupAtom = atom<number | undefined>(
  undefined,
);
