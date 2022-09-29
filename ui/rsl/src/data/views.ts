import { atom } from "jotai";

export type MainPage = "trips" | "groups" | "stats";

export const mainPageAtom = atom<MainPage>("trips");
export const showSimPanelAtom = atom(false);
