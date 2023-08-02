import { ChevronDownIcon, XMarkIcon } from "@heroicons/react/20/solid";
import { useCombobox } from "downshift";
import { useAtom } from "jotai";
import { useState } from "react";

import { TripServiceInfo } from "@/api/protocol/motis";
import { PaxMonTripInfo } from "@/api/protocol/motis/paxmon";

import { ServiceClass } from "@/api/constants";
import { usePaxMonFindTripsQuery } from "@/api/paxmon";

import { universeAtom } from "@/data/multiverse";

import TripServiceInfoView from "@/components/TripServiceInfoView";

import { cn } from "@/lib/utils";

function filterTrips(trips: PaxMonTripInfo[]) {
  return trips.filter(
    (trip) =>
      /* eslint-disable @typescript-eslint/no-unsafe-enum-comparison */
      trip.tsi.service_infos.some(
        (si) =>
          si.clasz === ServiceClass.ICE ||
          si.clasz === ServiceClass.IC ||
          si.clasz === ServiceClass.OTHER,
      ),
    /* eslint-enable */
  );
}

function shortTripName(tsi: TripServiceInfo) {
  const names = [
    ...new Set(tsi.service_infos.map((si) => `${si.category} ${si.train_nr}`)),
  ];
  return names[0] ?? "?";
}

interface TripPickerProps {
  onTripPicked: (trip: TripServiceInfo | undefined) => void;
  clearOnPick: boolean;
  longDistanceOnly: boolean;
  className?: string;
  initialTrip?: TripServiceInfo | undefined;
}

function TripPicker({
  onTripPicked,
  clearOnPick,
  longDistanceOnly,
  className = "",
  initialTrip,
}: TripPickerProps): JSX.Element {
  const [universe] = useAtom(universeAtom);
  const [trainNr, setTrainNr] = useState<number>();
  const { data } = usePaxMonFindTripsQuery(universe, trainNr, {
    keepPreviousData: true,
  });
  const tripList = longDistanceOnly
    ? filterTrips(data?.trips ?? [])
    : data?.trips ?? [];

  const initialSelectedItem = initialTrip
    ? {
        tsi: initialTrip,
        has_paxmon_data: true,
        all_edges_have_capacity_info: true,
        has_passengers: true,
      }
    : null;

  const {
    isOpen,
    getToggleButtonProps,
    getMenuProps,
    getInputProps,
    highlightedIndex,
    getItemProps,
    selectedItem,
    reset,
  } = useCombobox({
    items: tripList,
    itemToString: (item: PaxMonTripInfo | null) =>
      item !== null ? shortTripName(item.tsi) : "",
    initialSelectedItem,
    onInputValueChange: ({ inputValue }) => {
      if (inputValue != undefined) {
        const parsed = parseInt(inputValue);
        if (!isNaN(parsed) || inputValue === "") {
          setTrainNr(parsed);
        }
      }
    },
    onSelectedItemChange: (changes) => {
      if (changes.selectedItem != null || !clearOnPick) {
        onTripPicked(changes.selectedItem?.tsi);
      }
      if (changes.selectedItem != null && clearOnPick) {
        reset();
      }
    },
  });

  return (
    <div className={cn("relative flex", className)}>
      {/* <label {...getLabelProps()}>Trip:</label> */}
      <div className="relative w-full">
        <input
          {...getInputProps()}
          type="text"
          className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
        />
        {selectedItem ? (
          <button
            tabIndex={-1}
            onClick={() => reset()}
            aria-label="clear selection"
            className="absolute top-0 right-0 h-full px-2 flex items-center justify-center"
          >
            <XMarkIcon className="h-5 w-5 text-gray-500" />
          </button>
        ) : (
          <button
            type="button"
            {...getToggleButtonProps()}
            aria-label="toggle menu"
            className="absolute top-0 right-0 h-full px-2 flex items-center justify-center"
          >
            <ChevronDownIcon className="h-5 w-5 text-gray-500" />
          </button>
        )}
      </div>
      <ul
        {...getMenuProps()}
        className={`${
          isOpen && tripList.length > 0 ? "" : "hidden"
        } absolute w-64 z-50 top-12 bg-white rounded-md shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none p-2`}
      >
        {isOpen &&
          tripList.map((item, index) => (
            <li
              className={`${
                highlightedIndex === index
                  ? "bg-blue-500 text-white"
                  : "text-gray-900"
              } group flex items-center w-full p-2 rounded-md text-sm select-none cursor-pointer`}
              key={index}
              {...getItemProps({ item, index })}
            >
              <TripServiceInfoView tsi={item.tsi} format="Long" />
            </li>
          ))}
      </ul>
    </div>
  );
}

export default TripPicker;
