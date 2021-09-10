import React, { useState } from "react";
import { useCombobox } from "downshift";
import { ChevronDownIcon, XIcon } from "@heroicons/react/solid";

import { PaxMonTripInfo } from "./api/protocol/motis/paxmon";
import { TripId, TripServiceInfo } from "./api/protocol/motis";
import { ServiceClass } from "./api/constants";
import { usePaxMonFindTripsQuery } from "./api/paxmon";
import TripView from "./TripView";

function filterTrips(trips: PaxMonTripInfo[]) {
  return trips.filter((trip) =>
    trip.tsi.service_infos.some(
      (si) =>
        si.clasz === ServiceClass.ICE ||
        si.clasz === ServiceClass.IC ||
        si.clasz === ServiceClass.OTHER
    )
  );
}

function shortTripName(tsi: TripServiceInfo) {
  const names = [
    ...new Set(
      tsi.service_infos.map((si) =>
        si.line ? `${si.name} [${si.train_nr}]` : si.name
      )
    ),
  ];
  return names.join(", ");
}

type TripPickerProps = {
  onTripPicked: (trip: TripId | null) => void;
};

function TripPicker({ onTripPicked }: TripPickerProps): JSX.Element {
  const [trainNr, setTrainNr] = useState<number>();
  const { data } = usePaxMonFindTripsQuery(trainNr);
  const tripList = filterTrips(data?.trips || []);

  const {
    isOpen,
    getToggleButtonProps,
    getLabelProps,
    getMenuProps,
    getInputProps,
    getComboboxProps,
    highlightedIndex,
    getItemProps,
    selectedItem,
    reset,
  } = useCombobox({
    items: tripList,
    itemToString: (item: PaxMonTripInfo | null) =>
      item !== null ? shortTripName(item.tsi) : "",
    onInputValueChange: ({ inputValue }) => {
      if (inputValue != undefined) {
        const parsed = parseInt(inputValue);
        if (!isNaN(parsed) || inputValue === "") {
          setTrainNr(parsed);
        }
      }
    },
    onSelectedItemChange: (changes) => {
      onTripPicked(changes.selectedItem?.tsi?.trip ?? null);
    },
  });

  return (
    <div className="relative flex items-center gap-2">
      <label {...getLabelProps()}>Trip:</label>
      <div {...getComboboxProps()} className="relative">
        <input
          {...getInputProps()}
          type="text"
          className="w-60 rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
        />
        {selectedItem ? (
          <button
            tabIndex={-1}
            onClick={() => reset()}
            aria-label="clear selection"
            className="absolute top-0 right-0 h-full px-2 flex items-center justify-center"
          >
            <XIcon className="h-4 w-4 text-gray-500" />
          </button>
        ) : (
          <button
            type="button"
            {...getToggleButtonProps()}
            aria-label="toggle menu"
            className="absolute top-0 right-0 h-full px-2 flex items-center justify-center"
          >
            <ChevronDownIcon className="h-4 w-4 text-gray-500" />
          </button>
        )}
      </div>
      <ul
        {...getMenuProps()}
        className={`${
          isOpen && tripList.length > 0 ? "" : "hidden"
        } absolute top-12 bg-white rounded-md shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none p-2`}
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
              <TripView tsi={item.tsi} format="Long" />
            </li>
          ))}
      </ul>
    </div>
  );
}

export default TripPicker;
