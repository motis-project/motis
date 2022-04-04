import { ChevronDownIcon } from "@heroicons/react/solid";
import { useCombobox } from "downshift";
import { useState } from "react";

import { Station } from "@/api/protocol/motis";

import { useStationGuesserQuery } from "@/api/guesser";

type StationPickerProps = {
  onStationPicked: (station: Station | undefined) => void;
  clearOnPick: boolean;
  initialStation?: Station | undefined;
};

function StationPicker({
  onStationPicked,
  clearOnPick,
  initialStation,
}: StationPickerProps): JSX.Element {
  const [input, setInput] = useState("");
  const { data } = useStationGuesserQuery(
    { input, guess_count: 10 },
    { keepPreviousData: true }
  );
  const stationList = data?.guesses || [];

  const {
    isOpen,
    getToggleButtonProps,
    getMenuProps,
    getInputProps,
    getComboboxProps,
    highlightedIndex,
    getItemProps,
    reset,
  } = useCombobox({
    items: stationList,
    itemToString: (item: Station | null) => (item !== null ? item.name : ""),
    initialSelectedItem: initialStation ?? null,
    onInputValueChange: ({ inputValue }) => {
      if (inputValue != undefined) {
        setInput(inputValue);
      }
    },
    onSelectedItemChange: (changes) => {
      if (changes.selectedItem != null || !clearOnPick) {
        onStationPicked(changes.selectedItem ?? undefined);
      }
      if (changes.selectedItem != null && clearOnPick) {
        reset();
      }
    },
  });

  return (
    <div className="relative flex">
      <div {...getComboboxProps()} className="relative w-full">
        <input
          {...getInputProps()}
          type="text"
          className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
        />
        <button
          type="button"
          {...getToggleButtonProps()}
          aria-label="toggle menu"
          className="absolute top-0 right-0 h-full px-2 flex items-center justify-center"
        >
          <ChevronDownIcon className="h-4 w-4 text-gray-500" />
        </button>
      </div>
      <ul
        {...getMenuProps()}
        className={`${
          isOpen && stationList.length > 0 ? "" : "hidden"
        } absolute z-50 top-12 bg-white rounded-md shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none p-2`}
      >
        {isOpen &&
          stationList.map((item, index) => (
            <li
              className={`${
                highlightedIndex === index
                  ? "bg-blue-500 text-white"
                  : "text-gray-900"
              } group flex items-center w-full p-2 rounded-md text-sm select-none cursor-pointer`}
              key={index}
              {...getItemProps({ item, index })}
            >
              {item.name}
            </li>
          ))}
      </ul>
    </div>
  );
}

export default StationPicker;
