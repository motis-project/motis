import { ChevronDownIcon, XMarkIcon } from "@heroicons/react/20/solid";
import { useCombobox } from "downshift";
import { useState } from "react";

import { Station } from "@/api/protocol/motis";

import { useStationGuesserQuery } from "@/api/guesser";

interface StationPickerProps {
  onStationPicked: (station: Station | undefined) => void;
  clearOnPick: boolean;
  clearButton: boolean;
  initialStation?: Station | undefined;
}

function StationPicker({
  onStationPicked,
  clearOnPick,
  clearButton,
  initialStation,
}: StationPickerProps): JSX.Element {
  const [input, setInput] = useState("");
  const { data } = useStationGuesserQuery({ input, guess_count: 10 }, true);
  const stationList = data?.guesses ?? [];

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
      <div className="relative w-full">
        <input
          {...getInputProps()}
          type="text"
          className="w-full rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
        />
        {selectedItem && clearButton ? (
          <button
            tabIndex={-1}
            onClick={() => reset()}
            aria-label="clear selection"
            className="absolute right-0 top-0 flex h-full items-center justify-center px-2"
          >
            <XMarkIcon className="h-5 w-5 text-gray-500" />
          </button>
        ) : (
          <button
            type="button"
            {...getToggleButtonProps()}
            aria-label="toggle menu"
            className="absolute right-0 top-0 flex h-full items-center justify-center px-2"
          >
            <ChevronDownIcon className="h-5 w-5 text-gray-500" />
          </button>
        )}
      </div>
      <ul
        {...getMenuProps()}
        className={`${
          isOpen && stationList.length > 0 ? "" : "hidden"
        } absolute top-12 z-50 rounded-md bg-white p-2 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none`}
      >
        {isOpen &&
          stationList.map((item, index) => (
            <li
              className={`${
                highlightedIndex === index
                  ? "bg-blue-500 text-white"
                  : "text-gray-900"
              } group flex w-full cursor-pointer select-none items-center rounded-md p-2 text-sm`}
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
