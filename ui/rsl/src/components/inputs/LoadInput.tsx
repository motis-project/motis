import { Listbox, Transition } from "@headlessui/react";
import { CheckIcon, ChevronUpDownIcon } from "@heroicons/react/20/solid";
import React, { Fragment } from "react";

import { LoadLevel } from "@/api/protocol/motis/paxforecast";

import { loadLevelInfos } from "@/util/loadLevelInfos";

import { cn } from "@/lib/utils";

export const allLoadLevels: LoadLevel[] = ["Unknown", "Low", "NoSeats", "Full"];
export const knownLoadLevels: LoadLevel[] = ["Low", "NoSeats", "Full"];
export const highLoadLevels: LoadLevel[] = ["NoSeats", "Full"];
export const lowLoadLevels: LoadLevel[] = ["Low", "NoSeats"];
export const lowOrUnknownLoadLevels: LoadLevel[] = [
  "Unknown",
  "Low",
  "NoSeats",
];

export interface LoadInputProps {
  loadLevels?: LoadLevel[];
  selectedLevel: LoadLevel;
  onLevelSelected: (level: LoadLevel) => void;
  className?: string;
}

function LoadInput({
  loadLevels = allLoadLevels,
  selectedLevel,
  onLevelSelected,
  className = "",
}: LoadInputProps) {
  return (
    <Listbox value={selectedLevel} onChange={onLevelSelected}>
      <div className={cn("relative w-full h-full", className)}>
        <Listbox.Button className="relative w-full py-2 pl-3 pr-10 text-left bg-white dark:bg-gray-700 rounded-lg shadow-sm cursor-default focus:outline-none focus-visible:ring-2 focus-visible:ring-opacity-75 focus-visible:ring-white focus-visible:ring-offset-orange-300 focus-visible:ring-offset-2 focus-visible:border-indigo-500">
          <span className="block truncate">
            <LoadLevelLabel level={selectedLevel} />
          </span>
          <span className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
            <ChevronUpDownIcon
              className="w-5 h-5 text-gray-400"
              aria-hidden="true"
            />
          </span>
        </Listbox.Button>
        <Transition
          as={Fragment}
          leave="transition ease-in duration-100"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <Listbox.Options className="absolute z-20 w-full py-1 mt-1 overflow-auto text-base bg-white rounded-md shadow-lg max-h-60 ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
            {loadLevels.map((level) => (
              <Listbox.Option
                key={level}
                value={level}
                className={({ active }) =>
                  cn(
                    "cursor-default select-none relative py-2 pl-10 pr-4",
                    active ? "text-amber-900 bg-amber-100" : "text-gray-900",
                  )
                }
              >
                {({ selected, active }) => (
                  <>
                    <span
                      className={cn(
                        "block truncate",
                        selected ? "font-medium" : "font-normal",
                      )}
                    >
                      <LoadLevelLabel level={level} />
                    </span>
                    {selected ? (
                      <span
                        className={cn(
                          "absolute inset-y-0 left-0 flex items-center pl-3",
                          active ? "text-amber-600" : "text-amber-600",
                        )}
                      >
                        <CheckIcon className="w-5 h-5" aria-hidden="true" />
                      </span>
                    ) : null}
                  </>
                )}
              </Listbox.Option>
            ))}
          </Listbox.Options>
        </Transition>
      </div>
    </Listbox>
  );
}

interface LoadLevelLabelProps {
  level: LoadLevel;
}

function LoadLevelLabel({ level }: LoadLevelLabelProps) {
  const lli = loadLevelInfos[level];
  return (
    <div className="flex items-center gap-2">
      <span className={`inline-block w-4 h-4 rounded-full ${lli.bgColor}`} />
      {lli.label}
    </div>
  );
}

export default LoadInput;
