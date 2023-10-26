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
      <div className={cn("relative h-full w-full", className)}>
        <Listbox.Button className="relative w-full cursor-default rounded-lg bg-white py-2 pl-3 pr-10 text-left shadow-sm focus:outline-none focus-visible:border-indigo-500 focus-visible:ring-2 focus-visible:ring-white focus-visible:ring-opacity-75 focus-visible:ring-offset-2 focus-visible:ring-offset-orange-300 dark:bg-gray-700">
          <span className="block truncate">
            <LoadLevelLabel level={selectedLevel} />
          </span>
          <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
            <ChevronUpDownIcon
              className="h-5 w-5 text-gray-400"
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
          <Listbox.Options className="absolute z-20 mt-1 max-h-60 w-full overflow-auto rounded-md bg-white py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm">
            {loadLevels.map((level) => (
              <Listbox.Option
                key={level}
                value={level}
                className={({ active }) =>
                  cn(
                    "relative cursor-default select-none py-2 pl-10 pr-4",
                    active ? "bg-amber-100 text-amber-900" : "text-gray-900",
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
                        <CheckIcon className="h-5 w-5" aria-hidden="true" />
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
      <span className={`inline-block h-4 w-4 rounded-full ${lli.bgColor}`} />
      {lli.label}
    </div>
  );
}

export default LoadInput;
