import { Listbox, Transition } from "@headlessui/react";
import { AdjustmentsVerticalIcon, CheckIcon } from "@heroicons/react/20/solid";
import React, { Fragment } from "react";

import { ServiceClass } from "@/api/constants";

import { cn } from "@/lib/utils";

interface LabeledServiceClass {
  sc: ServiceClass;
  label: string;
}

const serviceClassOptions: LabeledServiceClass[] = [
  {
    sc: ServiceClass.ICE,
    label: "Hochgeschwindigkeitszüge",
  },
  { sc: ServiceClass.IC, label: "Fernzüge" },
  { sc: ServiceClass.COACH, label: "Fernbusse" },
  { sc: ServiceClass.N, label: "Nachtzüge" },
  { sc: ServiceClass.RE, label: "Regional-Express" },
  { sc: ServiceClass.RB, label: "Regionalbahnen" },
  { sc: ServiceClass.S, label: "S-Bahnen" },
  { sc: ServiceClass.U, label: "U-Bahnen" },
  { sc: ServiceClass.STR, label: "Straßenbahnen" },
  { sc: ServiceClass.BUS, label: "Busse" },
  { sc: ServiceClass.SHIP, label: "Schiffe" },
  { sc: ServiceClass.AIR, label: "Flugzeuge" },
  { sc: ServiceClass.OTHER, label: "Sonstige" },
];

export interface ServiceClassFilterProps {
  selectedServiceClasses: number[];
  setSelectedServiceClasses: React.Dispatch<React.SetStateAction<number[]>>;
  popupPosition: string;
}

function ServiceClassFilter({
  selectedServiceClasses,
  setSelectedServiceClasses,
  popupPosition,
}: ServiceClassFilterProps) {
  return (
    <Listbox
      value={selectedServiceClasses}
      onChange={setSelectedServiceClasses}
      multiple
    >
      <div className="relative">
        <Listbox.Button className="align-center mb-0.5 flex justify-center rounded-full bg-white p-2 text-black shadow-sm outline-0 dark:bg-gray-600 dark:text-gray-100">
          <AdjustmentsVerticalIcon className="h-5 w-5" aria-hidden="true" />
        </Listbox.Button>
        <Transition
          as={Fragment}
          leave="transition ease-in duration-100"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <Listbox.Options
            className={cn(
              "absolute z-20 mt-1 overflow-auto rounded-md bg-white py-1 text-base shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none sm:text-sm",
              popupPosition,
            )}
          >
            {serviceClassOptions.map((opt) => (
              <Listbox.Option
                key={opt.sc}
                value={opt.sc}
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
                      {opt.label}
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

export default ServiceClassFilter;
