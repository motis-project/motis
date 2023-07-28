import { parseISO } from "date-fns";
import React, { ForwardedRef, ReactElement, forwardRef } from "react";

import { formatISODate } from "@/util/dateFormat";

export interface DatePickerProps
  extends Omit<
    React.ComponentPropsWithoutRef<"input">,
    "type" | "value" | "onChange"
  > {
  value: Date | undefined | null;
  onChange: (date: Date | null) => void;
}

const DatePicker = forwardRef<HTMLInputElement, DatePickerProps>(
  function datePicker(
    componentProps: DatePickerProps,
    ref: ForwardedRef<HTMLInputElement>,
  ): ReactElement {
    const { value, onChange, ...restProps } = componentProps;
    return (
      <input
        type="date"
        {...restProps}
        ref={ref}
        value={value && isValidDate(value) ? formatISODate(value) : ""}
        onChange={(e) => {
          // e.target.valueAsDate returns 00:00 UTC, we want 00:00 local time
          // e.target.value is in YYYY-MM-DD format if a date was selected or empty
          if (e.target.value === "") {
            onChange(null);
            return;
          }
          try {
            const date = parseISO(e.target.value);
            onChange(isValidDate(date) ? date : null);
          } catch (ex) {
            console.log("Could not parse date input:", e.target.value, ex);
          }
        }}
        className="block w-full text-sm rounded-md bg-white dark:bg-gray-700 border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
      />
    );
  },
);

function isValidDate(d: Date | undefined | null): boolean {
  return d instanceof Date && !isNaN(d.getTime());
}

export default DatePicker;
