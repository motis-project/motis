import React from "react";
import { format, parse } from "date-fns";

type TimeInputProps = {
  value: Date;
  onChange: (date: Date) => void;
};

const dateTimeFormat = "yyyy-MM-dd'T'HH:mm";

// TODO: time zones
// TODO: custom time picker component (datetime-local requires firefox 93+)

function TimeInput({ value, onChange }: TimeInputProps): JSX.Element {
  let textValue = "";
  try {
    textValue = format(value, dateTimeFormat);
  } catch (ex) {
    console.log("TimeInput: invalid value:", value, ex);
  }
  return (
    <input
      type="datetime-local"
      className="w-72 rounded-md border-gray-300 shadow-sm focus:border-blue-300 focus:ring focus:ring-blue-200 focus:ring-opacity-50"
      value={textValue}
      onChange={(e) => {
        try {
          const ts = parse(e.target.value, dateTimeFormat, new Date());
          onChange(ts);
        } catch (ex) {
          console.log("invalid date time input:", e.target.value, ex);
        }
      }}
    />
  );
}

export default TimeInput;
