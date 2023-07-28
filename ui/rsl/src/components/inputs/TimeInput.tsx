import { format, parse } from "date-fns";
import React, { ForwardedRef, ReactElement, forwardRef } from "react";

export interface TimeInputProps
  extends Omit<
    React.ComponentPropsWithoutRef<"input">,
    "type" | "value" | "onChange"
  > {
  value: Date;
  onChange: (date: Date) => void;
}

const dateTimeFormat = "yyyy-MM-dd'T'HH:mm";

// TODO: time zones
// TODO: custom time picker component (datetime-local requires firefox 93+)

const TimeInput = forwardRef<HTMLInputElement, TimeInputProps>(
  function timeInput(
    componentProps: TimeInputProps,
    ref: ForwardedRef<HTMLInputElement>,
  ): ReactElement {
    // https://github.com/yannickcr/eslint-plugin-react/issues/3140
    const { value, onChange, ...restProps } = componentProps;
    let textValue = "";
    try {
      textValue = format(value, dateTimeFormat);
    } catch (ex) {
      console.log("TimeInput: invalid value:", value, ex);
    }
    return (
      <input
        type="datetime-local"
        {...restProps}
        ref={ref}
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
  },
);

export default TimeInput;
