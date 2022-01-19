import { format, parse } from "date-fns";

export interface TimeInputProps
  extends Omit<React.ComponentPropsWithoutRef<"input">, "value" | "onChange"> {
  value: Date;
  onChange: (date: Date) => void;
}

const dateTimeFormat = "yyyy-MM-dd'T'HH:mm";

// TODO: time zones
// TODO: custom time picker component (datetime-local requires firefox 93+)

function TimeInput({
  value,
  onChange,
  ...restProps
}: TimeInputProps): JSX.Element {
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
