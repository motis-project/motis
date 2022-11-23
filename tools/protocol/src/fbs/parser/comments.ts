import {
  char,
  choice,
  endOfInput,
  everyCharUntil,
  sequenceOf,
  str,
} from "arcsecond";

export const lineComment = sequenceOf([
  str("//"),
  everyCharUntil(choice([char("\n"), endOfInput])),
]).map((x) => {
  return { t: "comment", value: x[1] };
});

export const blockComment = sequenceOf([
  str("/*"),
  everyCharUntil(str("*/")),
  str("*/"),
]).map((x) => {
  return { t: "comment", value: x[1] };
});
