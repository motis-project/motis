export const INT_TYPES = [
  "byte",
  "ubyte",
  "short",
  "ushort",
  "int",
  "uint",
  "long",
  "ulong",
  "int8",
  "uint8",
  "int16",
  "uint16",
  "int32",
  "uint32",
  "int64",
  "uint64",
] as const;

export type IntTypeName = typeof INT_TYPES[number];

export const FLOAT_TYPES = ["float", "double", "float32", "float64"] as const;

export type FloatTypeName = typeof FLOAT_TYPES[number];

export type BasicType =
  | { sc: "bool"; type: "bool" }
  | { sc: "int"; type: IntTypeName }
  | { sc: "float"; type: FloatTypeName }
  | { sc: "string"; type: "string" };
