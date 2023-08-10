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

export type IntTypeName = (typeof INT_TYPES)[number];

export interface IntTypeProperties {
  bits: 8 | 16 | 32 | 64;
  unsigned: boolean;
}

export const INT_TYPE_PROPERTIES: Record<IntTypeName, IntTypeProperties> = {
  byte: { bits: 8, unsigned: false },
  int8: { bits: 8, unsigned: false },
  ubyte: { bits: 8, unsigned: true },
  uint8: { bits: 8, unsigned: true },
  short: { bits: 16, unsigned: false },
  int16: { bits: 16, unsigned: false },
  ushort: { bits: 16, unsigned: true },
  uint16: { bits: 16, unsigned: true },
  int: { bits: 32, unsigned: false },
  int32: { bits: 32, unsigned: false },
  uint: { bits: 32, unsigned: true },
  uint32: { bits: 32, unsigned: true },
  long: { bits: 64, unsigned: false },
  int64: { bits: 64, unsigned: false },
  ulong: { bits: 64, unsigned: true },
  uint64: { bits: 64, unsigned: true },
};

export const FLOAT_TYPES = ["float", "double", "float32", "float64"] as const;

export type FloatTypeName = (typeof FLOAT_TYPES)[number];

export interface FloatTypeProperties {
  bits: 32 | 64;
}

export const FLOAT_TYPE_PROPERTIES: Record<FloatTypeName, FloatTypeProperties> =
  {
    float: { bits: 32 },
    float32: { bits: 32 },
    double: { bits: 64 },
    float64: { bits: 64 },
  };

export type BasicType =
  | { sc: "bool"; type: "bool" }
  | ({ sc: "int"; type: IntTypeName } & IntTypeProperties)
  | ({ sc: "float"; type: FloatTypeName } & FloatTypeProperties)
  | { sc: "string"; type: "string" };
