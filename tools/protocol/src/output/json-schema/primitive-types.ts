import { BasicType, IntTypeName } from "../../fbs/basic_types";
import { JSContext } from "./context";

const jsIntTypes: Record<IntTypeName, any> = {
  byte: { type: "integer", minimum: -128, maximum: 127 },
  int8: { type: "integer", minimum: -128, maximum: 127 },
  ubyte: { type: "integer", minimum: 0, maximum: 255 },
  uint8: { type: "integer", minimum: 0, maximum: 255 },
  short: { type: "integer", minimum: -32768, maximum: 32767 },
  int16: { type: "integer", minimum: -32768, maximum: 32767 },
  ushort: { type: "integer", minimum: 0, maximum: 65535 },
  uint16: { type: "integer", minimum: 0, maximum: 65535 },
  int: { type: "integer", minimum: -2_147_483_648, maximum: 2_147_483_647 },
  int32: { type: "integer", minimum: -2_147_483_648, maximum: 2_147_483_647 },
  uint: { type: "integer", minimum: 0, maximum: 4_294_967_295 },
  uint32: { type: "integer", minimum: 0, maximum: 4_294_967_295 },
  long: {
    type: "integer",
    /*minimum: -9_223_372_036_854_775_808,
    maximum: 9_223_372_036_854_775_807,*/
  },
  int64: {
    type: "integer",
    /*minimum: -9_223_372_036_854_775_808,
    maximum: 9_223_372_036_854_775_807,*/
  },
  ulong: {
    type: "integer",
    minimum: 0 /*, maximum: 18_446_744_073_709_551_615*/,
  },
  uint64: {
    type: "integer",
    minimum: 0 /*, maximum: 18_446_744_073_709_551_615*/,
  },
};

export function basicTypeToJS(ctx: JSContext, type: BasicType): any {
  switch (type.sc) {
    case "bool":
      return { type: "boolean" };
    case "int": {
      if (ctx.strictIntTypes) {
        return jsIntTypes[type.type];
      } else {
        const js: any = { type: "integer" };
        if (type.type.startsWith("u")) {
          js.minimum = 0;
        }
        return js;
      }
    }
    case "float":
      return { type: "number" };
    case "string":
      return { type: "string" };
  }
  throw new Error(`unhandled basic type: ${JSON.stringify(type)}`);
}
