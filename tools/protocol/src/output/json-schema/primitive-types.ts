import { BasicType } from "@/fbs/basic_types";
import { JSContext } from "@/output/json-schema/context";
import { JSONSchema } from "@/output/json-schema/types";

export function basicTypeToJS(ctx: JSContext, type: BasicType): JSONSchema {
  switch (type.sc) {
    case "bool":
      return { type: "boolean" };
    case "int": {
      const js: JSONSchema = { type: "integer" };
      if (type.unsigned) {
        js.minimum = 0;
      } else if (ctx.numberFormats) {
        if (type.bits === 32) {
          js.format = "int32";
        } else if (type.bits === 64) {
          js.format = "int64";
        }
      }
      if (ctx.strictIntTypes && type.bits < 64) {
        if (type.unsigned) {
          js.maximum = 2 ** type.bits - 1;
        } else {
          js.minimum = (-2) ** (type.bits - 1);
          js.maximum = 2 ** (type.bits - 1) - 1;
        }
      }
      return js;
    }
    case "float": {
      const js: JSONSchema = { type: "number" };
      if (ctx.numberFormats) {
        js.format = type.bits === 32 ? "float" : "double";
      }
      return js;
    }
    case "string":
      return { type: "string" };
  }
  throw new Error(`unhandled basic type: ${JSON.stringify(type)}`);
}
