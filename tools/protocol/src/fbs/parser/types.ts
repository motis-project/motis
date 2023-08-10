import { Parser, choice, recursiveParser, str } from "arcsecond";

import { AstCustomType, AstFieldType } from "@/fbs/ast";
import {
  BasicType,
  FLOAT_TYPES,
  FLOAT_TYPE_PROPERTIES,
  FloatTypeName,
  INT_TYPES,
  INT_TYPE_PROPERTIES,
  IntTypeName,
} from "@/fbs/basic_types";
import { identWithOptionalNamespace } from "@/fbs/parser/constants";
import { betweenBrackets } from "@/fbs/parser/helpers";

export const boolType: Parser<BasicType> = str("bool").map(() => {
  return { sc: "bool", type: "bool" };
});

export const stringType: Parser<BasicType> = str("string").map(() => {
  return { sc: "string", type: "string" };
});

export const intType: Parser<BasicType> = choice(INT_TYPES.map(str)).map(
  (x) => {
    return { sc: "int", type: x, ...INT_TYPE_PROPERTIES[x as IntTypeName] };
  },
);

export const floatType: Parser<BasicType> = choice(FLOAT_TYPES.map(str)).map(
  (x) => {
    return {
      sc: "float",
      type: x,
      ...FLOAT_TYPE_PROPERTIES[x as FloatTypeName],
    };
  },
);

export const basicType: Parser<BasicType> = choice([
  boolType,
  stringType,
  intType,
  floatType,
]);

export const customType: Parser<AstCustomType> = identWithOptionalNamespace.map(
  (x) => {
    return { c: "custom" as const, type: x };
  },
);

export const type: Parser<AstFieldType> = recursiveParser(() =>
  choice([
    basicType.map((x) => {
      return { c: "basic" as const, type: x };
    }),
    betweenBrackets(type).map((x) => {
      return { c: "vector" as const, type: x as AstFieldType };
    }),
    customType,
  ]),
);
