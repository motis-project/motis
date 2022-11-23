import { choice, Parser, recursiveParser, str } from "arcsecond";
import { BasicType, FLOAT_TYPES, INT_TYPES } from "../basic_types";
import { AstCustomType, AstFieldType } from "../ast";
import { betweenBrackets } from "./helpers";
import { identWithOptionalNamespace } from "./constants";

export const boolType: Parser<BasicType> = str("bool").map(() => {
  return { sc: "bool", type: "bool" };
});

export const stringType: Parser<BasicType> = str("string").map(() => {
  return { sc: "string", type: "string" };
});

export const intType: Parser<BasicType> = choice(INT_TYPES.map(str)).map(
  (x) => {
    return { sc: "int", type: x };
  }
);

export const floatType: Parser<BasicType> = choice(FLOAT_TYPES.map(str)).map(
  (x) => {
    return { sc: "float", type: x };
  }
);

export const basicType: Parser<BasicType> = choice([
  boolType,
  stringType,
  intType,
  floatType,
]);

export const customType: Parser<AstCustomType> = identWithOptionalNamespace.map(
  (x) => {
    return { c: "custom", type: x };
  }
);

export const type: Parser<AstFieldType> = recursiveParser(() =>
  choice([
    basicType.map((x) => {
      return { c: "basic", type: x };
    }),
    betweenBrackets(type).map((x) => {
      return { c: "vector", type: x as AstFieldType };
    }),
    customType,
  ])
);
