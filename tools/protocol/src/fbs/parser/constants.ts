import {
  Parser,
  char,
  choice,
  everyCharUntil,
  regex,
  sepBy1,
  sequenceOf,
  str,
} from "arcsecond";

import {
  AstBooleanLiteral,
  AstLiteral,
  AstNumericLiteral,
  AstRefLiteral,
  AstScalarLiteral,
  AstStringLiteral,
} from "@/fbs/ast";

export const ident: Parser<string> = regex(/^[a-zA-Z_][a-zA-Z0-9_]*/);

export const identWithOptionalNamespace: Parser<string> = sepBy1(char("."))(
  ident,
).map((x) => x.join("."));

const booleanConstant = choice([str("true"), str("false")]);
export const booleanLiteral: Parser<AstBooleanLiteral> = booleanConstant.map(
  (x) => {
    return { t: "boolean", value: x === "true" };
  },
);

const sign = regex(/^[-+]?/);

const decIntegerConstant = regex(/^[-+]?[0-9]+/).map(Number.parseInt);

const hexIntegerConstant = regex(/^[-+]?0[xX][0-9a-fA-F]+/).map(
  Number.parseInt,
);

const decFloatConstant = regex(
  /^[-+]?((\.\d+)|(\d+\.\d*)|(\d+))([eE][-+]?\d+)?/,
).map(Number.parseFloat);

const specialFloatConstant = sequenceOf([
  sign,
  choice([str("nan"), str("inf"), str("infinity")]),
]).map(([s, t]) => {
  if (t === "nan") {
    return Number.NaN;
  } else {
    return s === "-" ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;
  }
});

export const integerConstant = choice([hexIntegerConstant, decIntegerConstant]);
export const floatConstant = choice([decFloatConstant, specialFloatConstant]);
export const numericConstant = choice([floatConstant, integerConstant]);

export const numericLiteral: Parser<AstNumericLiteral> = numericConstant.map(
  (x) => {
    return { t: "numeric", value: x };
  },
);

export const scalar: Parser<AstScalarLiteral> = choice([
  booleanLiteral,
  numericLiteral,
]);

export const scalarOrRefLiteral: Parser<AstScalarLiteral | AstRefLiteral> =
  choice([
    scalar,
    ident.map((x) => {
      return { t: "ref" as const, value: x };
    }),
  ]);

export const stringConstant = sequenceOf([
  char('"'),
  everyCharUntil(char('"')),
  char('"'),
]).map((x) => {
  return x[1];
});

export const stringLiteral: Parser<AstStringLiteral> = stringConstant.map(
  (x) => {
    return { t: "string", value: x };
  },
);

export const literal: Parser<AstLiteral> = choice([scalar, stringLiteral]);
