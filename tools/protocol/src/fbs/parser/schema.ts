import {
  Parser,
  char,
  choice,
  endOfInput,
  many,
  possibly,
  sequenceOf,
  str,
} from "arcsecond";

import {
  AstAttribute,
  AstAttributeDecl,
  AstEnum,
  AstEnumVal,
  AstField,
  AstInclude,
  AstMetadata,
  AstNamespace,
  AstRootType,
  AstTable,
  AstTopLevel,
  AstUnion,
  AstUnionVal,
} from "@/fbs/ast";
import {
  ident,
  identWithOptionalNamespace,
  literal,
  numericLiteral,
  scalarOrRefLiteral,
  stringConstant,
} from "@/fbs/parser/constants";
import {
  betweenBraces,
  betweenParens,
  commaSeparated,
  optionalWhitespaceOrComments,
  whitespaceOrComments,
  whitespaceSurrounded,
} from "@/fbs/parser/helpers";
import { customType, type } from "@/fbs/parser/types";

export const metadata: Parser<AstMetadata> = possibly(
  sequenceOf([
    optionalWhitespaceOrComments,
    betweenParens(
      commaSeparated(
        sequenceOf([
          identWithOptionalNamespace,
          possibly(
            sequenceOf([
              optionalWhitespaceOrComments,
              char(":"),
              optionalWhitespaceOrComments,
              literal,
            ]).map((x) => x[3]),
          ),
        ]).map(([id, value]): AstAttribute => {
          return { id, value };
        }),
      ),
    ),
  ]).map(([_, x]) => x),
).map((x) => x as AstMetadata);

export const include: Parser<AstInclude> = sequenceOf([
  str("include"),
  whitespaceOrComments,
  stringConstant,
  optionalWhitespaceOrComments,
  char(";"),
]).map((x) => {
  return { t: "include", value: x[2] };
});

export const namespaceDecl: Parser<AstNamespace> = sequenceOf([
  str("namespace"),
  whitespaceOrComments,
  identWithOptionalNamespace,
  optionalWhitespaceOrComments,
  char(";"),
]).map((x) => {
  return { t: "namespace", value: x[2] };
});

const enumOrUnionValueAssignment = sequenceOf([
  optionalWhitespaceOrComments,
  char("="),
  optionalWhitespaceOrComments,
  numericLiteral,
]).map((x) => x[3]);

export const enumValDecl: Parser<AstEnumVal> = sequenceOf([
  ident,
  possibly(enumOrUnionValueAssignment),
  metadata,
]).map(([id, value, metadata]) => {
  return { id, value, metadata };
});

export const enumDecl: Parser<AstEnum> = sequenceOf([
  str("enum"),
  whitespaceOrComments,
  ident,
  metadata,
  optionalWhitespaceOrComments,
  char(":"),
  optionalWhitespaceOrComments,
  type /* , metadata */,
  optionalWhitespaceOrComments,
  betweenBraces(commaSeparated(enumValDecl)),
]).map((x) => {
  return {
    t: "enum",
    name: x[2],
    metadata: x[3],
    baseType: x[7],
    values: x[9] as AstEnumVal[],
  };
});

export const unionValDecl: Parser<AstUnionVal> = sequenceOf([
  identWithOptionalNamespace,
  possibly(enumOrUnionValueAssignment),
  metadata,
]).map(([id, value, metadata]) => {
  return { id, value, metadata };
});

export const unionDecl: Parser<AstUnion> = sequenceOf([
  str("union"),
  whitespaceOrComments,
  ident,
  metadata,
  optionalWhitespaceOrComments,
  betweenBraces(commaSeparated(unionValDecl)),
]).map((x) => {
  return {
    t: "union",
    name: x[2],
    metadata: x[3],
    values: x[5] as AstEnumVal[],
  };
});

export const fieldDecl: Parser<AstField> = sequenceOf([
  ident,
  optionalWhitespaceOrComments,
  char(":"),
  optionalWhitespaceOrComments,
  type,
  optionalWhitespaceOrComments,
  possibly(
    sequenceOf([
      char("="),
      optionalWhitespaceOrComments,
      scalarOrRefLiteral,
      optionalWhitespaceOrComments,
    ]).map((x) => x[2]),
  ),
  metadata,
  char(";"),
]).map((x) => {
  return { name: x[0], type: x[4], defaultValue: x[6], metadata: x[7] };
});

export const typeDecl: Parser<AstTable> = sequenceOf([
  choice([str("table"), str("struct")]),
  whitespaceOrComments,
  ident,
  metadata,
  optionalWhitespaceOrComments,
  betweenBraces(many(whitespaceSurrounded(fieldDecl))),
]).map((x) => {
  return {
    t: "table",
    isStruct: x[0] === "struct",
    name: x[2],
    metadata: x[3],
    fields: x[5] as AstField[],
  };
});

export const rootTypeDecl: Parser<AstRootType> = sequenceOf([
  str("root_type"),
  whitespaceOrComments,
  customType,
  optionalWhitespaceOrComments,
  char(";"),
]).map((x) => {
  return { t: "root_type", type: x[2] };
});

export const attributeDecl: Parser<AstAttributeDecl> = sequenceOf([
  str("attribute"),
  whitespaceOrComments,
  choice([ident, stringConstant]),
  optionalWhitespaceOrComments,
  char(";"),
]).map((x) => {
  return { t: "attribute", id: x[2] };
});

export const schema: Parser<AstTopLevel[]> = sequenceOf([
  many(
    whitespaceSurrounded(
      choice([
        include,
        namespaceDecl,
        enumDecl,
        unionDecl,
        typeDecl,
        rootTypeDecl,
        attributeDecl,
      ]),
    ),
  ),
  endOfInput,
]).map((x) => x[0] as AstTopLevel[]);
