import { BasicType } from "@/fbs/basic_types";

export type AstCustomType = { c: "custom"; type: string };
export type AstFieldType =
  | { c: "basic"; type: BasicType }
  | { c: "vector"; type: AstFieldType }
  | AstCustomType;

export type AstBooleanLiteral = { t: "boolean"; value: boolean };
export type AstNumericLiteral = { t: "numeric"; value: number };
export type AstStringLiteral = { t: "string"; value: string };
export type AstRefLiteral = { t: "ref"; value: string };
export type AstScalarLiteral = AstBooleanLiteral | AstNumericLiteral;
export type AstLiteral = AstScalarLiteral | AstStringLiteral;

export type AstInclude = { t: "include"; value: string };
export type AstNamespace = { t: "namespace"; value: string };

export type AstAttribute = { id: string; value: AstLiteral | null };
export type AstMetadata = null | AstAttribute[];

export type AstEnumVal = {
  id: string;
  value: AstNumericLiteral | null;
  metadata: AstMetadata;
};

export type AstEnum = {
  t: "enum";
  name: string;
  baseType: AstFieldType;
  metadata: AstMetadata;
  values: AstEnumVal[];
};

export type AstUnionVal = AstEnumVal;

export type AstUnion = {
  t: "union";
  name: string;
  metadata: AstMetadata;
  values: AstUnionVal[];
};

export type AstField = {
  name: string;
  type: AstFieldType;
  defaultValue: AstScalarLiteral | AstRefLiteral | null;
  metadata: AstMetadata;
};

export type AstTable = {
  t: "table";
  isStruct: boolean;
  name: string;
  metadata: AstMetadata;
  fields: AstField[];
};

export type AstRootType = { t: "root_type"; type: AstCustomType };

export type AstAttributeDecl = { t: "attribute"; id: string };

export type AstTopLevel =
  | AstInclude
  | AstNamespace
  | AstEnum
  | AstUnion
  | AstTable
  | AstRootType
  | AstAttributeDecl;
