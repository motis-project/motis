import { BasicType } from "@/fbs/basic_types";

export interface AstCustomType {
  c: "custom";
  type: string;
}
export type AstFieldType =
  | { c: "basic"; type: BasicType }
  | { c: "vector"; type: AstFieldType }
  | AstCustomType;

export interface AstBooleanLiteral {
  t: "boolean";
  value: boolean;
}
export interface AstNumericLiteral {
  t: "numeric";
  value: number;
}
export interface AstStringLiteral {
  t: "string";
  value: string;
}
export interface AstRefLiteral {
  t: "ref";
  value: string;
}
export type AstScalarLiteral = AstBooleanLiteral | AstNumericLiteral;
export type AstLiteral = AstScalarLiteral | AstStringLiteral;

export interface AstInclude {
  t: "include";
  value: string;
}
export interface AstNamespace {
  t: "namespace";
  value: string;
}

export interface AstAttribute {
  id: string;
  value: AstLiteral | null;
}
export type AstMetadata = null | AstAttribute[];

export interface AstEnumVal {
  id: string;
  value: AstNumericLiteral | null;
  metadata: AstMetadata;
}

export interface AstEnum {
  t: "enum";
  name: string;
  baseType: AstFieldType;
  metadata: AstMetadata;
  values: AstEnumVal[];
}

export type AstUnionVal = AstEnumVal;

export interface AstUnion {
  t: "union";
  name: string;
  metadata: AstMetadata;
  values: AstUnionVal[];
}

export interface AstField {
  name: string;
  type: AstFieldType;
  defaultValue: AstScalarLiteral | AstRefLiteral | null;
  metadata: AstMetadata;
}

export interface AstTable {
  t: "table";
  isStruct: boolean;
  name: string;
  metadata: AstMetadata;
  fields: AstField[];
}

export interface AstRootType {
  t: "root_type";
  type: AstCustomType;
}

export interface AstAttributeDecl {
  t: "attribute";
  id: string;
}

export type AstTopLevel =
  | AstInclude
  | AstNamespace
  | AstEnum
  | AstUnion
  | AstTable
  | AstRootType
  | AstAttributeDecl;
