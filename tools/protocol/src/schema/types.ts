import {
  AstEnumVal,
  AstMetadata,
  AstNumericLiteral,
  AstRefLiteral,
  AstScalarLiteral,
} from "@/fbs/ast";
import { BasicType } from "@/fbs/basic_types";

export type Namespace = string[];

export interface TypeRef {
  input: string;
  currentNamespace: string[];
  resolvedFqtn: string[];
}

export interface TypeBase {
  name: string;
  ns: Namespace;
  fbsFile: string;
  relativeFbsFile: string;
  metadata: AstMetadata;
}

export interface EnumType extends TypeBase {
  type: "enum";
  values: AstEnumVal[];
}

export interface UnionValue {
  typeRef: TypeRef;
  value: AstNumericLiteral | null;
  metadata: AstMetadata;
}

export interface UnionType extends TypeBase {
  type: "union";
  values: UnionValue[];
}

export type FieldType =
  | { c: "basic"; type: BasicType }
  | { c: "vector"; type: FieldType }
  | { c: "custom"; type: TypeRef };

export interface TableField {
  name: string;
  type: FieldType;
  defaultValue: AstScalarLiteral | AstRefLiteral | null;
  metadata: AstMetadata;
}

export interface TableType extends TypeBase {
  type: "table";
  isStruct: boolean;
  fields: TableField[];
}

export type SchemaType = EnumType | UnionType | TableType;

export interface SchemaTypes {
  types: Map<string, SchemaType>;
  rootType: TypeRef | undefined;
}
